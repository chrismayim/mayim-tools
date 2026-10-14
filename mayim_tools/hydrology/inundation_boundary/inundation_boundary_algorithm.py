"""Processing Toolbox algorithm: DEM + water level (or a range of levels),
optional bounding polygon and seed points in -> smoothed inundation
boundary polylines and polygons, a depth raster and a summary CSV out.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_inundation_boundary_core.py).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools.hydrology._common.d8_network import GridError, write_raster
from mayim_tools.hydrology._common.depression import (
    polygon_window,
    raster_info,
    read_window,
    window_geotransform,
)
from mayim_tools.hydrology.terrain_storage.terrain_storage_algorithm import (
    STYLE_DEPTH,
    STYLES_DIR,
    _StylePostProcessor,
)

from .core import (
    DEFAULT_MIN_CELLS,
    DEFAULT_NODATA,
    DEFAULT_SMOOTHING,
    MODES,
    run_inundation,
    water_levels,
)
from .export import write_summary_csv
from .report import write_report_docx

TOOL_VERSION = "0.3.0"

STYLE_DEPTH_PATH = STYLES_DIR / STYLE_DEPTH  # stage-storage "Depth at top stage"
STYLE_WSE_PATH = (
    Path(__file__).resolve().parent / "styles" / "water_surface_elevation.qml"
)

MODE_OPTIONS = [
    "All ground below the level (static bathtub)",
    "Only ground connected to the seed point(s)",
]

_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

POLYGON_FIELDS = [
    ("level_m", _D, "water level, m"),
    ("area_m2", _D, "area of the polygon (after clipping), m2"),
    ("area_ha", _D, "area of the polygon, ha"),
    ("perim_m", _D, "perimeter of the polygon (incl. islands), m"),
    ("n_islands", _I, "dry islands (holes) in the polygon"),
    ("lvl_vol_m3", _D, "volume below the level - all water at this level, m3"),
    (
        "lvl_area_m2",
        _D,
        "inundated area on the DEM cells - all water at this level, m2",
    ),
    ("lvl_maxdep", _D, "maximum depth at this level, m"),
    ("mode", _S, "all or connected"),
]

LINE_FIELDS = [
    ("level_m", _D, "water level, m"),
    ("length_m", _D, "length of the boundary line, m"),
    ("mode", _S, "all or connected"),
]


def _fields(specs) -> QgsFields:
    fields = QgsFields()
    for name, qtype, _ in specs:
        fields.append(QgsField(name, qtype))
    return fields


def _describe_destination(dest) -> str:
    """Readable location of a Processing output."""
    text = str(dest or "")
    if text.startswith("memory:") or "TEMPORARY_OUTPUT" in text or not text:
        return "Temporary layer (not saved)"
    return text


def _is_null(value) -> bool:
    return value is None or str(value) in ("", "NULL")


class InundationBoundaryAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    LEVEL_FROM = "LEVEL_FROM"
    LEVEL_TO = "LEVEL_TO"
    LEVEL_INTERVAL = "LEVEL_INTERVAL"
    BOUNDING = "BOUNDING"
    MODE = "MODE"
    SEEDS = "SEEDS"
    SMOOTHING = "SMOOTHING"
    MIN_CELLS = "MIN_CELLS"
    OUTPUT_LINES = "OUTPUT_LINES"
    OUTPUT_POLYGONS = "OUTPUT_POLYGONS"
    OUTPUT_DEPTH = "OUTPUT_DEPTH"
    OUTPUT_WSE = "OUTPUT_WSE"
    OUTPUT_TABLE = "OUTPUT_TABLE"
    OUTPUT_LEVEL_FOLDER = "OUTPUT_LEVEL_FOLDER"
    OUTPUT_DOCX = "OUTPUT_DOCX"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return InundationBoundaryAlgorithm()

    def name(self):
        return "inundation_boundary"

    def displayName(self):
        return "DEM Inundation Boundary"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        glossary = "\n".join(f"  {n}: {d}" for n, _, d in POLYGON_FIELDS)
        return (
            f"DEM Inundation Boundary (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tThe extent of water standing at a given elevation "
            "(bathtub inundation) - for one level, or a range of levels at an "
            "interval - as smoothed boundary polylines and matching polygons, "
            "with the area, volume and depth at each level.\n"
            "\n"
            "INPUTS:\tA DEM in a projected CRS in metres; a water level, or a "
            "range (from, to and interval; 'to' is always included). Optional: "
            "a bounding polygon that limits the analysis (all its features "
            "together; without it the whole DEM is used), and seed points for "
            "the connected mode.\n"
            "\n"
            "INUNDATED GROUND:\t'All ground below the level' floods every cell "
            "below the level, including separate hollows. 'Only ground "
            "connected to the seed point(s)' floods only ground that water at "
            "that level can reach from a seed (e.g. a river, dam or outlet "
            "point), through cells below the level (8-connected); hollows behind "
            "higher ground stay dry. Without a seed the lowest cell is used.\n"
            "\n"
            "BOUNDARY:\tThe water's edge is interpolated between DEM cell "
            "centres (marching squares) and rounded with Chaikin corner "
            "cutting; the polygons are built from the same lines (outer ring = "
            "water, inner rings = dry islands), so line and polygon match. "
            "Where the water reaches the bounding polygon, both are cut at its "
            "edge. Rings smaller than the minimum size (DEM noise) are "
            "dropped.\n"
            "\n"
            "OUTPUTS:\tBoundary polylines and inundation polygons (one feature "
            "per water body per level); the depth and water-surface elevation "
            "rasters at the highest level (the water-surface raster holds the "
            "level on every inundated cell - drape a cross-section profile "
            "over it together with the DEM); optionally both rasters for every "
            "level of a range, written to a folder as depth_<level>.tif and "
            "wse_<level>.tif; and a summary CSV per level (area and volume on "
            "the DEM cells, and the polygon area and perimeter); and an optional "
            "Word report with the method, settings, results per level, depth "
            "classes, a plan of the extent and the level-area-volume curves. "
            "The rasters "
            "follow the DEM cells, so their edge is stepped; the smoothed "
            "polygons show the interpolated water's edge.\n"
            "\n"
            "SEED POINTS:\tWater spreads outward from the seed cell, so a level "
            "at or below the ground at the seed floods nothing. Place seeds on "
            "the lowest ground of the water body (channel bed, dam basin), not "
            "on a bank.\n"
            "\n"
            "LIMITATIONS:\tA level-pool (bathtub) extent - no flow, no "
            "hydraulic gradient, no structures below DEM resolution. For "
            "flood lines along a river use a hydraulic model.\n"
            "\n"
            "POLYGON ATTRIBUTES:\n" + glossary
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.DEM, "DEM (projected, metres)")
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.LEVEL_FROM,
                "Water level, or start of the range (m)",
                type=QgsProcessingParameterNumber.Type.Double,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.LEVEL_TO,
                "End of the range (m; blank = single level)",
                type=QgsProcessingParameterNumber.Type.Double,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.LEVEL_INTERVAL,
                "Interval for a range (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.001,
                defaultValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.BOUNDING,
                "Bounding polygon (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE, "Inundated ground", options=MODE_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SEEDS,
                "Seed point(s) for the connected option (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPoint],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMOOTHING,
                "Smoothing passes (Chaikin; 0 = none)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                maxValue=8,
                defaultValue=DEFAULT_SMOOTHING,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_CELLS,
                "Drop rings smaller than (cells)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=DEFAULT_MIN_CELLS,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_LINES,
                "Inundation boundary (lines)",
                type=QgsProcessing.SourceType.TypeVectorLine,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_POLYGONS,
                "Inundation extent (polygons)",
                type=QgsProcessing.SourceType.TypeVectorPolygon,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_DEPTH,
                "Depth at the highest level",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_WSE,
                "Water-surface elevation at the highest level",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_LEVEL_FOLDER,
                "Depth and water-surface rasters for every level (folder)",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_TABLE,
                "Summary per level (CSV)",
                fileFilter="CSV files (*.csv)",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_DOCX,
                "Inundation report (Word)",
                fileFilter="Word documents (*.docx)",
                optional=True,
                createByDefault=False,
            )
        )

    # ------------------------------------------------------------------

    def _register_style(self, context, dest_id, style_path=STYLE_DEPTH_PATH, **options):
        """Loads the stage-storage 'Depth at top stage' style onto the depth
        raster (ramp stretched to this run's depths). The post-processor is
        kept alive on self - Processing holds only a weak reference."""
        if not dest_id or not context.willLoadLayerOnCompletion(dest_id):
            return
        details = context.layerToLoadOnCompletionDetails(dest_id)
        processor = _StylePostProcessor(style_path, **options)
        details.setPostProcessor(processor)
        self._post_processors.append(processor)

    @staticmethod
    def _write_raster(path, array, result, projection):
        write_raster(
            path,
            np.where(np.isnan(array), DEFAULT_NODATA, array).astype("float32"),
            result.window_gt,
            projection,
            DEFAULT_NODATA,
        )

    def _transform(self, source, crs, context):
        if source is None or source.sourceCrs() == crs:
            return None
        return QgsCoordinateTransform(
            source.sourceCrs(), crs, context.transformContext()
        )

    def _read_bounding(self, source, crs, context):
        """All bounding features as one geometry and as polygon rings."""
        if source is None:
            return None, None
        transform = self._transform(source, crs, context)
        parts = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            geom.convertToStraightSegment()
            if not geom.isGeosValid():
                geom = geom.makeValid()
            parts.append(geom)
        if not parts:
            return None, None
        union = QgsGeometry.unaryUnion(parts)
        union.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
        polys = union.asMultiPolygon() if union.isMultipart() else [union.asPolygon()]
        rings = [
            [[(p.x(), p.y()) for p in ring] for ring in poly if ring]
            for poly in polys
            if poly
        ]
        return union, rings

    def _read_seeds(self, source, crs, context):
        if source is None:
            return []
        transform = self._transform(source, crs, context)
        points = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            pts = geom.asMultiPoint() if geom.isMultipart() else [geom.asPoint()]
            points.extend((p.x(), p.y()) for p in pts)
        return points

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if dem_layer is None:
            raise QgsProcessingException("A DEM is required.")
        crs = dem_layer.crs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The DEM is in a geographic CRS (degrees). Areas, volumes and "
                "lengths need a projected CRS in metres - reproject the DEM first."
            )
        start = self.parameterAsDouble(parameters, self.LEVEL_FROM, context)
        raw_to = parameters.get(self.LEVEL_TO)
        end = (
            None
            if _is_null(raw_to)
            else self.parameterAsDouble(parameters, self.LEVEL_TO, context)
        )
        interval = self.parameterAsDouble(parameters, self.LEVEL_INTERVAL, context)
        mode = MODES[self.parameterAsEnum(parameters, self.MODE, context)]
        smoothing = self.parameterAsInt(parameters, self.SMOOTHING, context)
        min_cells = self.parameterAsDouble(parameters, self.MIN_CELLS, context)
        try:
            levels = water_levels(start, end, interval)
            shape, gt, projection, nodata = raster_info(dem_layer.source())
        except GridError as e:
            raise QgsProcessingException(str(e)) from e

        bounding_source = self.parameterAsSource(parameters, self.BOUNDING, context)
        seed_source = self.parameterAsSource(parameters, self.SEEDS, context)
        clip, bounding = self._read_bounding(bounding_source, crs, context)
        if bounding is not None:
            window = polygon_window(bounding, gt, shape, pad=2)
            if window is None:
                raise QgsProcessingException(
                    "The bounding polygon does not overlap the DEM."
                )
        else:
            window = (0, 0, shape[0], shape[1])
            if shape[0] * shape[1] > 25_000_000:
                feedback.pushWarning(
                    f"No bounding polygon: the whole DEM ({shape[1]} x {shape[0]} "
                    "cells) is processed, which may be slow. A bounding polygon "
                    "is recommended."
                )
        seeds = self._read_seeds(seed_source, crs, context)
        if mode == "all" and seeds:
            feedback.pushInfo("Seed points are only used by the connected option.")
        feedback.pushInfo(
            f"{len(levels)} level(s) from {min(levels):.3f} to {max(levels):.3f} m, "
            f"mode: {mode}, window {window[3]} x {window[2]} cells"
        )

        try:
            dem = read_window(dem_layer.source(), window)
            result = run_inundation(
                dem,
                nodata,
                window_geotransform(gt, window[0], window[1]),
                window,
                levels,
                bounding=bounding,
                mode=mode,
                seeds=seeds,
                smoothing=smoothing,
                min_cells=min_cells,
                is_canceled=feedback.isCanceled,
            )
        except GridError as e:
            raise QgsProcessingException(str(e)) from e
        if feedback.isCanceled():
            return {}
        for message in result.warnings:
            feedback.pushWarning(message)

        outputs = {}
        geometry = {}
        line_sink, line_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_LINES,
            context,
            _fields(LINE_FIELDS),
            Qgis.WkbType.MultiLineString,
            crs,
        )
        poly_sink, poly_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_POLYGONS,
            context,
            _fields(POLYGON_FIELDS),
            Qgis.WkbType.MultiPolygon,
            crs,
        )
        for i, lv in enumerate(result.levels):
            feedback.setProgress(int(100 * i / max(len(result.levels), 1)))
            total_area = total_perim = 0.0
            count = 0
            if line_sink is not None and lv.lines:
                parts = [[QgsPointXY(x, y) for x, y in ring] for ring in lv.lines]
                geom = QgsGeometry.fromMultiPolylineXY(parts)
                if clip is not None:
                    geom = geom.intersection(clip)
                    if QgsWkbTypes.flatType(geom.wkbType()) not in (
                        Qgis.WkbType.LineString,
                        Qgis.WkbType.MultiLineString,
                    ):
                        geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Line)
                if not geom.isEmpty():
                    geom.convertToMultiType()
                    feat = QgsFeature(_fields(LINE_FIELDS))
                    feat.setGeometry(geom)
                    feat["level_m"] = round(lv.level, 4)
                    feat["length_m"] = round(geom.length(), 2)
                    feat["mode"] = mode
                    line_sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
            for poly in lv.polygons:
                rings = [[QgsPointXY(x, y) for x, y in ring] for ring in poly]
                geom = QgsGeometry.fromPolygonXY(rings).makeValid()
                if clip is not None:
                    geom = geom.intersection(clip)
                if QgsWkbTypes.flatType(geom.wkbType()) not in (
                    Qgis.WkbType.Polygon,
                    Qgis.WkbType.MultiPolygon,
                ):
                    geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
                if geom.isEmpty():
                    continue
                geom.convertToMultiType()
                area, perim = geom.area(), geom.length()
                total_area += area
                total_perim += perim
                count += 1
                if poly_sink is None:
                    continue
                feat = QgsFeature(_fields(POLYGON_FIELDS))
                feat.setGeometry(geom)
                feat["level_m"] = round(lv.level, 4)
                feat["area_m2"] = round(area, 2)
                feat["area_ha"] = round(area / 1e4, 6)
                feat["perim_m"] = round(perim, 2)
                feat["n_islands"] = sum(len(part) - 1 for part in geom.asMultiPolygon())
                feat["lvl_vol_m3"] = round(lv.volume_m3, 2)
                feat["lvl_area_m2"] = round(lv.area_m2, 2)
                feat["lvl_maxdep"] = round(lv.max_depth, 3)
                feat["mode"] = mode
                poly_sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
            geometry[lv.level] = (total_area, total_perim, count)
            feedback.pushInfo(
                f"Level {lv.level:.3f} m: {count} water body(ies), "
                f"{total_area:.0f} m2, volume {lv.volume_m3:.0f} m3, maximum "
                f"depth {lv.max_depth:.2f} m"
            )
        if line_sink is not None:
            outputs[self.OUTPUT_LINES] = line_id
        if poly_sink is not None:
            outputs[self.OUTPUT_POLYGONS] = poly_id

        top = result.depth_level
        depth_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_DEPTH, context)
        if depth_path:
            depth = result.depth_at(top)
            self._write_raster(depth_path, depth, result, projection)
            outputs[self.OUTPUT_DEPTH] = depth_path
            valid = depth[np.isfinite(depth)]
            self._register_style(
                context,
                depth_path,
                raster_range=(
                    (float(valid.min()), float(valid.max())) if valid.size else None
                ),
            )
        wse_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_WSE, context)
        if wse_path:
            self._write_raster(
                wse_path, result.water_surface_at(top), result, projection
            )
            outputs[self.OUTPUT_WSE] = wse_path
            self._register_style(context, wse_path, STYLE_WSE_PATH)
        folder = self.parameterAsString(parameters, self.OUTPUT_LEVEL_FOLDER, context)
        if folder and not _is_null(folder):
            Path(folder).mkdir(parents=True, exist_ok=True)
            for lv in result.levels:
                tag = f"{lv.level:.3f}".replace(".", "_").replace("-", "m")
                self._write_raster(
                    str(Path(folder) / f"depth_{tag}.tif"),
                    result.depth_at(lv.level),
                    result,
                    projection,
                )
                self._write_raster(
                    str(Path(folder) / f"wse_{tag}.tif"),
                    result.water_surface_at(lv.level),
                    result,
                    projection,
                )
            feedback.pushInfo(
                f"Depth and water-surface rasters for {len(result.levels)} "
                f"level(s) written to {folder}"
            )
            outputs[self.OUTPUT_LEVEL_FOLDER] = folder
        table_path = self.parameterAsFileOutput(parameters, self.OUTPUT_TABLE, context)
        if table_path:
            write_summary_csv(table_path, result, geometry)
            outputs[self.OUTPUT_TABLE] = table_path

        docx_path = self.parameterAsFileOutput(parameters, self.OUTPUT_DOCX, context)
        if docx_path:
            feedback.setProgressText("Writing report")
            levels_text = (
                f"{levels[0]:.3f} m"
                if len(levels) == 1
                else f"{min(levels):.3f} to {max(levels):.3f} m "
                f"({len(levels)} levels, interval {interval:g} m)"
            )
            context_info = {
                "tool": self.displayName(),
                "version": TOOL_VERSION,
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "dem": {
                    "path": dem_layer.source(),
                    "crs": f"{crs.authid()} - {crs.description()}",
                    "cell_x": abs(gt[1]),
                    "cell_y": abs(gt[5]),
                },
                "levels_text": levels_text,
                "bounding": (
                    None
                    if bounding is None
                    else f"{bounding_source.sourceName()} "
                    f"({bounding_source.featureCount()} feature(s))"
                ),
                "seeds": {
                    "layer": None if seed_source is None else seed_source.sourceName()
                },
                "smoothing": smoothing,
                "min_cells": min_cells,
                "outputs": [
                    (label, _describe_destination(outputs.get(key)))
                    for key, label in (
                        (self.OUTPUT_LINES, "Inundation boundary (lines)"),
                        (self.OUTPUT_POLYGONS, "Inundation extent (polygons)"),
                        (self.OUTPUT_DEPTH, "Depth at the highest level"),
                        (
                            self.OUTPUT_WSE,
                            "Water-surface elevation at the highest level",
                        ),
                        (self.OUTPUT_LEVEL_FOLDER, "Rasters for every level (folder)"),
                        (self.OUTPUT_TABLE, "Summary per level (CSV)"),
                    )
                    if outputs.get(key)
                ]
                + [("This report", docx_path)],
                "warnings": list(result.warnings),
            }
            try:
                write_report_docx(
                    docx_path,
                    result,
                    context_info,
                    geometry,
                    bounding=bounding,
                    seeds_xy=seeds if mode == "connected" else (),
                )
                outputs[self.OUTPUT_DOCX] = docx_path
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package, which is not "
                    "installed in QGIS's Python. Install it from the OSGeo4W Shell "
                    "with: python -m pip install python-docx. All other outputs "
                    "were written."
                )
            except Exception as e:  # report failure must not lose the results
                feedback.pushWarning(f"The Word report could not be written: {e}")
        feedback.setProgress(100)
        return outputs
