"""Processing Toolbox algorithm: DEM + one water level, optional bounding
polygon and seed points in -> smoothed inundation boundary (lines) and
extent (polygons), water-surface elevation and depth rasters, a summary
CSV and a Word report, written to one output folder.

The user picks an output folder and ticks the outputs wanted; the
boundary, the extent and the water-surface elevation raster are ticked
by default, and every output layer written is loaded into the map.

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
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsVectorFileWriter,
    QgsVectorLayer,
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
)
from .export import write_summary_csv
from .report import write_report_docx

TOOL_VERSION = "0.4.0"

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
    ("mode", _S, "all or connected"),
]

LINE_FIELDS = [
    ("level_m", _D, "water level, m"),
    ("length_m", _D, "length of the boundary line, m"),
    ("mode", _S, "all or connected"),
]

# (parameter, label, default, file suffix)
OUTPUT_CHOICES = [
    ("SAVE_BOUNDARY", "Inundation boundary (lines, GeoPackage)", True, "boundary.gpkg"),
    ("SAVE_EXTENT", "Inundation extent (polygons, GeoPackage)", True, "extent.gpkg"),
    ("SAVE_WSE", "Water-surface elevation raster (GeoTIFF)", True, "wse.tif"),
    ("SAVE_DEPTH", "Depth raster (GeoTIFF)", False, "depth.tif"),
    ("SAVE_CSV", "Summary table (CSV)", False, "summary.csv"),
    ("SAVE_REPORT", "Inundation report (Word)", False, "report.docx"),
]


def _fields(specs) -> QgsFields:
    fields = QgsFields()
    for name, qtype, _ in specs:
        fields.append(QgsField(name, qtype))
    return fields


def _locked_message(path, error) -> str:
    return (
        f"Could not write '{path}': {error}. If a layer from an earlier run at "
        "the same level is still open in QGIS, remove it from the map (or "
        "choose another output folder) and run the tool again."
    )


def level_tag(level: float) -> str:
    """File-name tag for a level: 1617.41 -> '1617_410' (no dots)."""
    return f"{level:.3f}".replace(".", "_").replace("-", "m")


class InundationBoundaryAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    WATER_LEVEL = "WATER_LEVEL"
    BOUNDING = "BOUNDING"
    MODE = "MODE"
    SEEDS = "SEEDS"
    SMOOTHING = "SMOOTHING"
    MIN_CELLS = "MIN_CELLS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

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
        files = "\n".join(
            f"  {label}: inundation_<level>_{suffix}"
            for _, label, _, suffix in OUTPUT_CHOICES
        )
        return (
            f"DEM Inundation Boundary (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tThe extent of water standing at one elevation (bathtub "
            "inundation), as a smoothed boundary line and matching polygons, "
            "with the inundated area, volume and depth.\n"
            "\n"
            "INPUTS:\tA DEM in a projected CRS in metres and the water level. "
            "Optional: a bounding polygon that limits the analysis (all its "
            "features together; without it the whole DEM is used), and seed "
            "points for the connected option.\n"
            "\n"
            "INUNDATED GROUND:\t'All ground below the level' floods every cell "
            "below the level, including separate hollows. 'Only ground "
            "connected to the seed point(s)' floods only ground that water at "
            "the level can reach from a seed (e.g. a river, dam or outlet "
            "point), through cells below the level (8-connected); hollows behind "
            "higher ground stay dry. Without a seed the lowest cell is used.\n"
            "\n"
            "SEED POINTS:\tWater spreads outward from the seed cell, so a level "
            "at or below the ground at the seed floods nothing. Place seeds on "
            "the lowest ground of the water body (channel bed, dam basin), not "
            "on a bank.\n"
            "\n"
            "BOUNDARY:\tThe water's edge is interpolated between DEM cell "
            "centres (marching squares) and rounded with Chaikin corner "
            "cutting; the polygons are built from the same lines (outer ring = "
            "water, inner rings = dry islands), so line and polygon match. "
            "Where the water reaches the bounding polygon, both are cut at its "
            "edge. Rings smaller than the minimum size (DEM noise) are "
            "dropped.\n"
            "\n"
            "OUTPUTS:\tChoose an output folder and tick the outputs wanted. The "
            "boundary, the extent and the water-surface elevation raster are "
            "ticked by default. Every layer written is loaded into the map. "
            "Files are named after the level and overwrite an earlier run at the "
            "same level:\n" + files + "\n"
            "The water-surface elevation raster holds the level on every "
            "inundated cell - profile it with the DEM for cross-sections. The "
            "rasters follow the DEM cells, so their edge is stepped; the "
            "smoothed polygons show the interpolated water's edge.\n"
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
                self.WATER_LEVEL,
                "Water level (m)",
                type=QgsProcessingParameterNumber.Type.Double,
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
        for key, label, default, _ in OUTPUT_CHOICES:
            self.addParameter(
                QgsProcessingParameterBoolean(key, label, defaultValue=default)
            )
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "Output folder")
        )
        for param in (
            QgsProcessingParameterNumber(
                self.SMOOTHING,
                "Smoothing passes (Chaikin; 0 = none)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                maxValue=8,
                defaultValue=DEFAULT_SMOOTHING,
            ),
            QgsProcessingParameterNumber(
                self.MIN_CELLS,
                "Drop rings smaller than (cells)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=DEFAULT_MIN_CELLS,
            ),
        ):
            param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
            self.addParameter(param)

    # ------------------------------------------------------------------
    # Helpers

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

    @staticmethod
    def _boundary_geometry(result, clip):
        if not result.lines:
            return None
        parts = [[QgsPointXY(x, y) for x, y in ring] for ring in result.lines]
        geom = QgsGeometry.fromMultiPolylineXY(parts)
        if clip is not None:
            geom = geom.intersection(clip)
            if QgsWkbTypes.flatType(geom.wkbType()) not in (
                Qgis.WkbType.LineString,
                Qgis.WkbType.MultiLineString,
            ):
                geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Line)
        if geom.isEmpty():
            return None
        geom.convertToMultiType()
        return geom

    @staticmethod
    def _extent_geometries(result, clip):
        out = []
        for poly in result.polygons:
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
            out.append(geom)
        return out

    @staticmethod
    def _write_gpkg(path, layer_name, specs, wkb_type, crs, rows, context):
        """Writes features [(geometry, {field: value})] to a GeoPackage: an
        in-memory layer filled in one batch, then saved in one call (the
        idiom proven in the Import RAS 2D data tool)."""
        layer = QgsVectorLayer(
            QgsWkbTypes.displayString(wkb_type), layer_name, "memory"
        )
        layer.setCrs(crs)
        provider = layer.dataProvider()
        provider.addAttributes(list(_fields(specs)))
        layer.updateFields()
        features = []
        for geom, values in rows:
            feat = QgsFeature(layer.fields())
            feat.setGeometry(geom)
            for key, value in values.items():
                feat[key] = value
            features.append(feat)
        provider.addFeatures(features)
        layer.updateExtents()
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"
        options.layerName = layer_name
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, path, context.transformContext(), options
        )
        if result[0] != QgsVectorFileWriter.WriterError.NoError:
            message = result[1] if len(result) > 1 else ""
            raise QgsProcessingException(_locked_message(path, message))

    @staticmethod
    def _write_raster(path, array, result, projection):
        try:
            write_raster(
                path,
                np.where(np.isnan(array), DEFAULT_NODATA, array).astype("float32"),
                result.window_gt,
                projection,
                DEFAULT_NODATA,
            )
        except Exception as e:
            raise QgsProcessingException(_locked_message(path, e)) from e

    def _load(self, path, name, context, style_path=None, **style_options):
        """Loads a written file into the map when the run finishes, with an
        optional QML style (post-processor kept alive on self - Processing
        holds only a weak reference)."""
        details = QgsProcessingContext.LayerDetails(name, context.project())
        if style_path is not None:
            processor = _StylePostProcessor(style_path, **style_options)
            details.setPostProcessor(processor)
            self._post_processors.append(processor)
        context.addLayerToLoadOnCompletion(path, details)

    # ------------------------------------------------------------------

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
        level = self.parameterAsDouble(parameters, self.WATER_LEVEL, context)
        mode = MODES[self.parameterAsEnum(parameters, self.MODE, context)]
        smoothing = self.parameterAsInt(parameters, self.SMOOTHING, context)
        min_cells = self.parameterAsDouble(parameters, self.MIN_CELLS, context)
        wanted = {
            key: self.parameterAsBool(parameters, key, context)
            for key, _, _, _ in OUTPUT_CHOICES
        }
        if not any(wanted.values()):
            raise QgsProcessingException("Tick at least one output.")
        folder = self.parameterAsFileOutput(
            parameters, self.OUTPUT_FOLDER, context
        ) or self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not folder:
            raise QgsProcessingException("An output folder is required.")
        Path(folder).mkdir(parents=True, exist_ok=True)
        try:
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
            f"Water level {level:.3f} m, mode: {mode}, window {window[3]} x "
            f"{window[2]} cells"
        )

        try:
            dem = read_window(dem_layer.source(), window)
            result = run_inundation(
                dem,
                nodata,
                window_geotransform(gt, window[0], window[1]),
                window,
                level,
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
        feedback.setProgress(50)

        # Geometry (also measured for the CSV and the report)
        boundary = self._boundary_geometry(result, clip)
        extents = self._extent_geometries(result, clip)
        poly_area = sum(g.area() for g in extents)
        poly_perim = sum(g.length() for g in extents)
        geometry = (poly_area, poly_perim, len(extents))
        feedback.pushInfo(
            f"{len(extents)} water body(ies), {poly_area:.0f} m2, volume "
            f"{result.volume_m3:.0f} m3, maximum depth {result.max_depth:.2f} m"
        )

        base = Path(folder) / f"inundation_{level_tag(level)}"
        paths = {
            key: f"{base}_{suffix}"
            for key, _, _, suffix in OUTPUT_CHOICES
            if wanted[key]
        }
        lv = f"{level:.2f} m"
        written = []

        if wanted["SAVE_BOUNDARY"]:
            rows = []
            if boundary is not None:
                rows.append(
                    (
                        boundary,
                        {
                            "level_m": round(level, 4),
                            "length_m": round(boundary.length(), 2),
                            "mode": mode,
                        },
                    )
                )
            self._write_gpkg(
                paths["SAVE_BOUNDARY"],
                "inundation_boundary",
                LINE_FIELDS,
                Qgis.WkbType.MultiLineString,
                crs,
                rows,
                context,
            )
            self._load(paths["SAVE_BOUNDARY"], f"Inundation boundary {lv}", context)
            written.append(("Inundation boundary (lines)", paths["SAVE_BOUNDARY"]))

        if wanted["SAVE_EXTENT"]:
            rows = [
                (
                    geom,
                    {
                        "level_m": round(level, 4),
                        "area_m2": round(geom.area(), 2),
                        "area_ha": round(geom.area() / 1e4, 6),
                        "perim_m": round(geom.length(), 2),
                        "n_islands": sum(
                            len(part) - 1 for part in geom.asMultiPolygon()
                        ),
                        "mode": mode,
                    },
                )
                for geom in extents
            ]
            self._write_gpkg(
                paths["SAVE_EXTENT"],
                "inundation_extent",
                POLYGON_FIELDS,
                Qgis.WkbType.MultiPolygon,
                crs,
                rows,
                context,
            )
            self._load(paths["SAVE_EXTENT"], f"Inundation extent {lv}", context)
            written.append(("Inundation extent (polygons)", paths["SAVE_EXTENT"]))

        if wanted["SAVE_WSE"]:
            self._write_raster(
                paths["SAVE_WSE"], result.water_surface, result, projection
            )
            self._load(
                paths["SAVE_WSE"],
                f"Water-surface elevation {lv}",
                context,
                STYLE_WSE_PATH,
            )
            written.append(("Water-surface elevation raster", paths["SAVE_WSE"]))

        if wanted["SAVE_DEPTH"]:
            depth = result.depth
            self._write_raster(paths["SAVE_DEPTH"], depth, result, projection)
            valid = depth[np.isfinite(depth)]
            self._load(
                paths["SAVE_DEPTH"],
                f"Depth {lv}",
                context,
                STYLE_DEPTH_PATH,
                raster_range=(
                    (float(valid.min()), float(valid.max())) if valid.size else None
                ),
            )
            written.append(("Depth raster", paths["SAVE_DEPTH"]))

        if wanted["SAVE_CSV"]:
            write_summary_csv(paths["SAVE_CSV"], result, geometry)
            written.append(("Summary table (CSV)", paths["SAVE_CSV"]))
        feedback.setProgress(80)

        if wanted["SAVE_REPORT"]:
            feedback.setProgressText("Writing report")
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
                "outputs": written + [("This report", paths["SAVE_REPORT"])],
                "warnings": list(result.warnings),
            }
            try:
                write_report_docx(
                    paths["SAVE_REPORT"],
                    result,
                    context_info,
                    geometry,
                    bounding=bounding,
                    seeds_xy=seeds if mode == "connected" else (),
                )
                written.append(("Inundation report (Word)", paths["SAVE_REPORT"]))
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package, which is not "
                    "installed in QGIS's Python. Install it from the OSGeo4W Shell "
                    "with: python -m pip install python-docx. All other outputs "
                    "were written."
                )
            except Exception as e:  # report failure must not lose the results
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        feedback.setProgress(100)
        return {self.OUTPUT_FOLDER: folder}
