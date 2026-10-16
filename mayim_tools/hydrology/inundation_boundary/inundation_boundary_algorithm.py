"""Processing Toolbox algorithm: DEM + one water level, optional bounding
polygon and seed points in -> smoothed inundation boundary (lines) and
extent (polygons), water-surface elevation and depth rasters, a summary
CSV and a Word report.

Outputs follow the Mayim report and figure standard: the user ticks the
outputs wanted and chooses one folder per kind (rasters, shapefiles, CSV,
Word); files are named <project number>_<description> with underscores;
every layer written is loaded into the project with its style.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_inundation_boundary_core.py).
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
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
    QgsProcessingParameterString,
    QgsProcessingUtils,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    RASTER,
    SHAPEFILE,
    WORD,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.hydrology._common.contours import dem_contours
from mayim_tools.hydrology._common.d8_network import (
    GridError,
    nodata_mask,
    write_raster,
)
from mayim_tools.hydrology._common.depression import (
    polygon_window,
    raster_info,
    read_window,
    window_geotransform,
)
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import save_shapefile
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
from .qgis_outputs import (
    VectorStyler,
    load_on_completion,
    make_plan_map,
    memory_layer,
    points_layer,
    polygon_layer,
)
from .report import write_report_docx

TOOL_VERSION = "0.5.0"

STYLE_DEPTH_PATH = str(STYLES_DIR / STYLE_DEPTH)  # stage-storage "Depth at top stage"
STYLE_WSE_PATH = str(
    Path(__file__).resolve().parent / "styles" / "water_surface_elevation.qml"
)

MODE_OPTIONS = [
    "All ground below the level (static bathtub)",
    "Only ground connected to the seed point(s)",
]

_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

# shapefile field names - all at most 10 characters
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

# Selectable outputs: key, dialog label, file-name description, folder kind,
# ticked by default.
OUTPUT_SPECS = [
    OutputSpec(
        "OUTPUT_BOUNDARY", "Inundation boundary (lines)", "Inundation Boundary",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_EXTENT", "Inundation extent (polygons)", "Inundation Extent",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_WSE", "Water-surface elevation raster", "Water Surface Elevation",
        RASTER,
    ),
    OutputSpec("OUTPUT_DEPTH", "Depth raster", "Inundation Depth", RASTER, False),
    OutputSpec("OUTPUT_CSV", "Summary table (CSV)", "Inundation Summary", CSV, False),
    OutputSpec(
        "OUTPUT_DOCX", "Inundation report (Word)", "Inundation Report", WORD, False
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


class InundationBoundaryAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    WATER_LEVEL = "WATER_LEVEL"
    BOUNDING = "BOUNDING"
    MODE = "MODE"
    SEEDS = "SEEDS"
    SMOOTHING = "SMOOTHING"
    MIN_CELLS = "MIN_CELLS"
    CONTOUR_INTERVAL = "CONTOUR_INTERVAL"
    REPORT_BASEMAP = "REPORT_BASEMAP"
    PROJECT_NUMBER = "PROJECT_NUMBER"

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
        glossary = "\n".join(
            f"  {n}: {d}" for n, _, d in POLYGON_FIELDS + LINE_FIELDS[1:2]
        )
        files = "\n".join(
            f"  {spec.label}: <project number>_"
            f"{clean_name_part(spec.description)}{EXTENSIONS[spec.kind]}"
            for spec in OUTPUT_SPECS
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
            "OUTPUTS:\tTick the outputs wanted (boundary, extent and "
            "water-surface elevation raster are ticked by default) and choose "
            "an output folder for each kind used: rasters (GeoTIFF), vector "
            "layers (shapefiles), CSV and Word. Files are named <project "
            "number>_<description> with underscores (just <description> "
            "without a project number) and replace files of the same name; a "
            "file that is open in QGIS cannot be replaced - remove it from the "
            "project first (the other outputs are still written). Every layer "
            "written is loaded into the project with its style.\n" + files + "\n"
            "The water-surface elevation raster holds the level on every "
            "inundated cell - profile it with the DEM for cross-sections. The "
            "rasters follow the DEM cells, so their edge is stepped; the "
            "smoothed polygons show the interpolated water's edge.\n"
            "\n"
            "WORD REPORT:\tAn annexure in the Mayim report standard: summary, "
            "method, inputs, parameters with definitions, depth classes, a "
            "layout map (rendered by QGIS: depth, boundary, bounding polygon, "
            "seeds, DEM contours at the set interval and an OpenStreetMap "
            "background - needs internet), the depth distribution, outputs, "
            "notes and references.\n"
            "\n"
            "LIMITATIONS:\tA level-pool (bathtub) extent - no flow, no "
            "hydraulic gradient, no structures below DEM resolution. For "
            "flood lines along a river use a hydraulic model.\n"
            "\n"
            "SHAPEFILE FIELDS (all 10 characters or fewer):\n" + glossary
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
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
                optional=True,
            )
        )
        for spec in OUTPUT_SPECS:
            self.addParameter(
                QgsProcessingParameterBoolean(
                    spec.key, f"Output: {spec.label}", defaultValue=spec.default
                )
            )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
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
            QgsProcessingParameterNumber(
                self.CONTOUR_INTERVAL,
                "Word report: contour interval on the layout map (m; 0 = none)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=1.0,
            ),
            QgsProcessingParameterBoolean(
                self.REPORT_BASEMAP,
                "Word report: OpenStreetMap background on the layout map "
                "(needs internet)",
                defaultValue=True,
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
    def _write_raster(path, array, window_gt, projection):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                raise RuntimeError("the existing file is in use") from e
        write_raster(
            path,
            np.where(np.isnan(array), DEFAULT_NODATA, array).astype("float32"),
            window_gt,
            projection,
            DEFAULT_NODATA,
        )

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
        contour_interval = self.parameterAsDouble(
            parameters, self.CONTOUR_INTERVAL, context
        )
        use_basemap = self.parameterAsBoolean(parameters, self.REPORT_BASEMAP, context)
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()

        selected = [
            spec
            for spec in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, spec.key, context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {spec.kind for spec in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {spec.key: spec.path(folders, project_number) for spec in selected}
        want = set(paths)

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
            f"Water level {level:.2f} m, mode: {mode}, window {window[3]} x "
            f"{window[2]} cells"
        )

        try:
            dem = read_window(dem_layer.source(), window)
            window_gt = window_geotransform(gt, window[0], window[1])
            result = run_inundation(
                dem,
                nodata,
                window_gt,
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

        # geometry (also measured for the CSV and the report)
        boundary = self._boundary_geometry(result, clip)
        extents = self._extent_geometries(result, clip)
        poly_area = sum(g.area() for g in extents)
        poly_perim = sum(g.length() for g in extents)
        geometry = (poly_area, poly_perim, len(extents))
        feedback.pushInfo(
            f"{len(extents)} water body(ies), {poly_area:.0f} m2, volume "
            f"{result.volume_m3:.0f} m3, maximum depth {result.max_depth:.2f} m"
        )
        boundary_layer = memory_layer(
            "Inundation boundary",
            LINE_FIELDS,
            Qgis.WkbType.MultiLineString,
            crs,
            (
                []
                if boundary is None
                else [
                    (
                        boundary,
                        {
                            "level_m": round(level, 4),
                            "length_m": round(boundary.length(), 2),
                            "mode": mode,
                        },
                    )
                ]
            ),
        )
        extent_layer = memory_layer(
            "Inundation extent",
            POLYGON_FIELDS,
            Qgis.WkbType.MultiPolygon,
            crs,
            [
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
            ],
        )

        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []  # (label, path) for the log and the report

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        def failed(key, path, error):
            feedback.reportError(
                f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be written "
                f"({error}). If it is open in QGIS, remove it from the project "
                "and run again.",
                fatalError=False,
            )

        # shapefiles, loaded into the project with their styles
        feedback.setProgressText("Writing layers")
        for key, layer, kind in (
            ("OUTPUT_BOUNDARY", boundary_layer, "boundary"),
            ("OUTPUT_EXTENT", extent_layer, "extent"),
        ):
            if key not in want:
                continue
            try:
                path = save_shapefile(layer, paths[key], context.transformContext())
            except Exception as e:
                failed(key, paths[key], e)
                continue
            load_on_completion(
                context, path, Path(path).stem, VectorStyler(kind),
                self._post_processors,
            )  # fmt: skip
            done(key, path)

        depth = result.depth
        valid = depth[np.isfinite(depth)]
        depth_range = (float(valid.min()), float(valid.max())) if valid.size else None
        rasters = (
            ("OUTPUT_WSE", result.water_surface, STYLE_WSE_PATH, {}),
            (
                "OUTPUT_DEPTH",
                depth,
                STYLE_DEPTH_PATH,
                {"raster_range": depth_range},
            ),
        )
        for key, array, qml, options in rasters:
            if key not in want:
                continue
            path = paths[key]
            try:
                self._write_raster(path, array, window_gt, projection)
            except Exception as e:
                failed(key, path, e)
                continue
            load_on_completion(
                context, path, Path(path).stem, _StylePostProcessor(qml, **options),
                self._post_processors,
            )  # fmt: skip
            done(key, path)

        if "OUTPUT_CSV" in want:
            path = paths["OUTPUT_CSV"]
            try:
                write_summary_csv(path, result, geometry)
                done("OUTPUT_CSV", path)
            except Exception as e:
                failed("OUTPUT_CSV", path, e)
        feedback.setProgress(70)

        if "OUTPUT_DOCX" in want:
            docx_path = paths["OUTPUT_DOCX"]
            feedback.setProgressText("Writing report")
            map_messages = []
            depth_map_path = outputs.get("OUTPUT_DEPTH")
            if depth_map_path is None and valid.size:
                depth_map_path = os.path.join(
                    QgsProcessingUtils.tempFolder(), "inundation_depth_map.tif"
                )
                try:
                    self._write_raster(depth_map_path, depth, window_gt, projection)
                except Exception as e:
                    map_messages.append(f"Layout map: depth not shown ({e}).")
                    depth_map_path = None
            dem_nan = np.where(nodata_mask(dem, nodata), np.nan, dem).astype(float)

            def contours_for(extent):
                lines, warnings = dem_contours(
                    dem_nan, window_gt, contour_interval, extent=extent
                )
                map_messages.extend(warnings)
                return lines

            plan_map = make_plan_map(
                crs,
                context.transformContext(),
                boundary=boundary_layer.clone(),
                seeds=(
                    points_layer("Seed points", seeds, crs)
                    if mode == "connected" and seeds
                    else None
                ),
                study_area=(
                    None
                    if clip is None
                    else polygon_layer("Bounding polygon", clip, crs)
                ),
                depth_path=depth_map_path,
                depth_qml=STYLE_DEPTH_PATH,
                depth_range=depth_range,
                bounding=bounding,
                contours_for=contours_for,
                contour_interval=contour_interval,
                use_osm=use_basemap,
                messages=map_messages,
            )
            context_info = {
                "tool": self.displayName(),
                "version": TOOL_VERSION,
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "project_number": project_number or None,
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
                "outputs": written + [("This report", docx_path)],
                "warnings": list(result.warnings),
            }
            try:
                messages = write_report_docx(
                    docx_path,
                    result,
                    context_info,
                    geometry,
                    bounding=bounding,
                    seeds_xy=seeds if mode == "connected" else (),
                    plan_map=plan_map,
                )
                for message in dict.fromkeys(map_messages + (messages or [])):
                    feedback.pushWarning(message)
                done("OUTPUT_DOCX", docx_path)
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package, which is not "
                    "installed in QGIS's Python. Install it from the OSGeo4W Shell "
                    "with: python -m pip install python-docx. All other outputs "
                    "were written."
                )
            except PermissionError as e:
                failed("OUTPUT_DOCX", docx_path, e)
            except Exception as e:  # report failure must not lose the results
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        feedback.setProgress(100)
        return outputs
