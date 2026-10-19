"""Processing Toolbox algorithm: DEM in -> hydrologically corrected DEM, the
elevation change, remaining-sink points, a sectioned CSV, a classification
table and a Word report out (Mayim report and figure standard; outputs are
ticked, written to per-kind folders and named <project number>_<description>).

Default is continuous drainage: every depression is resolved (filled or
breached, whichever disturbs the DEM less) and a drainage-enforcement pass
removes the artefact pits those corrections can leave behind. Genuine sinks
(pans, regional sinks, lakes) are protected only when the user supplies them
as a polygon or point layer.

Thin wrapper - all real logic lives in core/ (zero QGIS dependency,
independently testable; see tests/test_dem_hydro_processing_*.py).
"""

import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingUtils,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    RASTER,
    SHAPEFILE,
    WORD,
    OutputSpec,
)
from mayim_tools.hydrology._common.contours import dem_contours

from .core.classification import DEM_SOURCE_VERTICAL_ERROR_M
from .core.conditioning import (
    MODE_CONTINUOUS,
    MODE_PRESERVE,
    ConditioningError,
    ConditioningSettings,
    run_conditioning,
)
from .core.keep_sinks import KeepFeature
from .core.report import read_classification_report, write_classification_report
from .core.stream_evidence import RasterTransform
from .export import write_sectioned_csv
from .qgis_map import (
    STYLE_CHANGE,
    STYLE_SINKS,
    copy_source_layer,
    load_on_completion,
    make_plan_map,
    save_shapefile,
    sinks_layer,
)
from .report import change_classes_present, elevation_change, write_report_docx

TOOL_VERSION = "1.1.0"
SOURCE_KEYS = list(DEM_SOURCE_VERTICAL_ERROR_M) + ["custom"]
SOURCE_LABELS = {
    "survey": "Ground survey",
    "lidar": "Airborne lidar",
    "uav_photogrammetry": "UAV photogrammetry",
    "copernicus_glo30": "Copernicus GLO-30",
    "srtm": "SRTM",
    "aster": "ASTER GDEM",
    "custom": "Custom (enter the error below)",
}
MODE_OPTIONS = [
    "Continuous drainage - resolve everything except confirmed sinks (default)",
    "Preserve Likely Natural depressions (classified by persistence)",
]


def _advanced(param):
    param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
    return param


# Selectable outputs: (key, dialog label, file-name description, folder kind,
# ticked by default).
OUTPUT_SPECS = [
    OutputSpec("OUTPUT_DEM", "Corrected DEM", "Corrected DEM", RASTER),
    OutputSpec(
        "OUTPUT_CHANGE",
        "Elevation change (corrected minus original)",
        "DEM Elevation Change",
        RASTER,
        default=False,
    ),
    OutputSpec("OUTPUT_SINKS", "Remaining sinks", "Remaining Sinks", SHAPEFILE),
    OutputSpec("OUTPUT_CSV", "Results (CSV)", "DEM Processing Results", CSV),
    OutputSpec(
        "OUTPUT_CLASSES",
        "Classification table (CSV, reusable)",
        "Depression Classification",
        CSV,
        default=False,
    ),
    OutputSpec("OUTPUT_DOCX", "Report (Word)", "DEM Processing Report", WORD),
]
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


def _software_version() -> str:
    """'Mayim Tools <version>, QGIS <version>' for the report title page."""
    parts = []
    metadata = Path(__file__).resolve().parents[2] / "metadata.txt"
    try:
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.startswith("version="):
                parts.append(f"Mayim Tools {line.split('=', 1)[1].strip()}")
                break
    except OSError:
        pass
    parts.append(f"QGIS {Qgis.version()}")
    return ", ".join(parts)


def _read_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise QgsProcessingException(f"Could not open the DEM at {path} with GDAL.")
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    ds = None
    if abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9:
        raise QgsProcessingException(
            "The DEM is a rotated raster; only north-up rasters are supported."
        )
    return array, nodata, gt


def _write_raster(path, array, template_path, nodata):
    """Float32 GeoTIFF on the template's grid, replacing an existing file."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            raise RuntimeError(
                f"{Path(path).name} exists and could not be replaced - it is "
                "probably open in QGIS. Remove it from the project and run again."
            ) from e
    template = gdal.Open(template_path)
    out = gdal.GetDriverByName("GTiff").Create(
        path, template.RasterXSize, template.RasterYSize, 1, gdal.GDT_Float32
    )
    if out is None:
        raise RuntimeError(f"{Path(path).name} could not be created.")
    out.SetGeoTransform(template.GetGeoTransform())
    out.SetProjection(template.GetProjection())
    band = out.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array.astype(np.float32))
    band.FlushCache()
    out = None
    template = None


class DemHydroProcessingAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    DEM_SOURCE = "DEM_SOURCE"
    CUSTOM_ERROR = "CUSTOM_ERROR"
    MODE = "MODE"
    KEEP = "KEEP"
    KEEP_NAME = "KEEP_NAME"
    KEEP_RADIUS = "KEEP_RADIUS"
    STREAMS = "STREAMS"
    CLASSIFY = "CLASSIFY"
    REVIEW_TABLE = "REVIEW_TABLE"
    N_ITER = "N_ITER"
    CORR_LEN = "CORR_LEN"
    MAX_BREACH_LEN = "MAX_BREACH_LEN"
    MAX_BREACH_DEPTH = "MAX_BREACH_DEPTH"
    SEED = "SEED"
    STUDY_AREA = "STUDY_AREA"
    CONTOUR_INTERVAL = "CONTOUR_INTERVAL"
    REPORT_OSM = "REPORT_OSM"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"
    OUTPUT_DEM = "OUTPUT_DEM"
    OUTPUT_CHANGE = "OUTPUT_CHANGE"
    OUTPUT_SINKS = "OUTPUT_SINKS"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_DOCX = "OUTPUT_DOCX"
    OUTPUT_CLASSES = "OUTPUT_CLASSES"

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return DemHydroProcessingAlgorithm()

    def name(self):
        return "dem_hydro_processing"

    def displayName(self):
        return "DEM Hydrological Processing"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        return (
            f"DEM Hydrological Processing (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tCorrects a DEM so every cell drains, ready for flow "
            "direction and accumulation. Depressions are found with a "
            "depression hierarchy, each is filled or breached (whichever "
            "changes the DEM less), and a drainage-enforcement pass removes "
            "any pits those corrections leave behind.\n"
            "\n"
            "DEFAULT:\tContinuous drainage. Everything is resolved except "
            "sinks you confirm. No table editing is needed.\n"
            "\n"
            "CONFIRMED SINKS:\tOptionally supply pans, regional sinks or "
            "lakes as polygons or points. A polygon protects every "
            "depression it overlaps. A point is snapped to the nearest "
            "depression within the search radius; if none is found, the "
            "lowest cell in the radius becomes the sink. Every feature's "
            "outcome is written to the log and the report.\n"
            "\n"
            "MODE:\t'Preserve Likely Natural' keeps depressions that persist "
            "under simulated DEM vertical error (Lindsay & Creed). That label "
            "only means deeper than the DEM noise; dams and borrow pits also "
            "qualify, so review the result. Classification can be run in "
            "continuous mode for information only, or reused from an earlier "
            "table to skip the Monte Carlo step.\n"
            "\n"
            "INPUTS:\tA north-up DEM. Cell size is taken in metres (converted "
            "approximately for geographic CRS). Choose the DEM source so the "
            "vertical error is right. The study area and contour interval are "
            "used on the report's layout map only.\n"
            "\n"
            "OUTPUTS:\tTick the outputs you need and choose a folder for each "
            "kind (rasters, shapefiles, CSV, Word). Files are named "
            "<project number>_<description>, e.g. MC2025_014_Corrected_DEM.tif; "
            "existing files are replaced. Layers are loaded with the Mayim "
            "styles. Outputs: corrected DEM, elevation change (corrected minus "
            "original), remaining sinks (points; fields sink_id, kind, reason, "
            "cells, elev_m), results CSV, reusable classification table and "
            "the Word report (title page, layout map, tables; needs "
            "python-docx and matplotlib in QGIS's Python).\n"
            "\n"
            "Version 1.1.0 replaced the file outputs with output tick boxes "
            "and folders; Processing models saved with 1.0 need the outputs "
            "set again."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.DEM, "DEM"))
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEM_SOURCE,
                "DEM source (sets vertical accuracy)",
                options=[SOURCE_LABELS[k] for k in SOURCE_KEYS],
                defaultValue=SOURCE_KEYS.index("copernicus_glo30"),
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CUSTOM_ERROR,
                "Custom vertical error (m) - used when source is Custom",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=1.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE, "Mode", options=MODE_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.KEEP,
                "Confirmed sinks to keep (polygons or points, optional)",
                types=[
                    QgsProcessing.SourceType.TypeVectorPolygon,
                    QgsProcessing.SourceType.TypeVectorPoint,
                ],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.KEEP_NAME,
                "Confirmed sink name field (optional)",
                parentLayerParameterName=self.KEEP,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KEEP_RADIUS,
                "Point search radius (cells)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STREAMS,
                "Stream centrelines (optional; adds stream relation)",
                types=[QgsProcessing.SourceType.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CLASSIFY,
                "Classify depressions in continuous mode (information only; " "slower)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.REVIEW_TABLE,
                "Earlier classification table to reuse (CSV, optional)",
                extension="csv",
                optional=True,
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.N_ITER,
                    "Monte Carlo iterations",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    minValue=10,
                    defaultValue=ConditioningSettings().n_iterations,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.CORR_LEN,
                    "Error correlation length (cells)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    minValue=0.0,
                    defaultValue=ConditioningSettings().correlation_length_cells,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.MAX_BREACH_LEN,
                    "Maximum breach length (cells; 0 = unlimited)",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    minValue=0,
                    defaultValue=0,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.MAX_BREACH_DEPTH,
                    "Maximum breach depth (m; 0 = unlimited)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    minValue=0.0,
                    defaultValue=0.0,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.SEED,
                    "Random seed",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=0,
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STUDY_AREA,
                "Study area boundary for the report layout map (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONTOUR_INTERVAL,
                "Contour interval for the report layout map (m; 0 = no contours)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=1.0,
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.REPORT_OSM,
                    "Word report: OpenStreetMap background on the layout map "
                    "(needs internet)",
                    defaultValue=True,
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix and report, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project / site name (report title page, optional)",
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

    # ------------------------------------------------------------------

    @staticmethod
    def _cell_size_m(gt, rows, crs):
        px, py = abs(gt[1]), abs(gt[5])
        if crs.isGeographic():
            lat = gt[3] + gt[5] * rows / 2.0
            return (
                px * 111320.0 * math.cos(math.radians(lat)),
                py * 110574.0,
            )
        return px, py

    def _read_keep(self, source, name_field, dem_crs, context, feedback):
        if source is None:
            return []
        xform = None
        if source.sourceCrs() != dem_crs:
            xform = QgsCoordinateTransform(
                source.sourceCrs(), dem_crs, context.transformContext()
            )
        out = []
        for n, feat in enumerate(source.getFeatures(), start=1):
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if xform is not None:
                geom = QgsGeometry(geom)
                geom.transform(xform)
            label = str(feat[name_field]) if name_field else f"feature {n}"
            if geom.type() == Qgis.GeometryType.Polygon:
                parts = (
                    geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
                )
                for p, rings in enumerate(parts, start=1):
                    tag = label if len(parts) == 1 else f"{label} (part {p})"
                    out.append(
                        KeepFeature(
                            tag,
                            "polygon",
                            rings=[[(q.x(), q.y()) for q in ring] for ring in rings],
                        )
                    )
            elif geom.type() == Qgis.GeometryType.Point:
                pts = geom.asMultiPoint() if geom.isMultipart() else [geom.asPoint()]
                for p, q in enumerate(pts, start=1):
                    tag = label if len(pts) == 1 else f"{label} (point {p})"
                    out.append(KeepFeature(tag, "point", xy=(q.x(), q.y())))
        feedback.pushInfo(f"Confirmed sinks read: {len(out)} feature(s).")
        return out

    def _read_streams(self, source, dem_crs, context):
        if source is None:
            return None
        xform = None
        if source.sourceCrs() != dem_crs:
            xform = QgsCoordinateTransform(
                source.sourceCrs(), dem_crs, context.transformContext()
            )
        lines = []
        for feat in source.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            if xform is not None:
                geom = QgsGeometry(geom)
                geom.transform(xform)
            parts = (
                geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
            )
            for part in parts:
                lines.append([(q.x(), q.y()) for q in part])
        return lines

    def processAlgorithm(self, parameters, context, feedback):
        from .core import breaching as _breaching

        feedback.pushInfo(
            f"DEM Hydrological Processing {TOOL_VERSION}; breach search cap "
            f"{getattr(_breaching, 'MAX_EXPLORED_CELLS', 'NOT PRESENT (old code)')}; "
            f"code in {Path(__file__).resolve().parent}"
        )
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if dem_layer is None:
            raise QgsProcessingException("Select a DEM.")

        # outputs and folders first, so a missing folder stops the run early
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        ).strip()
        project_name = self.parameterAsString(
            parameters, self.PROJECT_NAME, context
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
        dem_path = dem_layer.source()
        crs = dem_layer.crs()
        dem, nodata, gt = _read_raster(dem_path)
        rows, cols = dem.shape
        transform = RasterTransform(
            origin_x=gt[0],
            origin_y=gt[3],
            pixel_width=abs(gt[1]),
            pixel_height=abs(gt[5]),
        )
        cell_size = self._cell_size_m(gt, rows, crs)
        if crs.isGeographic():
            feedback.pushWarning(
                "The DEM uses a geographic CRS; volumes and areas use an "
                "approximate cell size in metres. A projected CRS is preferred."
            )
        feedback.pushInfo(
            f"DEM: {cols} x {rows} cells, cell size {cell_size[0]:.3f} x "
            f"{cell_size[1]:.3f} m."
        )

        src_key = SOURCE_KEYS[
            self.parameterAsEnum(parameters, self.DEM_SOURCE, context)
        ]
        if src_key == "custom":
            vert = self.parameterAsDouble(parameters, self.CUSTOM_ERROR, context)
            if not vert > 0:
                raise QgsProcessingException(
                    "Enter a custom vertical error greater than zero."
                )
        else:
            vert = DEM_SOURCE_VERTICAL_ERROR_M[src_key]

        mode_idx = self.parameterAsEnum(parameters, self.MODE, context)
        classify = self.parameterAsBool(parameters, self.CLASSIFY, context)
        review_path = self.parameterAsFile(parameters, self.REVIEW_TABLE, context)
        review_labels = None
        if review_path:
            review_labels = read_classification_report(review_path)
            feedback.pushInfo(
                f"Reusing classification table ({len(review_labels)} labels); "
                "Monte Carlo step skipped."
            )
        n_breach_len = self.parameterAsInt(parameters, self.MAX_BREACH_LEN, context)
        breach_depth = self.parameterAsDouble(
            parameters, self.MAX_BREACH_DEPTH, context
        )
        settings = ConditioningSettings(
            mode=MODE_CONTINUOUS if mode_idx == 0 else MODE_PRESERVE,
            classify=classify,
            vertical_error_std=vert,
            n_iterations=self.parameterAsInt(parameters, self.N_ITER, context),
            correlation_length_cells=self.parameterAsDouble(
                parameters, self.CORR_LEN, context
            ),
            seed=self.parameterAsInt(parameters, self.SEED, context),
            max_breach_length=n_breach_len or None,
            max_breach_depth=breach_depth or None,
            keep_search_cells=self.parameterAsInt(
                parameters, self.KEEP_RADIUS, context
            ),
        )

        keep_source = self.parameterAsSource(parameters, self.KEEP, context)
        name_field = self.parameterAsString(parameters, self.KEEP_NAME, context) or None
        keep = self._read_keep(keep_source, name_field, crs, context, feedback)
        streams = self._read_streams(
            self.parameterAsSource(parameters, self.STREAMS, context), crs, context
        )

        def progress(frac, text):
            feedback.setProgress(int(100 * frac))
            if text:
                feedback.setProgressText(text)
                if text.startswith(("Slow unit", "Resolving", "Classifying")):
                    feedback.pushInfo(text)
                elif " units resolved" in text:
                    feedback.pushInfo(text)

        try:
            result = run_conditioning(
                dem,
                nodata,
                transform,
                cell_size,
                settings=settings,
                stream_polylines=streams,
                keep_features=keep,
                review_labels=review_labels,
                progress=progress,
                is_canceled=feedback.isCanceled,
            )
        except ConditioningError as e:
            raise QgsProcessingException(str(e)) from e

        st = result.stats
        feedback.pushInfo(
            f"Depressions found: {st['depressions_found']} "
            f"({st['resolution_units']} resolution units; "
            f"{st['units_filled']} filled, {st['units_breached']} breached, "
            f"{st['kept_depressions']} kept)."
        )
        feedback.pushInfo(
            f"Drainage enforcement: {st['pits_after_fill_breach']} cell(s) with "
            f"no downhill neighbour after fill/breach; "
            f"{st['cells_raised_by_enforcement']} cell(s) raised."
        )
        feedback.pushInfo(
            f"Cells changed: {st['cells_changed']} "
            f"({st['pct_cells_changed']:.2f}% of valid); largest raise "
            f"{st['max_raise_m']:.3f} m, largest lowering "
            f"{st['max_lowering_m']:.3f} m."
        )
        for m in result.keep_matches:
            feedback.pushInfo(
                f"Confirmed sink '{m.label}' ({m.kind}): {m.status}. {m.detail}"
            )
        n_unres = st["unresolved_sink_clusters"]
        feedback.pushInfo(
            f"Sink clusters remaining: {st['remaining_sink_clusters']} "
            f"({n_unres} not confirmed)."
        )
        if n_unres:
            feedback.pushWarning(
                f"{n_unres} sink cluster(s) remain that were not confirmed; "
                "see the remaining sinks layer and report section 5.3."
            )
        for w in result.warnings:
            feedback.pushWarning(w)

        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []  # (label, path) for the report

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        def failed(key, error):
            feedback.reportError(
                f"{SPEC_BY_KEY[key].label} not written: {error}", fatalError=False
            )

        diff = elevation_change(result, nodata)
        diff_nodata = nodata if nodata is not None else -9999.0
        diff_out = np.where(np.isfinite(diff), diff, diff_nodata)

        # rasters
        if self.OUTPUT_DEM in paths:
            path = paths[self.OUTPUT_DEM]
            try:
                _write_raster(path, result.corrected, dem_path, nodata)
                load_on_completion(context, path, None, self._post_processors)
                done(self.OUTPUT_DEM, path)
            except Exception as e:
                failed(self.OUTPUT_DEM, e)
        change_path = None
        if self.OUTPUT_CHANGE in paths:
            path = paths[self.OUTPUT_CHANGE]
            try:
                _write_raster(path, diff_out, dem_path, diff_nodata)
                load_on_completion(context, path, STYLE_CHANGE, self._post_processors)
                done(self.OUTPUT_CHANGE, path)
                change_path = path
            except Exception as e:
                failed(self.OUTPUT_CHANGE, e)

        # remaining sinks (memory layer: written as a shapefile and mapped)
        sinks = sinks_layer(result.sinks, gt, crs)
        if self.OUTPUT_SINKS in paths:
            try:
                path = save_shapefile(
                    sinks, paths[self.OUTPUT_SINKS], context.transformContext()
                )
                load_on_completion(context, path, STYLE_SINKS, self._post_processors)
                done(self.OUTPUT_SINKS, path)
            except Exception as e:
                failed(self.OUTPUT_SINKS, e)

        ctx = {
            "tool": self.displayName(),
            "version": TOOL_VERSION,
            "software": _software_version(),
            "project_number": project_number,
            "project_name": project_name,
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dem_path": dem_path,
            "crs": f"{crs.authid()} - {crs.description()}",
            "cell_size": f"{cell_size[0]:.4g} x {cell_size[1]:.4g}",
            "cell_area_m2": cell_size[0] * cell_size[1],
            "grid": f"{cols} x {rows}",
            "dem_source": SOURCE_LABELS[src_key],
            "stream_layer": "yes" if streams is not None else "no",
            "keep_layer": f"yes ({len(keep)} feature(s))" if keep else "no",
            "review_table": review_path or "no",
            "geotransform": gt,
            "nodata": nodata,
        }

        # tables
        if self.OUTPUT_CSV in paths:
            path = paths[self.OUTPUT_CSV]
            try:
                write_sectioned_csv(path, result, ctx)
                done(self.OUTPUT_CSV, path)
            except Exception as e:
                failed(self.OUTPUT_CSV, e)
        if self.OUTPUT_CLASSES in paths:
            if result.classifications is None:
                feedback.pushWarning(
                    "No classification was run, so no classification table was "
                    "written. Tick the classify option or use Preserve mode."
                )
            else:
                path = paths[self.OUTPUT_CLASSES]
                try:
                    write_classification_report(
                        path,
                        dem,
                        result.hierarchy,
                        result.classifications,
                        ira_results=result.ira_results,
                        cell_area=cell_size[0] * cell_size[1],
                    )
                    done(self.OUTPUT_CLASSES, path)
                except Exception as e:
                    failed(self.OUTPUT_CLASSES, e)

        # Word report
        if self.OUTPUT_DOCX in paths:
            path = paths[self.OUTPUT_DOCX]
            feedback.setProgressText("Writing the report")
            self._write_report(
                path,
                result,
                ctx,
                written,
                diff,
                diff_out,
                diff_nodata,
                change_path,
                dem_path,
                sinks,
                crs,
                parameters,
                context,
                feedback,
            )
            if Path(path).exists():
                outputs[self.OUTPUT_DOCX] = path

        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        return outputs

    def _write_report(
        self,
        path,
        result,
        ctx,
        written,
        diff,
        diff_out,
        diff_nodata,
        change_path,
        dem_path,
        sinks,
        crs,
        parameters,
        context,
        feedback,
    ):
        """Word report with the QGIS layout map; problems are logged, never
        fatal."""
        study_source = self.parameterAsSource(parameters, self.STUDY_AREA, context)
        interval = self.parameterAsDouble(parameters, self.CONTOUR_INTERVAL, context)
        use_osm = self.parameterAsBoolean(parameters, self.REPORT_OSM, context)
        map_messages = []
        try:
            if change_path is None:  # the map needs the change as a file
                change_path = QgsProcessingUtils.generateTempFilename(
                    "dem_elevation_change.tif", context
                )
                _write_raster(change_path, diff_out, dem_path, diff_nodata)
            study = None
            if study_source is not None:
                study = copy_source_layer(
                    study_source, "Study area", crs, context.transformContext()
                )

            surface = result.corrected.astype("float64")  # NoData as NaN
            surface[~np.isfinite(diff)] = np.nan

            def contours(extent):
                lines, warnings = dem_contours(
                    surface, ctx["geotransform"], interval, extent=extent
                )
                map_messages.extend(warnings)
                return lines

            plan_map = make_plan_map(
                change_path,
                sinks,
                crs,
                context.transformContext(),
                ctx["geotransform"],
                result.corrected.shape,
                study_area=study,
                contours=contours,
                contour_interval=interval,
                present_classes=change_classes_present(diff),
                use_osm=use_osm,
                messages=map_messages,
            )
        except Exception as e:
            plan_map = None
            map_messages.append(f"Layout map not prepared ({e}).")
        ctx = dict(ctx)
        ctx.update(
            {
                "study_area": "yes" if study_source is not None else "no",
                "contour_interval": interval,
                "outputs": list(written)
                + [(SPEC_BY_KEY[self.OUTPUT_DOCX].label, path)],
                "map_messages": map_messages,
            }
        )
        try:
            messages = write_report_docx(path, result, ctx, plan_map=plan_map)
            written.append((SPEC_BY_KEY[self.OUTPUT_DOCX].label, path))
            for message in dict.fromkeys(messages + map_messages):
                feedback.pushWarning(message)
        except ImportError as e:
            feedback.pushWarning(
                f"Word report skipped: {e}. Install python-docx and matplotlib in "
                "QGIS's Python."
            )
        except Exception as e:
            feedback.reportError(
                f"{Path(path).name} could not be written ({e}) - it may be open "
                "in Word.",
                fatalError=False,
            )
