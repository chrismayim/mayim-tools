"""Processing Toolbox algorithm: DEM in -> hydrologically corrected DEM,
remaining-sink points, a sectioned CSV and an optional Word report out.

Default is continuous drainage: every depression is resolved (filled or
breached, whichever disturbs the DEM less) and a drainage-enforcement pass
removes the artefact pits those corrections can leave behind. Genuine sinks
(pans, regional sinks, lakes) are protected only when the user supplies them
as a polygon or point layer.

Thin wrapper - all real logic lives in core/ (zero QGIS dependency,
independently testable; see tests/test_dem_hydro_processing_*.py).
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsSingleSymbolRenderer,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

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
from .report import write_report_docx

TOOL_VERSION = "1.0.1"
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


def _describe_destination(dest) -> str:
    text = str(dest or "")
    if text.startswith("memory:") or "TEMPORARY_OUTPUT" in text or not text:
        return "Temporary layer (not saved)"
    return text


class _SinkStyle(QgsProcessingLayerPostProcessorInterface):
    def postProcessLayer(self, layer, context, feedback):
        try:
            symbol = QgsMarkerSymbol.createSimple(
                {
                    "name": "circle",
                    "color": "#eb6834",
                    "size": "3",
                    "outline_color": "#0b0b0b",
                }
            )
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Sink styling skipped: {e}")


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
    template = gdal.Open(template_path)
    out = gdal.GetDriverByName("GTiff").Create(
        path, template.RasterXSize, template.RasterYSize, 1, gdal.GDT_Float32
    )
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
    OUTPUT_DEM = "OUTPUT_DEM"
    OUTPUT_SINKS = "OUTPUT_SINKS"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_DOCX = "OUTPUT_DOCX"
    OUTPUT_CLASSES = "OUTPUT_CLASSES"

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
            "vertical error is right; the Word report needs python-docx and "
            "matplotlib in QGIS's Python.\n"
            "\n"
            "OUTPUTS:\tCorrected DEM, remaining sinks (points), sectioned "
            "CSV report, optional Word report, optional classification table."
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
            QgsProcessingParameterRasterDestination(self.OUTPUT_DEM, "Corrected DEM")
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_SINKS,
                "Remaining sinks",
                type=QgsProcessing.SourceType.TypeVectorPoint,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Results report (CSV)",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_DOCX,
                "Results report (Word)",
                fileFilter="Word documents (*.docx)",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CLASSES,
                "Classification table (CSV, reusable)",
                fileFilter="CSV files (*.csv)",
                optional=True,
                createByDefault=False,
            )
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
                "see the remaining sinks layer and report section 6."
            )
        for w in result.warnings:
            feedback.pushWarning(w)

        out_dem = self.parameterAsOutputLayer(parameters, self.OUTPUT_DEM, context)
        _write_raster(out_dem, result.corrected, dem_path, nodata)

        results = {self.OUTPUT_DEM: out_dem}

        fields = QgsFields()
        for fname, ftype in [
            ("sink_id", QMetaType.Type.Int),
            ("kind", QMetaType.Type.QString),
            ("reason", QMetaType.Type.QString),
            ("cells", QMetaType.Type.Int),
            ("elev_m", QMetaType.Type.Double),
        ]:
            fields.append(QgsField(fname, ftype))
        sink, sink_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_SINKS,
            context,
            fields,
            Qgis.WkbType.Point,
            crs,
        )
        if sink is not None:
            for k in result.sinks:
                x = transform.origin_x + (k.col + 0.5) * transform.pixel_width
                y = transform.origin_y - (k.row + 0.5) * transform.pixel_height
                f = QgsFeature(fields)
                f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                f.setAttributes(
                    [k.sink_id, k.kind, k.reason, k.cells, float(k.elevation)]
                )
                sink.addFeature(f, QgsFeatureSink.Flag.FastInsert)
            results[self.OUTPUT_SINKS] = sink_id
            if context.willLoadLayerOnCompletion(sink_id):
                self._style = _SinkStyle()
                context.layerToLoadOnCompletionDetails(sink_id).setPostProcessor(
                    self._style
                )

        ctx = {
            "tool": "DEM Hydrological Processing",
            "version": TOOL_VERSION,
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dem_path": dem_path,
            "crs": crs.authid(),
            "cell_size": f"{cell_size[0]:.4g} x {cell_size[1]:.4g}",
            "grid": f"{cols} x {rows}",
            "dem_source": SOURCE_LABELS[src_key],
            "stream_layer": "yes" if streams is not None else "no",
            "keep_layer": f"yes ({len(keep)} feature(s))" if keep else "no",
            "review_table": review_path or "no",
        }
        out_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)
        write_sectioned_csv(out_csv, result, ctx)
        results[self.OUTPUT_CSV] = out_csv
        feedback.pushInfo(f"CSV report: {_describe_destination(out_csv)}")

        out_docx = self.parameterAsFileOutput(parameters, self.OUTPUT_DOCX, context)
        if out_docx:
            try:
                write_report_docx(out_docx, result, ctx)
                results[self.OUTPUT_DOCX] = out_docx
                feedback.pushInfo(f"Word report: {out_docx}")
            except ImportError as e:
                feedback.pushWarning(
                    f"Word report skipped: {e}. Install python-docx and matplotlib."
                )

        out_cls = self.parameterAsFileOutput(parameters, self.OUTPUT_CLASSES, context)
        if out_cls:
            if result.classifications is None:
                feedback.pushWarning(
                    "No classification was run, so no classification table was "
                    "written. Tick the classify option or use Preserve mode."
                )
            else:
                write_classification_report(
                    out_cls,
                    dem,
                    result.hierarchy,
                    result.classifications,
                    ira_results=result.ira_results,
                    cell_area=cell_size[0] * cell_size[1],
                )
                results[self.OUTPUT_CLASSES] = out_cls
                feedback.pushInfo(f"Classification table: {out_cls}")
        return results
