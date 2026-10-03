"""Processing Toolbox algorithm "DEM depression stage-storage": DEM +
storage-area polygon(s) in -> stage-area-volume table, HEC-RAS / HEC-HMS
elevation-volume table, water-surface polygons, top-stage depth raster,
spill points and a Word report out. (Package name terrain_storage kept.)

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_terrain_storage_core.py).
"""

from datetime import datetime
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
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
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools.hydrology._common.grid_polygons import rings_to_multipolygon_wkt

from .core import (
    DEFAULT_DEPTH_NODATA,
    StorageArea,
    StorageError,
    check_geotransform,
    compute_storage,
    mosaic_depths,
    polygon_window,
    raster_info,
    read_window,
    window_geotransform,
    write_raster,
)
from .export import write_hec_table, write_stage_table_csv
from .report import write_report_docx

TOOL_VERSION = "0.2.0"

MODE_OPTIONS = [
    "Connected to the lowest point (pool fills over ridges in turn)",
    "All cells below the stage (level pool - HEC-RAS storage-area convention)",
]
MODE_KEYS = ["connected", "all"]

COVERAGE_OPTIONS = [
    "Fractional cell coverage (8 x 8 sub-cells)",
    "Cell centre inside polygon",
]
COVERAGE_FACTORS = [8, 1]

_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

SPILL_FIELDS = [
    ("storage_id", _S, "storage area name (ID field, or feature id)"),
    ("src_fid", _I, "source feature id"),
    ("mode", _S, "connected or all"),
    ("z_floor", _D, "lowest ground inside the polygon, m"),
    ("z_spill", _D, "spill level (pool first reaches the polygon edge), m"),
    ("z_top", _D, "top stage used for the table, m"),
    ("top_src", _S, "spill, user (global value) or field"),
    ("max_depth", _D, "z_top - z_floor, m"),
    ("area_m2", _D, "water-surface area at z_top, m2"),
    ("area_ha", _D, "water-surface area at z_top, ha"),
    ("vol_m3", _D, "storage volume at z_top, m3"),
    ("vol_ML", _D, "storage volume at z_top, ML"),
    ("mean_dep", _D, "mean depth vol_m3 / area_m2 at z_top, m"),
    ("poly_ha", _D, "polygon area on the DEM grid, ha"),
    ("wet_pct", _D, "water-surface area as % of the polygon area"),
    ("n_stages", _I, "rows in the stage table"),
    ("cell_m", _D, "DEM cell size, m"),
    ("x_floor", _D, "lowest cell centre X"),
    ("y_floor", _D, "lowest cell centre Y"),
]

FLOOR_FIELDS = [
    ("storage_id", _S, ""),
    ("src_fid", _I, ""),
    ("z_floor", _D, ""),
    ("x_floor", _D, ""),
    ("y_floor", _D, ""),
    ("z_top", _D, ""),
    ("max_depth", _D, ""),
]

CONTOUR_FIELDS = [
    ("storage_id", _S, ""),
    ("kind", _S, ""),
    ("stage_m", _D, ""),
    ("depth_m", _D, ""),
    ("d_top_m", _D, ""),
    ("length_m", _D, ""),
]

WS_FIELDS = [
    ("storage_id", _S, ""),
    ("stage_m", _D, ""),
    ("depth_m", _D, ""),
    ("area_m2", _D, ""),
    ("area_ha", _D, ""),
    ("volume_m3", _D, ""),
    ("volume_ML", _D, ""),
]


STYLES_DIR = Path(__file__).resolve().parent / "styles"
STYLE_DEPTH = "depth_top_stage.qml"
STYLE_FLOOR = "floor_points.qml"
STYLE_SPILL = "spill_points.qml"
STYLE_CONTOURS = "water_edge_contours.qml"
STYLE_WS = "water_surface_polygons.qml"


def _rescale_pseudocolor(layer, lo: float, hi: float) -> None:
    """Stretches a single-band pseudocolour style's ramp (saved for another
    data range) to [lo, hi], keeping its colours and their spacing."""
    renderer = layer.renderer()
    if not isinstance(renderer, QgsSingleBandPseudoColorRenderer) or hi <= lo:
        return
    fn = renderer.shader().rasterShaderFunction()
    if not isinstance(fn, QgsColorRampShader):
        return
    items = fn.colorRampItemList()
    old_lo, old_hi = fn.minimumValue(), fn.maximumValue()
    span = (old_hi - old_lo) or 1.0
    new_items = []
    for item in items:
        value = lo + (item.value - old_lo) / span * (hi - lo)
        new_items.append(
            QgsColorRampShader.ColorRampItem(value, item.color, f"{value:.2f}")
        )
    fn.setColorRampItemList(new_items)
    fn.setMinimumValue(lo)
    fn.setMaximumValue(hi)
    renderer.setClassificationMin(lo)
    renderer.setClassificationMax(hi)


def _rebuild_categories(layer) -> None:
    """Re-creates a categorised style's classes for the values actually in
    the layer (the saved style lists another run's stage values), using
    the style's own source symbol and colour ramp."""
    renderer = layer.renderer()
    if not isinstance(renderer, QgsCategorizedSymbolRenderer):
        return
    field_name = renderer.classAttribute()
    index = layer.fields().indexOf(field_name)
    if index < 0:
        return
    values = sorted(v for v in layer.uniqueValues(index) if v is not None)
    base = renderer.sourceSymbol()
    if base is None and renderer.categories():
        base = renderer.categories()[0].symbol()
    if base is None or not values:
        return
    categories = [
        QgsRendererCategory(
            v, base.clone(), f"{v:.2f}" if isinstance(v, float) else str(v)
        )
        for v in values
    ]
    new = QgsCategorizedSymbolRenderer(field_name, categories)
    new.setSourceSymbol(base.clone())
    ramp = renderer.sourceColorRamp()
    if ramp is not None:
        new.updateColorRamp(ramp.clone())
    layer.setRenderer(new)


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Loads a saved QML style onto an output layer once Processing has
    loaded it into the project (same pattern as the ERA5 / Design Rainfall
    tools), then adapts data-specific parts: a pseudocolour ramp is
    stretched to this run's depth range and categorised classes are
    rebuilt for this run's values."""

    def __init__(self, style_path: Path, raster_range=None, recategorise=False):
        super().__init__()
        self.style_path = str(style_path)
        self.raster_range = raster_range
        self.recategorise = recategorise

    def postProcessLayer(self, layer, context, feedback):
        layer.loadNamedStyle(self.style_path)
        try:
            if self.raster_range is not None:
                _rescale_pseudocolor(layer, *self.raster_range)
            if self.recategorise:
                _rebuild_categories(layer)
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style adjustment skipped: {e}")
        layer.triggerRepaint()


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


class DemDepressionStageStorageAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    STORAGE = "STORAGE"
    ID_FIELD = "ID_FIELD"
    INTERVAL = "INTERVAL"
    MAX_STAGE = "MAX_STAGE"
    MAX_STAGE_FIELD = "MAX_STAGE_FIELD"
    MODE = "MODE"
    COVERAGE = "COVERAGE"
    POLY_INTERVAL = "POLY_INTERVAL"
    OUTPUT_TABLE = "OUTPUT_TABLE"
    OUTPUT_HEC = "OUTPUT_HEC"
    OUTPUT_WS = "OUTPUT_WS"
    OUTPUT_CONTOURS = "OUTPUT_CONTOURS"
    CONTOUR_SMOOTH = "CONTOUR_SMOOTH"
    CONTOUR_INTERVAL = "CONTOUR_INTERVAL"
    OUTPUT_FLOOR = "OUTPUT_FLOOR"
    OUTPUT_DEPTH = "OUTPUT_DEPTH"
    OUTPUT_SPILL = "OUTPUT_SPILL"
    OUTPUT_DOCX = "OUTPUT_DOCX"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return DemDepressionStageStorageAlgorithm()

    def name(self):
        return "dem_depression_stage_storage"

    def displayName(self):
        return "DEM depression stage-storage"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        glossary = "\n".join(f"  {n}: {d}" for n, _, d in SPILL_FIELDS)
        return (
            f"DEM depression stage-storage (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tStage-storage (elevation-area-volume) curves for ponds, "
            "dams, detention basins and natural depressions, read straight off "
            "a DEM inside each storage-area polygon.\n"
            "\n"
            "INPUTS:\tAn UNFILLED DEM in a projected CRS in metres (filling "
            "or breaching removes the depressions being measured), and one or "
            "more storage-area polygons (reprojected to the DEM CRS if needed). "
            "Draw each polygon along or just beyond the crest / embankment. "
            "Holes (islands) and multipart polygons are supported.\n"
            "\n"
            "NO DEPRESSION:\tIf the lowest ground inside a polygon is on its "
            "edge, the polygon holds no water before it spills. That polygon "
            "is skipped with a message, and the tool fails (no output layers) "
            "when no polygon encloses a depression. Set a maximum stage to "
            "compute storage behind the polygon edge instead (e.g. a proposed "
            "embankment).\n"
            "\n"
            "METHOD:\tEvery DEM cell is a flat-topped prism. At water level h, "
            "a wet cell stores (h - z) x cell area x the fraction of the cell "
            "inside the polygon; the water-surface area is the sum of the wet "
            "cell areas. This is exact for the DEM as given - no average-end-"
            "area or conic approximation.\n"
            "\n"
            "CONNECTIVITY:\t'Connected' (default) only counts cells joined to "
            "the lowest point at that level, so a hollow behind a ridge fills "
            "only once the pool overtops the ridge. 'All cells' counts every "
            "cell below the level, as a HEC-RAS storage area does; the share of "
            "volume in disconnected hollows is reported when it exceeds 5%.\n"
            "\n"
            "TOP STAGE:\tBy default the table runs from the lowest cell up to "
            "the SPILL LEVEL - the lowest level at which the pool reaches the "
            "polygon edge (minimax flood path from the lowest cell). A polygon "
            "drawn beyond the crest still finds the true crest; one drawn "
            "inside it is flagged. A maximum stage (global value, or a per-"
            "feature field that overrides it) replaces the spill level, e.g. "
            "for a proposed embankment; above the spill level the polygon edge "
            "then acts as a vertical wall, which is flagged.\n"
            "\n"
            "OUTPUTS:\tStage table CSV (stage, depth, area m2/ha, volume m3/ML, "
            "incremental volume, wet cells). HEC-RAS / HEC-HMS table: tab-"
            "delimited elevation (m), volume (1000 m3), area (1000 m2) blocks "
            "to paste into a RAS storage area or an HMS reservoir. Water-"
            "surface polygons at the chosen spacing (plus the top stage), "
            "clipped to the storage polygon. Water-edge contours: smooth lines "
            "at round DEPTHS above the floor (floor + k x contour interval), "
            "plus a final line at the spill level (kind = depth / spill / top) "
            "- the water's edge "
            "interpolated between cell centres (marching squares) and rounded "
            "with Chaikin corner cutting - with depth_m (pool depth at that "
            "stage) and d_top_m (water depth along the line at the top stage, "
            "so the set doubles as depth contours of the full pool). Depth "
            "raster at the top stage. "
            "Spill points at the controlling crest with a summary per storage. "
            "Floor points at the lowest DEM cell centre in each storage area. "
            "Outputs load with the standard Mayim 'DEM Stage Storage' styles; "
            "the depth colour ramp is stretched to each run's depth range and "
            "the water-surface classes are rebuilt for its stages. "
            "Optional Word (.docx) report: method, inputs, definitions, and per "
            "storage area a description, key results, the stage-area-volume "
            "and depth-distribution tables and figures (depth map, elevation-"
            "area-capacity, stage-storage, stage-area, depth-area and depth-"
            "volume curves, depth distribution). It needs python-docx (and "
            "matplotlib for figures) in QGIS's Python.\n"
            "\n"
            "SPILL POINT ATTRIBUTES:\n" + glossary
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.DEM, "DEM (projected, metres)")
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STORAGE,
                "Storage-area polygon(s)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ID_FIELD,
                "Storage name field (optional)",
                parentLayerParameterName=self.STORAGE,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INTERVAL,
                "Stage interval (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.001,
                defaultValue=0.1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_STAGE,
                "Maximum stage (m; blank = spill level)",
                type=QgsProcessingParameterNumber.Type.Double,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.MAX_STAGE_FIELD,
                "Maximum stage field (optional; overrides the value above)",
                parentLayerParameterName=self.STORAGE,
                type=Qgis.ProcessingFieldParameterDataType.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE, "Cells counted", options=MODE_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.COVERAGE,
                "Polygon edge cells",
                options=COVERAGE_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.POLY_INTERVAL,
                "Water-surface polygon spacing (m; 0 = every table stage)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONTOUR_INTERVAL,
                "Smoothed contour interval (m; 0 = every table stage)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONTOUR_SMOOTH,
                "Contour smoothing passes (Chaikin; 0 = none)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                maxValue=8,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_TABLE,
                "Stage-area-volume table (CSV)",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_HEC,
                "HEC-RAS / HEC-HMS elevation-volume table (TXT)",
                fileFilter="Text files (*.txt)",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_WS,
                "Water-surface polygons",
                type=QgsProcessing.SourceType.TypeVectorPolygon,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_CONTOURS,
                "Water-edge contours (smoothed)",
                type=QgsProcessing.SourceType.TypeVectorLine,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_DEPTH, "Depth at top stage", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_SPILL,
                "Spill points (summary per storage)",
                type=QgsProcessing.SourceType.TypeVectorPoint,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_FLOOR,
                "Floor points (lowest ground per storage)",
                type=QgsProcessing.SourceType.TypeVectorPoint,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_DOCX,
                "Stage-storage report (Word)",
                fileFilter="Word documents (*.docx)",
                optional=True,
                createByDefault=False,
            )
        )

    # ------------------------------------------------------------------

    def _read_areas(self, source, id_field, stage_field, crs, context, feedback):
        transform = None
        if source.sourceCrs() != crs:
            transform = QgsCoordinateTransform(
                source.sourceCrs(), crs, context.transformContext()
            )
        areas = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            geom.convertToStraightSegment()
            if not geom.isGeosValid():
                geom = geom.makeValid()
                geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
            if geom.type() != Qgis.GeometryType.Polygon or geom.isEmpty():
                feedback.pushWarning(f"Feature {feature.id()} skipped: not a polygon.")
                continue
            value = feature[id_field] if id_field else None
            name = str(feature.id() if _is_null(value) else value)
            polys = geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
            polygons = [
                [[(p.x(), p.y()) for p in ring] for ring in poly if ring]
                for poly in polys
                if poly
            ]
            stage = None
            if stage_field and not _is_null(feature[stage_field]):
                stage = float(feature[stage_field])
            areas.append((StorageArea(name, polygons, int(feature.id())), geom, stage))
        return areas

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
                "The DEM is in a geographic CRS (degrees). Areas and volumes "
                "need a projected CRS in metres - reproject the DEM first."
            )
        source = self.parameterAsSource(parameters, self.STORAGE, context)
        if source is None:
            raise QgsProcessingException("No storage-area polygon layer provided.")
        id_field = self.parameterAsString(parameters, self.ID_FIELD, context)
        stage_field = self.parameterAsString(parameters, self.MAX_STAGE_FIELD, context)
        interval = self.parameterAsDouble(parameters, self.INTERVAL, context)
        raw_max = parameters.get(self.MAX_STAGE)
        global_max = (
            None
            if _is_null(raw_max)
            else self.parameterAsDouble(parameters, self.MAX_STAGE, context)
        )
        mode = MODE_KEYS[self.parameterAsEnum(parameters, self.MODE, context)]
        supersample = COVERAGE_FACTORS[
            self.parameterAsEnum(parameters, self.COVERAGE, context)
        ]
        poly_interval = self.parameterAsDouble(parameters, self.POLY_INTERVAL, context)
        smoothing = self.parameterAsInt(parameters, self.CONTOUR_SMOOTH, context)
        contour_interval = self.parameterAsDouble(
            parameters, self.CONTOUR_INTERVAL, context
        )
        table_path = self.parameterAsFileOutput(parameters, self.OUTPUT_TABLE, context)
        hec_path = self.parameterAsFileOutput(parameters, self.OUTPUT_HEC, context)
        depth_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_DEPTH, context)
        docx_path = self.parameterAsFileOutput(parameters, self.OUTPUT_DOCX, context)

        areas = self._read_areas(source, id_field, stage_field, crs, context, feedback)
        if not areas:
            raise QgsProcessingException("The storage-area layer has no polygons.")

        try:
            shape, gt, projection, nodata = raster_info(dem_layer.source())
            check_geotransform(gt)
        except StorageError as e:
            raise QgsProcessingException(str(e)) from e
        feedback.pushInfo(
            f"{len(areas)} storage area(s), DEM cell {abs(gt[1]):g} x "
            f"{abs(gt[5]):g} m, mode: {mode}, stage interval {interval:g} m"
        )

        results, geoms, failures = [], {}, []
        for i, (area, geom, field_stage) in enumerate(areas):
            if feedback.isCanceled():
                return {}
            feedback.setProgress(int(100 * i / len(areas)))
            feedback.setProgressText(f"Storage '{area.storage_id}'")
            window = polygon_window(area.polygons, gt, shape)
            if window is None:
                failures.append(f"Storage '{area.storage_id}': outside the DEM.")
                feedback.pushWarning(f"Skipped - {failures[-1]}")
                continue
            max_stage = field_stage if field_stage is not None else global_max
            try:
                dem = read_window(dem_layer.source(), window)
                result = compute_storage(
                    dem,
                    nodata,
                    window_geotransform(gt, window[0], window[1]),
                    window,
                    area,
                    interval,
                    max_stage=max_stage,
                    mode=mode,
                    supersample=supersample,
                    polygon_interval=poly_interval,
                    contour_interval=contour_interval,
                    contour_smoothing=smoothing,
                    is_canceled=feedback.isCanceled,
                )
            except StorageError as e:
                failures.append(str(e))
                feedback.pushWarning(f"Skipped - {e}")
                continue
            if field_stage is not None:
                result.top_source = "field"
                result.attributes["top_src"] = "field"
            for message in result.warnings:
                feedback.pushWarning(message)
            a = result.attributes
            spill = "none" if a["z_spill"] is None else f"{a['z_spill']:.3f} m"
            feedback.pushInfo(
                f"{result.storage_id}: floor {a['z_floor']:.3f} m, spill {spill}, "
                f"top {a['z_top']:.3f} m ({a['top_src']}), "
                f"{a['vol_m3']:,.1f} m3 over {a['area_ha']:.4f} ha"
            )
            results.append(result)
            geoms[id(result)] = geom

        if feedback.isCanceled():
            return {}
        if not results:
            detail = "\n".join(failures) or "see the messages above."
            raise QgsProcessingException(
                "No stage-storage could be computed, so no outputs were "
                f"written:\n{detail}"
            )

        outputs = {}
        write_stage_table_csv(table_path, results)
        outputs[self.OUTPUT_TABLE] = table_path
        if hec_path:
            write_hec_table(hec_path, results)
            outputs[self.OUTPUT_HEC] = hec_path

        contour_id = self._write_contours(parameters, context, results, geoms, crs)
        if contour_id is not None:
            outputs[self.OUTPUT_CONTOURS] = contour_id
            self._register_style(context, contour_id, STYLE_CONTOURS)
        ws_id = self._write_water_surfaces(
            parameters, context, feedback, results, geoms, crs
        )
        if ws_id is not None:
            outputs[self.OUTPUT_WS] = ws_id
            self._register_style(context, ws_id, STYLE_WS, recategorise=True)
        spill_id = self._write_spill_points(parameters, context, results, crs)
        if spill_id is not None:
            outputs[self.OUTPUT_SPILL] = spill_id
            self._register_style(context, spill_id, STYLE_SPILL)
        floor_id = self._write_floor_points(parameters, context, results, crs)
        if floor_id is not None:
            outputs[self.OUTPUT_FLOOR] = floor_id
            self._register_style(context, floor_id, STYLE_FLOOR)

        if depth_path:
            merged = mosaic_depths(results, DEFAULT_DEPTH_NODATA)
            if merged is not None:
                array, (r0, c0) = merged
                write_raster(
                    depth_path,
                    array,
                    window_geotransform(gt, r0, c0),
                    projection,
                    DEFAULT_DEPTH_NODATA,
                )
                outputs[self.OUTPUT_DEPTH] = depth_path
                valid = array[array != DEFAULT_DEPTH_NODATA]
                depth_range = (
                    (float(valid.min()), float(valid.max())) if valid.size else None
                )
                self._register_style(
                    context, depth_path, STYLE_DEPTH, raster_range=depth_range
                )

        if docx_path:
            feedback.setProgressText("Writing report")
            run_warnings = list(failures)
            for r in results:
                run_warnings.extend(r.warnings)
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
                "storage": {
                    "layer": source.sourceName(),
                    "n_features": len(areas),
                    "id_field": id_field,
                },
                "interval": f"{interval:g}",
                "max_stage": (
                    f"Field '{stage_field}'"
                    + ("" if global_max is None else f", else {global_max:g} m")
                    if stage_field
                    else (None if global_max is None else f"{global_max:g} m")
                ),
                "mode": mode,
                "coverage": COVERAGE_OPTIONS[
                    self.parameterAsEnum(parameters, self.COVERAGE, context)
                ],
                "outputs": [
                    (label, _describe_destination(outputs.get(key)))
                    for key, label in (
                        (self.OUTPUT_TABLE, "Stage-area-volume table (CSV)"),
                        (self.OUTPUT_HEC, "HEC-RAS / HEC-HMS table"),
                        (self.OUTPUT_WS, "Water-surface polygons"),
                        (self.OUTPUT_CONTOURS, "Water-edge contours (smoothed)"),
                        (self.OUTPUT_DEPTH, "Depth at top stage"),
                        (self.OUTPUT_SPILL, "Spill points"),
                        (self.OUTPUT_FLOOR, "Floor points"),
                    )
                    if outputs.get(key)
                ]
                + [("This report", docx_path)],
                "warnings": run_warnings,
            }
            try:
                write_report_docx(docx_path, results, context_info)
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

    # ------------------------------------------------------------------

    def _write_water_surfaces(self, parameters, context, feedback, results, geoms, crs):
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_WS,
            context,
            _fields(WS_FIELDS),
            Qgis.WkbType.MultiPolygon,
            crs,
        )
        if sink is None:
            return None
        for r in results:
            clip = geoms[id(r)]
            # highest (largest) stage first, so smaller extents draw on top
            for stage, row, polys in reversed(r.ws_polygons):
                geom = QgsGeometry.fromWkt(rings_to_multipolygon_wkt(polys))
                geom = geom.makeValid().intersection(clip)
                if QgsWkbTypes.flatType(geom.wkbType()) not in (
                    Qgis.WkbType.Polygon,
                    Qgis.WkbType.MultiPolygon,
                ):
                    geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
                if geom.isEmpty():
                    continue
                geom.convertToMultiType()
                feat = QgsFeature(_fields(WS_FIELDS))
                feat.setGeometry(geom)
                feat["storage_id"] = r.storage_id
                feat["stage_m"] = round(stage, 4)
                for name in ("depth_m", "area_m2", "area_ha", "volume_m3"):
                    feat[name] = row[name]
                feat["volume_ML"] = row["volume_ML"]
                sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_contours(self, parameters, context, results, geoms, crs):
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_CONTOURS,
            context,
            _fields(CONTOUR_FIELDS),
            Qgis.WkbType.MultiLineString,
            crs,
        )
        if sink is None:
            return None
        for r in results:
            clip = geoms[id(r)]
            for stage, row, lines in r.contours:
                parts = [[QgsPointXY(x, y) for x, y in line] for line in lines]
                geom = QgsGeometry.fromMultiPolylineXY(parts).intersection(clip)
                if QgsWkbTypes.flatType(geom.wkbType()) not in (
                    Qgis.WkbType.LineString,
                    Qgis.WkbType.MultiLineString,
                ):
                    geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Line)
                if geom.isEmpty():
                    continue
                geom.convertToMultiType()
                feat = QgsFeature(_fields(CONTOUR_FIELDS))
                feat.setGeometry(geom)
                feat["storage_id"] = r.storage_id
                feat["kind"] = row.get("kind", "depth")
                feat["stage_m"] = round(stage, 4)
                feat["depth_m"] = row["depth_m"]
                feat["d_top_m"] = round(r.top_stage - stage, 4)
                feat["length_m"] = round(geom.length(), 2)
                sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _register_style(self, context, dest_id, style_filename, **options):
        """Attaches a saved QML style to an output layer via a post-processor
        (kept alive on self._post_processors - Processing holds only a weak
        reference). Skipped when the output is not being loaded."""
        if not dest_id or not context.willLoadLayerOnCompletion(dest_id):
            return
        details = context.layerToLoadOnCompletionDetails(dest_id)
        processor = _StylePostProcessor(STYLES_DIR / style_filename, **options)
        details.setPostProcessor(processor)
        self._post_processors.append(processor)

    def _write_floor_points(self, parameters, context, results, crs):
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_FLOOR,
            context,
            _fields(FLOOR_FIELDS),
            Qgis.WkbType.Point,
            crs,
        )
        if sink is None:
            return None
        for r in results:
            a = r.attributes
            feat = QgsFeature(_fields(FLOOR_FIELDS))
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*r.floor_xy)))
            feat["storage_id"] = r.storage_id
            feat["src_fid"] = r.src_fid
            feat["z_floor"] = a["z_floor"]
            feat["x_floor"] = float(r.floor_xy[0])
            feat["y_floor"] = float(r.floor_xy[1])
            feat["z_top"] = a["z_top"]
            feat["max_depth"] = a["max_depth"]
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_spill_points(self, parameters, context, results, crs):
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_SPILL,
            context,
            _fields(SPILL_FIELDS),
            Qgis.WkbType.Point,
            crs,
        )
        if sink is None:
            return None
        for r in results:
            xy = r.spill_xy if r.spill_xy is not None else r.floor_xy
            feat = QgsFeature(_fields(SPILL_FIELDS))
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*xy)))
            for name, _, _ in SPILL_FIELDS:
                value = r.attributes.get(name)
                feat[name] = float(value) if isinstance(value, float) else value
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id
