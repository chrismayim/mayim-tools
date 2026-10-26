"""Processing Toolbox algorithm "DEM depression stage-storage": DEM +
storage-area polygon(s) in -> stage-area-volume table, HEC-RAS / HEC-HMS
elevation-volume table, water-surface polygons, top-stage depth raster,
spill points and a Word report out. Optionally (DEM surface is a water
surface) the bed below the surveyed water is estimated first - see
bathymetry.py - and the curve runs from the estimated bed. (Package name
terrain_storage kept.)

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_terrain_storage_core.py).

Outputs follow the Mayim output standard: a tick box per output, raster,
shapefile, CSV and Word folders, files named <project number>_<description>,
layers loaded with the Mayim 'DEM Stage Storage' styles; a locked file is
reported and the other outputs are still written.
"""

import math
import os
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
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
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
from mayim_tools.hydrology._common.grid_polygons import (
    group_polygons,
    ring_to_map,
    rings_to_multipolygon_wkt,
    simplify_ring,
    trace_rings,
)

from .bathymetry import DEFAULT_FLAT_TOL_M, write_bed_dem
from .bathymetry_export import write_bathymetry_summary_csv, write_profile_csv
from .core import (
    DEFAULT_DEPTH_NODATA,
    StorageArea,
    StorageError,
    check_geotransform,
    compute_storage,
    compute_storage_with_bed,
    mosaic_depths,
    polygon_window,
    raster_info,
    read_window,
    window_geotransform,
    write_raster,
)
from .export import write_hec_table, write_stage_table_csv
from .report import write_report_docx

TOOL_VERSION = "0.4.0"

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
    ("z_survey", _D, "bed estimate: water level at the time of survey, m"),
    ("bed_basis", _S, "bed estimate: user / valley / slope+hypsometry / slope"),
    ("ws_ha", _D, "bed estimate: surveyed water-surface area, ha"),
    ("vws_lo", _D, "bed estimate: volume below the surveyed water - low, m3"),
    ("vws_c", _D, "bed estimate: volume below the surveyed water - central, m3"),
    ("vws_hi", _D, "bed estimate: volume below the surveyed water - high, m3"),
    ("vol_lo", _D, "bed estimate: storage volume at z_top - low, m3"),
    ("vol_hi", _D, "bed estimate: storage volume at z_top - high, m3"),
]

SURVEY_FIELDS = [
    ("storage_id", _S, ""),
    ("src_fid", _I, ""),
    ("water_lvl", _D, ""),
    ("water_ha", _D, ""),
    ("basis", _S, ""),
    ("dmax_lo", _D, ""),
    ("dmax_c", _D, ""),
    ("dmax_hi", _D, ""),
    ("dmean_lo", _D, ""),
    ("dmean_c", _D, ""),
    ("dmean_hi", _D, ""),
    ("vol_lo", _D, ""),
    ("vol_c", _D, ""),
    ("vol_hi", _D, ""),
    ("d_slope", _D, ""),
    ("d_hyps", _D, ""),
    ("d_valley", _D, ""),
    ("d_user", _D, ""),
    ("slope_pc", _D, ""),
    ("shape_n", _D, ""),
    ("hl_depth", _D, ""),
    ("hl_vol_m3", _D, ""),
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


# Selectable outputs. The HEC-RAS / HEC-HMS table is tab-delimited text
# (.txt) in the CSV folder; the bed-estimate outputs apply only when 'DEM
# surface is a water surface' is ticked.
OUTPUT_SPECS = [
    OutputSpec(
        "OUTPUT_TABLE", "Stage-area-volume table (CSV)", "Stage Area Volume", CSV
    ),
    OutputSpec(
        "OUTPUT_HEC", "HEC-RAS / HEC-HMS elevation-volume table (TXT)",
        "HEC Elevation Volume", CSV, False,
    ),
    OutputSpec(
        "OUTPUT_WS", "Water-surface polygons (shapefile)", "Water Surface Polygons",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_CONTOURS", "Water-edge contours, smoothed (shapefile)",
        "Water Edge Contours", SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_DEPTH", "Depth at the top stage (raster)", "Depth Top Stage", RASTER
    ),
    OutputSpec(
        "OUTPUT_SPILL", "Spill points, summary per storage (shapefile)",
        "Spill Points", SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_FLOOR", "Floor points, lowest ground per storage (shapefile)",
        "Floor Points", SHAPEFILE,
    ),
    OutputSpec("OUTPUT_DOCX", "Stage-storage report (Word)", "Stage Storage Report",
               WORD),
    OutputSpec(
        "OUTPUT_BED_DEM", "Bed estimate: DEM with estimated bed, central (raster)",
        "Bed DEM Central", RASTER, False,
    ),
    OutputSpec(
        "OUTPUT_BED_DEM_LOW", "Bed estimate: DEM with estimated bed, low - shallow "
        "(raster)", "Bed DEM Low", RASTER, False,
    ),
    OutputSpec(
        "OUTPUT_BED_DEM_HIGH", "Bed estimate: DEM with estimated bed, high - deep "
        "(raster)", "Bed DEM High", RASTER, False,
    ),
    OutputSpec(
        "OUTPUT_SURVEY_WATER", "Bed estimate: surveyed water surface with the "
        "estimates (shapefile)", "Surveyed Water Surface", SHAPEFILE, False,
    ),
    OutputSpec(
        "OUTPUT_BED_TABLE", "Bed estimate: summary (CSV)", "Bed Estimate Summary",
        CSV, False,
    ),
    OutputSpec(
        "OUTPUT_PROFILE", "Bed estimate: centreline profiles (CSV)",
        "Bed Centreline Profiles", CSV, False,
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
BED_KEYS = (
    "OUTPUT_BED_DEM",
    "OUTPUT_BED_DEM_LOW",
    "OUTPUT_BED_DEM_HIGH",
    "OUTPUT_SURVEY_WATER",
    "OUTPUT_BED_TABLE",
    "OUTPUT_PROFILE",
)
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV and text tables"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


def output_path(spec: OutputSpec, folders: dict, project_number: str) -> str:
    path = spec.path(folders, project_number)
    if spec.key == "OUTPUT_HEC":
        path = str(Path(path).with_suffix(".txt"))
    return path


def _software_version() -> str:
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
    ESTIMATE_BED = "ESTIMATE_BED"
    CENTRELINES = "CENTRELINES"
    DEPTH_FIELD = "DEPTH_FIELD"
    FLAT_TOL = "FLAT_TOL"
    WATER_LEVEL = "WATER_LEVEL"
    SHORE_BUFFER = "SHORE_BUFFER"
    REACH = "REACH"
    HYDROLAKES = "HYDROLAKES"
    OUTPUT_BED_DEM = "OUTPUT_BED_DEM"
    OUTPUT_BED_DEM_LOW = "OUTPUT_BED_DEM_LOW"
    OUTPUT_BED_DEM_HIGH = "OUTPUT_BED_DEM_HIGH"
    OUTPUT_SURVEY_WATER = "OUTPUT_SURVEY_WATER"
    OUTPUT_BED_TABLE = "OUTPUT_BED_TABLE"
    OUTPUT_PROFILE = "OUTPUT_PROFILE"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

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
        files = "\n".join(
            f"  {s.label}: <project number>_{clean_name_part(s.description)}"
            + (".txt" if s.key == "OUTPUT_HEC" else EXTENSIONS[s.kind])
            for s in OUTPUT_SPECS
        )
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
            "Word report (Mayim report standard): title page, summary, method, "
            "inputs, parameters and equations, and per storage area a "
            "description, key results, the stage-area-volume and depth-"
            "distribution tables and figures (depth map, elevation-area-"
            "capacity, stage-storage, stage-area, depth-area and depth-volume "
            "curves, depth distribution), outputs, notes and references. It "
            "needs python-docx (and matplotlib for figures) in QGIS's Python.\n"
            "\n"
            "FILES:\tTick the outputs wanted and choose the raster, shapefile, "
            "CSV and Word folders. Files are named <project number>_"
            "<description> and replace files of the same name; a file that is "
            "open (a layer loaded in QGIS, a CSV in Excel) cannot be replaced "
            "and is reported, and the other outputs are still written. The CSVs "
            "are UTF-8 with a byte-order mark and start with a column key ('#' "
            "lines).\n" + files + "\n"
            "\n"
            "BED BELOW A WATER SURFACE:\tTick 'DEM surface is a water surface' "
            "when the DEM shows the water of a pan, pond or dam as surveyed (LiDAR, "
            "photogrammetry and radar do not see the bed). The bed below the "
            "flat water surface is then estimated as LOW / CENTRAL / HIGH and "
            "written into the DEM before the curve is computed, so the table, "
            "contours and report start at the estimated bed; low and high "
            "volumes are added to the table. Methods: shoreline slope "
            "(Hollister et al., 2011; usually an upper bound for pans), "
            "hypsometry of the dry ground above the water (after Heathcote et "
            "al., 2015) and, with a stream centreline for a built dam, the river "
            "bed upstream of the water and below the wall (V-shaped valley). A "
            "known maximum depth field overrides them (+-20%). The settings are "
            "under Advanced parameters. Outputs: DEMs with the estimated bed "
            "(central, low, high), the surveyed water surface with all "
            "estimates, a bed-estimate summary CSV and a centreline profile CSV "
            "(tick them under the outputs). "
            "It is an estimate for storage modelling, not a survey.\n"
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
            QgsProcessingParameterBoolean(
                self.ESTIMATE_BED,
                "DEM surface is a water surface: estimate the bed below it",
                defaultValue=False,
            )
        )
        advanced = [
            QgsProcessingParameterFeatureSource(
                self.CENTRELINES,
                "Bed estimate: stream centreline(s) for built dams (optional)",
                types=[QgsProcessing.SourceType.TypeVectorLine],
                optional=True,
            ),
            QgsProcessingParameterField(
                self.DEPTH_FIELD,
                "Bed estimate: known maximum depth field (optional; m below the "
                "water level)",
                parentLayerParameterName=self.STORAGE,
                type=Qgis.ProcessingFieldParameterDataType.Numeric,
                optional=True,
            ),
            QgsProcessingParameterNumber(
                self.FLAT_TOL,
                "Bed estimate: flat-water tolerance (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.001,
                defaultValue=DEFAULT_FLAT_TOL_M,
            ),
            QgsProcessingParameterNumber(
                self.WATER_LEVEL,
                "Bed estimate: water level (m; blank = detect the flat water)",
                type=QgsProcessingParameterNumber.Type.Double,
                optional=True,
            ),
            QgsProcessingParameterNumber(
                self.SHORE_BUFFER,
                "Bed estimate: shore slope buffer (m; 0 = automatic)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
            ),
            QgsProcessingParameterNumber(
                self.REACH,
                "Bed estimate: centreline reach outside the water (m; 0 = automatic)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
            ),
            QgsProcessingParameterFeatureSource(
                self.HYDROLAKES,
                "Bed estimate: HydroLAKES polygons for a check (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            ),
        ]
        for param in advanced:
            param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
            self.addParameter(param)
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project name (report title page, optional)",
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

    def _read_areas(
        self, source, id_field, stage_field, crs, context, feedback, depth_field=None
    ):
        """Storage polygons as (StorageArea, geometry, maximum stage), plus a
        dict id(StorageArea) -> known maximum depth (bed estimate)."""
        transform = None
        if source.sourceCrs() != crs:
            transform = QgsCoordinateTransform(
                source.sourceCrs(), crs, context.transformContext()
            )
        areas, depths = [], {}
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
            area = StorageArea(name, polygons, int(feature.id()))
            areas.append((area, geom, stage))
            if depth_field:
                depths[id(area)] = feature[depth_field]
        return areas, depths

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
        estimate_bed = self.parameterAsBoolean(parameters, self.ESTIMATE_BED, context)
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()
        project_name = (
            self.parameterAsString(parameters, self.PROJECT_NAME, context) or ""
        ).strip()
        want = [
            spec.key
            for spec in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, spec.key, context)
        ]
        if not estimate_bed:
            skipped = [SPEC_BY_KEY[k].label for k in want if k in BED_KEYS]
            if skipped:
                feedback.pushInfo(
                    "Bed-estimate outputs skipped ('DEM surface is a water "
                    "surface' is not ticked): " + "; ".join(skipped)
                )
            want = [k for k in want if k not in BED_KEYS]
        if not want:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {SPEC_BY_KEY[key].kind for key in want}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(
                    f"Choose the {label[0].lower() + label[1:]}."
                )
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {
            key: output_path(SPEC_BY_KEY[key], folders, project_number) for key in want
        }

        depth_field = self.parameterAsString(parameters, self.DEPTH_FIELD, context)
        bed = self._bed_settings(parameters, context, crs, feedback, estimate_bed)

        areas, area_depths = self._read_areas(
            source, id_field, stage_field, crs, context, feedback, depth_field
        )
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
            pad = 1
            lines, lake = [], None
            if estimate_bed:
                lines = [pts for lg, pts in bed["lines"] if lg.intersects(geom)]
                cell = min(abs(gt[1]), abs(gt[5]))
                reach = bed["reach_m"] or max(20.0 * cell, 200.0)
                pad = 3 + (math.ceil(reach / cell) if lines else 0)
                lake = self._match_hydrolake(bed["lakes"], geom)
            window = polygon_window(area.polygons, gt, shape, pad=pad)
            if window is None:
                failures.append(f"Storage '{area.storage_id}': outside the DEM.")
                feedback.pushWarning(f"Skipped - {failures[-1]}")
                continue
            max_stage = field_stage if field_stage is not None else global_max
            try:
                dem = read_window(dem_layer.source(), window)
                storage_kwargs = dict(
                    max_stage=max_stage,
                    mode=mode,
                    supersample=supersample,
                    polygon_interval=poly_interval,
                    contour_interval=contour_interval,
                    contour_smoothing=smoothing,
                    is_canceled=feedback.isCanceled,
                )
                window_gt = window_geotransform(gt, window[0], window[1])
                if estimate_bed:
                    known = None
                    if depth_field and not _is_null(area_depths.get(id(area))):
                        known = float(area_depths[id(area)])
                    result = compute_storage_with_bed(
                        dem,
                        nodata,
                        window_gt,
                        window,
                        area,
                        interval,
                        centrelines=lines,
                        known_max_depth=known,
                        flat_tol=bed["flat_tol"],
                        water_level=bed["water_level"],
                        buffer_m=bed["buffer_m"],
                        reach_m=bed["reach_m"],
                        hydrolakes=lake,
                        **storage_kwargs,
                    )
                else:
                    result = compute_storage(
                        dem, nodata, window_gt, window, area, interval, **storage_kwargs
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
                f"{a['vol_m3']:.1f} m3 over {a['area_ha']:.4f} ha"
            )
            if result.bathymetry is not None:
                ba = result.bathymetry.attributes
                feedback.pushInfo(
                    f"  bed estimate ({ba['basis']}): water level "
                    f"{ba['water_lvl']:.3f} m, maximum depth {ba['dmax_lo']:.2f} / "
                    f"{ba['dmax_c']:.2f} / {ba['dmax_hi']:.2f} m, volume below the "
                    f"water {ba['vol_lo']:.0f} / {ba['vol_c']:.0f} / "
                    f"{ba['vol_hi']:.0f} m3 (low / central / high)"
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

        from mayim_tools.hydrology.catchment_delineation.qgis_outputs import (
            MemoryLayers,
            save_shapefile,
        )

        outputs = {name: folders[kind] for kind, (name, _) in FOLDER_PARAMS.items()
                   if kind in folders}  # fmt: skip
        written = []  # (label, path) for the report

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))
            feedback.pushInfo(f"Written: {path}")

        def failed(key, error):
            feedback.reportError(
                f"{SPEC_BY_KEY[key].label}: {Path(paths[key]).name} could not be "
                f"written ({error}). If it is open in QGIS, Excel or Word, close "
                "it and run again.",
                fatalError=False,
            )

        def guarded(key, writer):
            if key not in paths:
                return False
            try:
                writer(paths[key])
            except Exception as e:  # locked or not writable
                failed(key, e)
                return False
            done(key, paths[key])
            return True

        guarded(self.OUTPUT_TABLE, lambda p: write_stage_table_csv(p, results))
        guarded(self.OUTPUT_HEC, lambda p: write_hec_table(p, results))

        # vector layers: filled as memory layers, then written as shapefiles
        feedback.setProgressText("Writing layers")
        memory = MemoryLayers()
        self._sink_target = memory
        if self.OUTPUT_CONTOURS in want:
            self._write_contours(parameters, context, results, geoms, crs)
        if self.OUTPUT_WS in want:
            self._write_water_surfaces(
                parameters, context, feedback, results, geoms, crs
            )
        if self.OUTPUT_SPILL in want:
            self._write_spill_points(parameters, context, results, crs)
        if self.OUTPUT_FLOOR in want:
            self._write_floor_points(parameters, context, results, crs)
        bathys = [r.bathymetry for r in results if r.bathymetry is not None]
        if bathys and self.OUTPUT_SURVEY_WATER in want:
            self._write_survey_water(parameters, context, results, geoms, crs)
        for key, style, options in (
            (self.OUTPUT_CONTOURS, STYLE_CONTOURS, {}),
            (self.OUTPUT_WS, STYLE_WS, {"recategorise": True}),
            (self.OUTPUT_SPILL, STYLE_SPILL, {}),
            (self.OUTPUT_FLOOR, STYLE_FLOOR, {}),
            (self.OUTPUT_SURVEY_WATER, None, {}),
        ):
            layer = memory.get(key)
            if key not in want or layer is None:
                continue
            if layer.featureCount() == 0:
                feedback.pushWarning(f"{SPEC_BY_KEY[key].label}: nothing to write.")
                continue
            try:
                path = save_shapefile(layer, paths[key], context.transformContext())
            except Exception as e:
                failed(key, e)
                continue
            self._load(context, path, style, **options)
            done(key, path)

        # rasters
        if self.OUTPUT_DEPTH in want:
            merged = mosaic_depths(results, DEFAULT_DEPTH_NODATA)
            if merged is not None:
                array, (r0, c0) = merged
                valid = array[array != DEFAULT_DEPTH_NODATA]
                depth_range = (
                    (float(valid.min()), float(valid.max())) if valid.size else None
                )
                if guarded(
                    self.OUTPUT_DEPTH,
                    lambda p: write_raster(
                        p, array, window_geotransform(gt, r0, c0), projection,
                        DEFAULT_DEPTH_NODATA,
                    ),
                ):  # fmt: skip
                    self._load(
                        context, paths[self.OUTPUT_DEPTH], STYLE_DEPTH,
                        raster_range=depth_range,
                    )  # fmt: skip
        if bathys:
            for key, scenario in (
                (self.OUTPUT_BED_DEM, "central"),
                (self.OUTPUT_BED_DEM_LOW, "low"),
                (self.OUTPUT_BED_DEM_HIGH, "high"),
            ):
                if guarded(
                    key,
                    lambda p, s=scenario: write_bed_dem(
                        dem_layer.source(), p, bathys, s
                    ),
                ):
                    self._load(context, paths[key], None)
            guarded(
                self.OUTPUT_BED_TABLE,
                lambda p: write_bathymetry_summary_csv(p, bathys),
            )
            guarded(self.OUTPUT_PROFILE, lambda p: write_profile_csv(p, bathys))

        if self.OUTPUT_DOCX in want:
            docx_path = paths[self.OUTPUT_DOCX]
            feedback.setProgressText("Writing report")
            run_warnings = list(failures)
            for r in results:
                run_warnings.extend(r.warnings)
            context_info = {
                "tool": self.displayName(),
                "version": TOOL_VERSION,
                "software": _software_version(),
                "project_number": project_number,
                "project_name": project_name,
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
                "bed_estimate": (
                    {
                        "flat_tol": bed["flat_tol"],
                        "water_level": bed["water_level"],
                        "centrelines": bool(bed["lines"]),
                        "depth_field": depth_field or None,
                        "hydrolakes": bool(bed["lakes"]),
                    }
                    if estimate_bed
                    else None
                ),
                "coverage": COVERAGE_OPTIONS[
                    self.parameterAsEnum(parameters, self.COVERAGE, context)
                ],
                "outputs": list(written) + [("This report", docx_path)],
                "warnings": run_warnings,
            }
            try:
                for message in write_report_docx(docx_path, results, context_info):
                    feedback.pushInfo(message)
                done(self.OUTPUT_DOCX, docx_path)
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package, which is not "
                    "installed in QGIS's Python. Install it from the OSGeo4W Shell "
                    "with: python -m pip install python-docx. All other outputs "
                    "were written."
                )
            except Exception as e:  # report failure must not lose the results
                failed(self.OUTPUT_DOCX, e)
        if not written:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )
        feedback.setProgress(100)
        return outputs

    # ------------------------------------------------------------------

    def _write_water_surfaces(self, parameters, context, feedback, results, geoms, crs):
        sink, dest_id = self._sink_target.parameterAsSink(
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
        sink, dest_id = self._sink_target.parameterAsSink(
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

    # -- bed estimate ----------------------------------------------------

    def _transform(self, source, crs, context):
        if source is None or source.sourceCrs() == crs:
            return None
        return QgsCoordinateTransform(
            source.sourceCrs(), crs, context.transformContext()
        )

    def _bed_settings(self, parameters, context, crs, feedback, enabled):
        settings = {"lines": [], "lakes": []}
        raw_wl = parameters.get(self.WATER_LEVEL)
        settings["water_level"] = (
            None
            if _is_null(raw_wl)
            else self.parameterAsDouble(parameters, self.WATER_LEVEL, context)
        )
        settings["flat_tol"] = self.parameterAsDouble(
            parameters, self.FLAT_TOL, context
        )
        settings["buffer_m"] = (
            self.parameterAsDouble(parameters, self.SHORE_BUFFER, context) or None
        )
        settings["reach_m"] = (
            self.parameterAsDouble(parameters, self.REACH, context) or None
        )
        if not enabled:
            return settings
        lines_src = self.parameterAsSource(parameters, self.CENTRELINES, context)
        if lines_src is not None:
            transform = self._transform(lines_src, crs, context)
            for feature in lines_src.getFeatures():
                geom = QgsGeometry(feature.geometry())
                if geom.isNull() or geom.isEmpty():
                    continue
                if transform is not None:
                    geom.transform(transform)
                geom.convertToStraightSegment()
                parts = (
                    geom.asMultiPolyline()
                    if geom.isMultipart()
                    else [geom.asPolyline()]
                )
                for part in parts:
                    if len(part) >= 2:
                        settings["lines"].append(
                            (
                                QgsGeometry.fromPolylineXY(part),
                                [(pt.x(), pt.y()) for pt in part],
                            )
                        )
        lakes_src = self.parameterAsSource(parameters, self.HYDROLAKES, context)
        if lakes_src is not None:
            names = lakes_src.fields().names()
            missing = [n for n in ("Depth_avg", "Vol_total") if n not in names]
            if missing:
                feedback.pushWarning(
                    "HydroLAKES layer has no field(s) "
                    + ", ".join(missing)
                    + " - the HydroLAKES check is skipped."
                )
            else:
                transform = self._transform(lakes_src, crs, context)
                for feature in lakes_src.getFeatures():
                    geom = QgsGeometry(feature.geometry())
                    if geom.isNull() or geom.isEmpty():
                        continue
                    if transform is not None:
                        geom.transform(transform)
                    settings["lakes"].append(
                        (geom, feature["Depth_avg"], feature["Vol_total"])
                    )
        return settings

    @staticmethod
    def _match_hydrolake(lakes, geom):
        best = None
        for lake_geom, depth, vol in lakes:
            if lake_geom.intersects(geom):
                overlap = lake_geom.intersection(geom).area()
                if best is None or overlap > best[0]:
                    best = (overlap, depth, vol)
        if best is None:
            return None
        _, depth, vol = best
        return {
            "depth_avg_m": None if _is_null(depth) else float(depth),
            # HydroLAKES Vol_total is in million m3
            "volume_m3": None if _is_null(vol) else float(vol) * 1e6,
        }

    def _write_survey_water(self, parameters, context, results, geoms, crs):
        sink, dest_id = self._sink_target.parameterAsSink(
            parameters,
            self.OUTPUT_SURVEY_WATER,
            context,
            _fields(SURVEY_FIELDS),
            Qgis.WkbType.MultiPolygon,
            crs,
        )
        for r in results:
            b = r.bathymetry
            if b is None:
                continue
            polys = group_polygons(trace_rings(b.water_mask))
            polys_map = [
                [ring_to_map(simplify_ring(ring), b.window_gt) for ring in poly]
                for poly in polys
            ]
            geom = QgsGeometry.fromWkt(rings_to_multipolygon_wkt(polys_map))
            geom = geom.makeValid().intersection(geoms[id(r)])
            if QgsWkbTypes.flatType(geom.wkbType()) not in (
                Qgis.WkbType.Polygon,
                Qgis.WkbType.MultiPolygon,
            ):
                geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
            geom.convertToMultiType()
            feat = QgsFeature(_fields(SURVEY_FIELDS))
            feat.setGeometry(geom)
            for name, _, _ in SURVEY_FIELDS:
                value = b.attributes.get(name)
                feat[name] = float(value) if isinstance(value, float) else value
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _load(self, context, path, style_filename, **options):
        """Loads a written file into the project when the run ends, with its
        saved QML style (adapted to this run, see _StylePostProcessor)."""
        name = Path(path).stem
        details = QgsProcessingContext.LayerDetails(name, context.project(), name)
        if style_filename:
            processor = _StylePostProcessor(STYLES_DIR / style_filename, **options)
            details.setPostProcessor(processor)
            self._post_processors.append(processor)
        context.addLayerToLoadOnCompletion(path, details)

    def _write_floor_points(self, parameters, context, results, crs):
        sink, dest_id = self._sink_target.parameterAsSink(
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
        sink, dest_id = self._sink_target.parameterAsSink(
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
