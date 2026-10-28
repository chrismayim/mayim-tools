"""Processing Toolbox algorithm: Prepare HEC-HMS project (Data Tools).

Sub-catchments + stream reaches + stream nodes (Catchment Delineation /
Stream Network outputs) + an SCS curve number raster + DDF / hyetograph /
Huff / ensemble rainfall + an optional DEM -> a HEC-HMS 4.14 (or 4.13) project
folder that opens and computes in HEC-HMS, with an optional HEC-HMS
compute of every run and the peak flows read back. Thin wrapper - the
logic is in core.py, storms.py, hms_files.py and project.py
(tests/test_hec_hms_project_core.py).
"""

import os
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMemoryProviderUtils,
    QgsPointXY,
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
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    SHAPEFILE,
    WORD,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.rainfall._common import arf as arfm

from . import core
from . import methods as mth
from . import storms as stm
from .project import HMS_DIR_TEMPLATE, write_project

LAMBDA_OPTIONS = [
    (0.1, "0.1 (SCS-SA, Schulze et al. 1992)"),
    (0.2, "0.2 (NEH 630 / TR-55)"),
    (0.05, "0.05 (Hawkins et al. 2002)"),
]
CN_METHODS = [
    (core.CN_MEAN, "Area-weighted mean of the raster cells"),
    (core.CN_RUNOFF, "Runoff-equivalent CN at the CN design depth"),
]
TRANSFORMS = [
    (core.TRANSFORM_SCS, "SCS unit hydrograph (lag = 0.6 Tc)"),
    (core.TRANSFORM_CLARK, "Clark unit hydrograph (Tc, R)"),
    (core.TRANSFORM_SNYDER, "Snyder unit hydrograph (Tp, Cp)"),
]
ROUTINGS = [
    (core.ROUTE_MUSKINGUM, "Muskingum (K and X from the stream network)"),
    (core.ROUTE_LAG, "Lag (travel time)"),
    (core.ROUTE_MC, "Muskingum-Cunge (trapezoidal channel)"),
    (core.ROUTE_LAGK, "Lag and K (travel time split into lag and K)"),
    (core.ROUTE_KW, "Kinematic wave (trapezoidal channel)"),
]
LOSSES = [
    (mth.LOSS_SCS, "SCS curve number"),
    (mth.LOSS_IC, "Initial and Constant"),
    (mth.LOSS_DC, "Deficit and Constant"),
    (mth.LOSS_GA, "Green and Ampt (soil rasters)"),
]
ANTECEDENTS = [
    (mth.ANTECEDENT_MOIST, "Moist: initial water content at field capacity"),
    (mth.ANTECEDENT_DRY, "Dry: initial water content at wilting point"),
]
SNYDER_TPS = [
    (mth.SNYDER_TP_LAG, "Tp = 0.6 Tc (the SCS lag)"),
    (mth.SNYDER_TP_CT, "Tp = 0.75 Ct (L Lca)^0.3 (fields lfp_km, lca_km)"),
]
SOURCES = [
    (stm.SOURCE_DDF, "DDF table - HEC-HMS frequency storm"),
    (stm.SOURCE_AB, "Alternating Block design hyetographs (CSV)"),
    (stm.SOURCE_HUFF, "Huff curves (CSV) x DDF depth"),
    (stm.SOURCE_ENSEMBLE, "Design storm ensembles (CSV) x DDF depth"),
]
QUARTILES = [(None, "Automatic (Huff 1967: by duration)")] + [
    (q, f"Quartile {q}") for q in (1, 2, 3, 4)
]
DT_OPTIONS = [None, 1, 2, 3, 5, 10, 15, 30, 60]

OUTPUT_SPECS = [
    OutputSpec("OUTPUT_ELEMENTS", "HEC-HMS elements (points)", "HMS Elements", SHAPEFILE),
    OutputSpec("OUTPUT_LINKS", "HEC-HMS links (lines)", "HMS Links", SHAPEFILE, False),
    OutputSpec("OUTPUT_CSV", "Model set-up and peak flows (sectioned CSV)", "HMS Model Setup", CSV),
    OutputSpec("OUTPUT_DOCX", "Model set-up report (Word)", "HMS Model Report", WORD),
]  # fmt: skip
SPEC_BY_KEY = {s.key: s for s in OUTPUT_SPECS}
FOLDER_PARAMS = {
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


ARF_REGIONS = ["Not set"] + [f"Region {r}" for r in range(1, 6)]


def arf_parameters(alg):
    """(main, advanced) areal reduction parameters, shared with Prepare SWMM
    project."""
    from qgis.core import (
        QgsProcessingParameterEnum,
        QgsProcessingParameterFile,
        QgsProcessingParameterNumber,
    )

    main = [
        QgsProcessingParameterEnum(
            alg.ARF_METHOD,
            "Areal reduction factor (point DDF -> catchment rainfall)",
            options=[label for _, label in arfm.METHODS],
            defaultValue=0,
        )
    ]
    adv = [
        QgsProcessingParameterNumber(
            alg.ARF_AREA,
            "ARF area (km2; 0 = total area draining to the outlet)",
            QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            minValue=0.0,
        ),
        QgsProcessingParameterEnum(
            alg.ARF_REGION,
            "Pietersen et al. (2024) ARF region (South Africa)",
            options=ARF_REGIONS,
            defaultValue=0,
        ),
        QgsProcessingParameterFile(
            alg.ARF_TABLE,
            "User ARF table (CSV: DurationMin, ARF[, AreaKm2])",
            extension="csv",
            optional=True,
        ),
        QgsProcessingParameterNumber(
            alg.ARF,
            "Constant ARF (method 'Constant factor')",
            QgsProcessingParameterNumber.Double,
            defaultValue=1.0,
            minValue=0.1,
            maxValue=1.0,
        ),
    ]
    return main, adv


def _advanced(param):
    param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
    return param


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


def _folder_behavior():
    try:
        return Qgis.ProcessingFileParameterBehavior.Folder
    except AttributeError:  # older QGIS
        return QgsProcessingParameterFile.Behavior.Folder


def _srs_authority(crs) -> tuple[str, int]:
    """(authority, code) of the CRS for the basin geometry SQLite, e.g.
    ('EPSG', 22289) or ('ESRI', 102567); a CRS without one gets a
    user-defined code."""
    auth, _, code = crs.authid().partition(":")
    if auth.upper() in ("EPSG", "ESRI") and code.isdigit():
        return auth.upper(), int(code)
    return "EPSG", crs.postgisSrid() or 999999


def _wkt1(crs) -> str:
    try:
        return crs.toWkt(Qgis.CrsWktVariant.Wkt1Gdal)
    except (AttributeError, TypeError):
        return crs.toWkt()


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN -> None


def _numeric_attrs(feature, names) -> dict:
    """{field: value} of the numeric attributes (parameter overrides)."""
    out = {}
    for n in names:
        v = _num(feature[n])
        if v is not None:
            out[n] = v
    return out


def _text(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.upper() in ("NULL", "NONE") else s


class PrepareHecHmsProjectAlgorithm(QgsProcessingAlgorithm):
    SUBCATCHMENTS = "SUBCATCHMENTS"
    SUB_NAME_FIELD = "SUB_NAME_FIELD"
    SUB_NODE_FIELD = "SUB_NODE_FIELD"
    TC_FIELD = "TC_FIELD"
    STREAMS = "STREAMS"
    STREAM_NAME_FIELD = "STREAM_NAME_FIELD"
    STREAM_DS_FIELD = "STREAM_DS_FIELD"
    NODES = "NODES"
    NODE_NAME_FIELD = "NODE_NAME_FIELD"
    DEM = "DEM"
    CN_RASTER = "CN_RASTER"
    CN_FIELD = "CN_FIELD"
    CN_METHOD = "CN_METHOD"
    LOSS = "LOSS"
    SOIL_FOLDER = "SOIL_FOLDER"
    KSAT_FACTOR = "KSAT_FACTOR"
    ANTECEDENT = "ANTECEDENT"
    SOIL_DEPTH = "SOIL_DEPTH"
    IL_DEFAULT = "IL_DEFAULT"
    CL_DEFAULT = "CL_DEFAULT"
    SNYDER_TP = "SNYDER_TP"
    SNYDER_CT = "SNYDER_CT"
    SNYDER_CP = "SNYDER_CP"
    LAGK_SHARE = "LAGK_SHARE"
    CN_DEPTH = "CN_DEPTH"
    LAMBDA = "LAMBDA"
    IMPERV_FIELD = "IMPERV_FIELD"
    TRANSFORM = "TRANSFORM"
    TC_MIN = "TC_MIN"
    CLARK_RATIO = "CLARK_RATIO"
    ROUTING = "ROUTING"
    VELOCITY = "VELOCITY"
    MUSK_X = "MUSK_X"
    MC_N = "MC_N"
    MC_WIDTH = "MC_WIDTH"
    MC_SIDE = "MC_SIDE"
    RAIN_SOURCES = "RAIN_SOURCES"
    DDF_CSV = "DDF_CSV"
    DDF_SITE = "DDF_SITE"
    AB_CSV = "AB_CSV"
    HUFF_CSV = "HUFF_CSV"
    HUFF_QUARTILE = "HUFF_QUARTILE"
    HUFF_STAT = "HUFF_STAT"
    ENS_CSV = "ENS_CSV"
    ENS_SHAPES = "ENS_SHAPES"
    ARIS = "ARIS"
    DURATIONS = "DURATIONS"
    PEAK = "PEAK"
    ENSEMBLES = "ENSEMBLES"
    HYETO_STEP = "HYETO_STEP"
    ARF = "ARF"
    ARF_METHOD = "ARF_METHOD"
    ARF_AREA = "ARF_AREA"
    ARF_REGION = "ARF_REGION"
    ARF_TABLE = "ARF_TABLE"
    DT = "DT"
    TAIL_H = "TAIL_H"
    HMS_NAME = "HMS_NAME"
    BASIN_NAME = "BASIN_NAME"
    REACH_PREFIX = "REACH_PREFIX"
    JUNCTION_PREFIX = "JUNCTION_PREFIX"
    SINK_PREFIX = "SINK_PREFIX"
    RUN_HMS = "RUN_HMS"
    HMS_DIR = "HMS_DIR"
    HMS_VERSION = "HMS_VERSION"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"
    FOLDER_HMS = "FOLDER_HMS"
    OUTPUT_PROJECT = "OUTPUT_PROJECT"

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return PrepareHecHmsProjectAlgorithm()

    def name(self):
        return "prepare_hec_hms_project"

    def displayName(self):
        return "Prepare HEC-HMS project"

    def group(self):
        return "Data Tools"

    def groupId(self):
        return "data_tools"

    def shortHelpString(self):
        return (
            f"Prepare HEC-HMS project (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE\n"
            "Builds a complete HEC-HMS 4.14 (or 4.13) project from the GIS outputs of the "
            "catchment tools: a basin model with sub-basins, reaches, junctions and "
            "outlets (with the sub-basin and reach shapefiles as map layers), the "
            "terrain, the curve number raster as a Curve Number grid, one "
            "meteorologic model and simulation run per design storm, the control "
            "specification(s), the precipitation gages and their DSS time series, "
            "and one ensemble analysis per storm method and return period. The "
            "project opens complete and computes in HEC-HMS as it is; the tool can "
            "also compute every run and read back the peak flows.\n"
            "\n"
            "METHOD\n"
            "Geometry: each sub-catchment is a sub-basin (area from the polygon) "
            "draining to the junction at its stream node (field 'node' of the "
            "Catchment Delineation incremental output). Each stream is a reach "
            "from the node at its top (stem + next number: the top of A3 is A4) to "
            "the top node of the stream it flows into ('ds_name'), or to the outlet. "
            "A node with no reach leaving it is the outlet of its system (a sink). "
            "Without a node field, every sub-catchment drains to its own outlet. "
            "Names keep the GIS names; reaches, junctions and outlets get a prefix "
            "(R-, J-, Outlet-), as HEC-HMS names must be unique across element "
            "types. The DEM is clipped to the basin, warped to its CRS and written "
            "as the HEC-HMS terrain.\n"
            "Loss: SCS curve number from the CN raster (band 'cn_design' of SCS "
            "Curve Numbers, else band 1) - the area-weighted mean of the cells, or "
            "the runoff-equivalent CN at the CN design depth - or a CN field; "
            "Ia = lambda S with S = 25400/CN - 254 mm. Initial and Constant: "
            "initial loss from an il_mm field, else Ia from the CN, else the "
            "default; constant rate from a cl_mmh field, else the soil Ksat "
            "(geometric mean over the sub-basin, x the Ksat factor), else the "
            "default. Deficit and Constant: maximum deficit (theta_s - theta_wp) x "
            "the soil depth, initial deficit (theta_s - theta_i) x the depth with "
            "theta_i at field capacity (moist) or wilting point (dry) - else S and "
            "Ia from the CN, or md_mm / id_mm fields; constant rate as above. "
            "Green and Ampt: K = soil Ksat x factor, suction psi_f, theta_s and "
            "theta_i from the soil rasters, or the ga_ks_mmh, ga_psi_mm, ga_ts, "
            "ga_ti fields. Soil rasters: the Regional Soil Parameterisation "
            "outputs in the soil raster folder (0-30 cm P50 bands). Every "
            "estimated value's source is listed in the CSV and report.\n"
            "Transform: SCS unit hydrograph with lag = 0.6 Tc; Clark with Tc and "
            "R from R/(Tc+R); or Snyder (standard) with Tp = 0.6 Tc or 0.75 Ct "
            "(L Lca)^0.3 (Catchment Delineation lfp_km, lca_km) and Cp (or "
            "snyder_tp_h / snyder_cp fields). Tc from the Tc field (min), at least "
            "the minimum Tc.\n"
            "Routing: Muskingum with K (musk_k_h), X (musk_x) of the stream network "
            "and sub-reaches set for the time step (1/(2(1-X)) <= K/(n dt) <= "
            "1/(2X)); reaches too short for it use Lag. Lag uses the travel time; "
            "Muskingum-Cunge a trapezoid with the reach length and slope. Without "
            "travel times, K = length / velocity. Lag and K: lag = share x travel "
            "time, K = the rest (or lagk_lag_h / lagk_k_h fields). Kinematic wave: "
            "the Muskingum-Cunge trapezoid (or kw_n / kw_width_m fields) with "
            "sub-reaches of about one time step of travel.\n"
            "Rainfall (one run per return period, duration and peak position or "
            "pattern): DDF table -> HEC-HMS frequency storm (depths at 5 min ... the "
            "duration, nested with the peak at each chosen position); Alternating "
            "Block CSV -> its hyetographs for each chosen peak position; Huff CSV or "
            "ensemble CSV -> their temporal pattern x the DDF depth. Hyetographs are "
            "written as precipitation gages in a DSS file (needs the hecdss "
            "package: python -m pip install hecdss).\n"
            "Areal reduction: the DDF depths are point rainfall (also Design Rainfall SA, which is gridded but holds point values, and gridded rainfall adjusted to a point DDF) and are reduced to catchment rainfall: SANRAL Drainage Manual Eq 3.10 (default), Pietersen et al. (2024) regional ARFs for 24 h - 7 days (South Africa; Eq 3.10 below 24 h), the UK Flood Estimation Handbook ARF (Keers and Wescott 1977), a user ARF table, a constant, or none (DDF already areal, e.g. unadjusted gridded rainfall). Nested frequency storms and alternating block storms are reduced per duration (each nested / cumulative depth with the ARF of its own duration, so they follow the areal DDF curve); Huff and ensemble storms with the ARF of the storm duration. The ARF area is the total area draining to the outlet (the largest system if there are several) unless given. Alternating Block CSVs already reduced by that tool (ArfMethod column) are not reduced again. The CSV and report list the ARF per duration and a burst check: the largest 15 min ... depth inside each hyetograph as a % of the areal DDF depth (well below 100 %: the pattern is smoother than the DDF, typical of ERA5 patterns on a small catchment).\n"
            "Ensemble analyses: one per storm method and return period (e.g. "
            "'Frequency 1in50'), with every duration and peak position / pattern "
            "of it as members and every element's outflow as an analysis point. "
            "HEC-HMS ensemble members must share the time window, so the runs then "
            "use one control specification ('Design storms': the longest storm "
            "plus the run-out time).\n"
            "Names: HEC-HMS turns every character other than letters and digits in "
            "the project name into '_' and looks for its files under that name, so "
            "the project name and file names are written that way (the folder "
            "keeps its name).\n"
            "Curve number grid: the CN band, clipped and warped to the basin CRS, "
            "is the 'Curve Numbers' grid in Grid Data, ready for the Gridded SCS "
            "loss (with Modified Clark and a structured discretization, set in "
            "HEC-HMS). HEC-HMS map layers are vector only; the sub-basin map layer "
            "carries the CN, Ia, Tc and lag of each sub-basin.\n"
            "Time: the time step is the largest of 1-15 min not above 0.29 x the "
            "shortest lag (and the finest hyetograph step), unless set; the run "
            "lasts the storm plus 5 x lag + routing, rounded up to the hour.\n"
            "\n"
            "BACKGROUND\n"
            "HEC-HMS 4.14 User's and Technical Reference Manuals (USACE-HEC); NEH 630 "
            "Ch 10 and 16 (USDA-NRCS); SCS-SA (Schulze et al. 1992); Clark (1945); "
            "Cunge (1969); Huff (1967). The file layouts were checked by opening, "
            "computing and re-saving generated projects with HEC-HMS 4.13 and 4.14.\n"
            "\n"
            "PARAMETERS\n"
            "Sub-catchments: polygons in a projected CRS (metres), e.g. the "
            "Catchment Delineation catchments (incremental mode); name field "
            "outlet_id, node field node, Tc field tc_sel_mn (the selected Tc of "
            "the Time of Concentration tool).\n"
            "Stream reaches / nodes: the Catchment Delineation or Stream Network "
            "stream and node layers (fields name, ds_name, length_m, slope, "
            "musk_k_h, musk_x, tt_min).\n"
            "CN raster: SCS Curve Numbers output, or give a CN field.\n"
            "Peak positions: tick one or more (25, 33, 50, 67, 75 %; all by "
            "default) - each is a run of every frequency and Alternating Block "
            "storm.\n"
            "Return periods: e.g. '2, 5, 10, 20, 50, 100' (blank: all in the DDF "
            "table / hyetograph CSV). Durations: e.g. '1h, 3h, 6h, 12h, 24h' (DDF "
            "frequency storms need HEC-HMS durations: 5, 10, 15, 30 min, 1, 2, 3, 6, "
            "12, 24 h, 2, 3, 4, 7, 10 days).\n"
            "Areal reduction factor: method (advanced: ARF area in km2, 0 = automatic; Pietersen region 1 - 5; user table CSV with DurationMin, ARF and optionally AreaKm2; constant).\n"
            "HEC-HMS version: 4.14 (default) or 4.13; the two read the same file "
            "layouts and differ only in the version written into the files.\n"
            "Compute in HEC-HMS: runs HEC-HMS.cmd from the HEC-HMS folder (blank: "
            f"{HMS_DIR_TEMPLATE.format(version='<version>')}) and reads the peak "
            "flows into the CSV and report.\n"
            "Outputs: the project folder <project number>_<HEC-HMS project name> in "
            "the HEC-HMS folder, plus the ticked layers, CSV and report named "
            "<project number>_<description>.\n"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SUBCATCHMENTS,
                "Sub-catchments (polygons)",
                [QgsProcessing.TypeVectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.SUB_NAME_FIELD,
                "Sub-catchment name field",
                defaultValue="outlet_id",
                parentLayerParameterName=self.SUBCATCHMENTS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.SUB_NODE_FIELD,
                "Field with the stream node each sub-catchment drains to",
                defaultValue="node",
                parentLayerParameterName=self.SUBCATCHMENTS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TC_FIELD,
                "Time of concentration field (min)",
                defaultValue="tc_sel_mn",
                parentLayerParameterName=self.SUBCATCHMENTS,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STREAMS,
                "Stream reaches (lines; optional)",
                [QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.STREAM_NAME_FIELD,
                "Stream name field",
                defaultValue="name",
                parentLayerParameterName=self.STREAMS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.STREAM_DS_FIELD,
                "Downstream stream name field",
                defaultValue="ds_name",
                parentLayerParameterName=self.STREAMS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.NODES,
                "Stream nodes (points; optional)",
                [QgsProcessing.TypeVectorPoint],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.NODE_NAME_FIELD,
                "Node name field",
                defaultValue="name",
                parentLayerParameterName=self.NODES,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM, "DEM for the HEC-HMS terrain (optional)", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.CN_RASTER,
                "SCS curve number raster (curve_numbers.tif)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.CN_FIELD,
                "Curve number field (instead of the raster; optional)",
                parentLayerParameterName=self.SUBCATCHMENTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.CN_METHOD,
                "Sub-basin curve number from the raster",
                options=[label for _, label in CN_METHODS],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LOSS,
                "Loss method",
                options=[label for _, label in LOSSES],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.SOIL_FOLDER,
                "Soil raster folder (Regional Soil Parameterisation; loss methods "
                "other than SCS)",
                behavior=_folder_behavior(),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LAMBDA,
                "Initial abstraction ratio Ia / S",
                options=[label for _, label in LAMBDA_OPTIONS],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.TRANSFORM,
                "Transform",
                options=[label for _, label in TRANSFORMS],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ROUTING,
                "Reach routing",
                options=[label for _, label in ROUTINGS],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RAIN_SOURCES,
                "Design storms from",
                options=[label for _, label in SOURCES],
                allowMultiple=True,
                defaultValue=[0],
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.DDF_CSV,
                "DDF table (CSV; Design Rainfall or Precipitation data to DDF)",
                extension="csv",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.AB_CSV,
                "Alternating Block incremental depth CSV (optional)",
                extension="csv",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.HUFF_CSV,
                "Huff curves CSV (optional)",
                extension="csv",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.ENS_CSV,
                "Design storm ensemble CSV (optional)",
                extension="csv",
                optional=True,
            )
        )
        for p in arf_parameters(self)[0]:
            self.addParameter(p)
        self.addParameter(
            QgsProcessingParameterString(
                self.ARIS,
                "Return periods (years; blank = all in the table)",
                defaultValue="2, 5, 10, 20, 50, 100",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DURATIONS,
                "Storm durations (blank = all Alternating Block rows)",
                defaultValue="1h, 3h, 6h, 12h, 24h",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PEAK,
                "Peak positions (% of the duration; frequency and Alternating Block "
                "storms - one run per position)",
                options=[f"{p} %" for p in stm.PEAK_POSITIONS],
                allowMultiple=True,
                defaultValue=list(range(len(stm.PEAK_POSITIONS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENSEMBLES,
                "HEC-HMS ensemble analyses (one per storm method and return period)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.HMS_NAME,
                "HEC-HMS project name",
                defaultValue="HEC-HMS Model",
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.HMS_VERSION,
                "HEC-HMS version",
                options=core.HMS_VERSIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_HMS,
                "Compute the runs in HEC-HMS and read the peak flows",
                defaultValue=False,
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
        # ---- advanced
        adv = [
            QgsProcessingParameterNumber(
                self.CN_DEPTH,
                "CN design depth for the runoff-equivalent CN (mm)",
                QgsProcessingParameterNumber.Double,
                defaultValue=50.0,
                minValue=1.0,
            ),
            QgsProcessingParameterField(
                self.IMPERV_FIELD,
                "Impervious area field (% of the sub-catchment; optional)",
                parentLayerParameterName=self.SUBCATCHMENTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            ),
            QgsProcessingParameterNumber(
                self.TC_MIN,
                "Minimum time of concentration (min)",
                QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=0.5,
            ),
            QgsProcessingParameterEnum(
                self.ANTECEDENT,
                "Soil moisture at the start of the storm (Deficit and Constant, "
                "Green and Ampt)",
                options=[label for _, label in ANTECEDENTS],
                defaultValue=0,
            ),
            QgsProcessingParameterNumber(
                self.KSAT_FACTOR,
                "Factor on the soil Ksat (constant rate, Green and Ampt K)",
                QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.01,
                maxValue=10.0,
            ),
            QgsProcessingParameterNumber(
                self.SOIL_DEPTH,
                "Soil layer depth for the Deficit and Constant deficits (mm)",
                QgsProcessingParameterNumber.Double,
                defaultValue=300.0,
                minValue=10.0,
            ),
            QgsProcessingParameterNumber(
                self.IL_DEFAULT,
                "Initial loss / deficit without CN, soil or field (mm)",
                QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
                minValue=0.0,
            ),
            QgsProcessingParameterNumber(
                self.CL_DEFAULT,
                "Constant loss rate without soil Ksat or field (mm/h)",
                QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=0.0,
            ),
            QgsProcessingParameterEnum(
                self.SNYDER_TP,
                "Snyder standard lag Tp",
                options=[label for _, label in SNYDER_TPS],
                defaultValue=0,
            ),
            QgsProcessingParameterNumber(
                self.SNYDER_CT,
                "Snyder Ct (with Tp = 0.75 Ct (L Lca)^0.3)",
                QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.1,
                maxValue=10.0,
            ),
            QgsProcessingParameterNumber(
                self.SNYDER_CP,
                "Snyder peaking coefficient Cp",
                QgsProcessingParameterNumber.Double,
                defaultValue=0.6,
                minValue=0.05,
                maxValue=1.0,
            ),
            QgsProcessingParameterNumber(
                self.LAGK_SHARE,
                "Lag and K: share of the travel time taken as lag (rest is K)",
                QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.0,
                maxValue=1.0,
            ),
            QgsProcessingParameterNumber(
                self.CLARK_RATIO,
                "Clark R / (Tc + R)",
                QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.05,
                maxValue=0.95,
            ),
            QgsProcessingParameterNumber(
                self.VELOCITY,
                "Velocity / index celerity where no travel time is given (m/s)",
                QgsProcessingParameterNumber.Double,
                defaultValue=1.5,
                minValue=0.05,
            ),
            QgsProcessingParameterNumber(
                self.MUSK_X,
                "Muskingum X where the reach has none",
                QgsProcessingParameterNumber.Double,
                defaultValue=0.2,
                minValue=0.0,
                maxValue=0.5,
            ),
            QgsProcessingParameterNumber(
                self.MC_N,
                "Muskingum-Cunge Manning n",
                QgsProcessingParameterNumber.Double,
                defaultValue=0.04,
                minValue=0.005,
            ),
            QgsProcessingParameterNumber(
                self.MC_WIDTH,
                "Muskingum-Cunge bottom width (m)",
                QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=0.0,
            ),
            QgsProcessingParameterNumber(
                self.MC_SIDE,
                "Muskingum-Cunge side slope (H:V)",
                QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.0,
            ),
            QgsProcessingParameterString(
                self.DDF_SITE,
                "DDF site (multi-site Design Rainfall CSV; blank = first)",
                optional=True,
            ),
            QgsProcessingParameterEnum(
                self.HUFF_QUARTILE,
                "Huff quartile",
                options=[label for _, label in QUARTILES],
                defaultValue=0,
            ),
            QgsProcessingParameterEnum(
                self.HUFF_STAT,
                "Huff curve",
                options=stm.HUFF_STATS,
                defaultValue=2,
            ),
            QgsProcessingParameterEnum(
                self.ENS_SHAPES,
                "Ensemble storm shapes",
                options=stm.SHAPES,
                allowMultiple=True,
                defaultValue=[0, 1, 2],
            ),
            QgsProcessingParameterNumber(
                self.HYETO_STEP,
                "Huff hyetograph time step (min)",
                QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=1.0,
            ),
            QgsProcessingParameterEnum(
                self.DT,
                "Model time step",
                options=["Automatic"] + [f"{d} min" for d in DT_OPTIONS[1:]],
                defaultValue=0,
            ),
            QgsProcessingParameterNumber(
                self.TAIL_H,
                "Simulation time after the storm (h; 0 = automatic)",
                QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            ),
            QgsProcessingParameterString(
                self.BASIN_NAME, "Basin model name", defaultValue="Existing"
            ),
            QgsProcessingParameterString(
                self.REACH_PREFIX, "Reach name prefix", defaultValue="R-", optional=True
            ),
            QgsProcessingParameterString(
                self.JUNCTION_PREFIX,
                "Junction name prefix",
                defaultValue="J-",
                optional=True,
            ),
            QgsProcessingParameterString(
                self.SINK_PREFIX,
                "Outlet (sink) name prefix",
                defaultValue="Outlet-",
                optional=True,
            ),
            QgsProcessingParameterFile(
                self.HMS_DIR,
                "HEC-HMS program folder (blank = the standard folder of the version)",
                behavior=_folder_behavior(),
                optional=True,
            ),
        ]
        for p in arf_parameters(self)[1] + adv:
            self.addParameter(_advanced(p))
        for spec in OUTPUT_SPECS:
            self.addParameter(
                QgsProcessingParameterBoolean(
                    spec.key, f"Output: {spec.label}", defaultValue=spec.default
                )
            )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.FOLDER_HMS, "Output folder for the HEC-HMS project"
            )
        )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
            )

    # ------------------------------------------------------------------ inputs

    def _transform(self, source, crs, context):
        if source.sourceCrs() == crs:
            return None
        return QgsCoordinateTransform(
            source.sourceCrs(), crs, context.transformContext()
        )

    def _read_subcatchments(self, parameters, context, feedback, crs):
        source = self.parameterAsSource(parameters, self.SUBCATCHMENTS, context)
        names = source.fields().names()

        def field(key):
            f = self.parameterAsString(parameters, key, context)
            if f and f not in names:
                feedback.pushWarning(f"Field '{f}' is not in the sub-catchment layer.")
                return ""
            return f

        name_f = field(self.SUB_NAME_FIELD)
        node_f = field(self.SUB_NODE_FIELD)
        tc_f = field(self.TC_FIELD)
        cn_f = field(self.CN_FIELD)
        imp_f = field(self.IMPERV_FIELD)
        if not tc_f:
            raise QgsProcessingException("Choose the time of concentration field.")
        to_wgs = QgsCoordinateTransform(
            crs, QgsCoordinateReferenceSystem("EPSG:4326"), context.transformContext()
        )
        subs, rings, wkts = [], {}, []
        for f in source.getFeatures():
            g = QgsGeometry(f.geometry())
            if g.isNull() or g.isEmpty():
                continue
            if not g.isGeosValid():
                g = g.makeValid()
            nm = _text(f[name_f]) if name_f else ""
            nm = nm or f"SB{f.id()}"
            c = g.centroid()
            if not g.contains(c):
                c = g.pointOnSurface()
            cp = c.asPoint()
            try:
                ll = to_wgs.transform(cp)
                lat, lon = ll.y(), ll.x()
            except Exception:  # noqa: BLE001
                lat = lon = None
            node = _text(f[node_f]) if node_f else ""
            x_out = _num(f["x_out"]) if "x_out" in names else None
            y_out = _num(f["y_out"]) if "y_out" in names else None
            g.convertToMultiType()
            subs.append(
                core.SubIn(
                    name=nm,
                    area_km2=g.area() / 1e6,
                    cx=cp.x(),
                    cy=cp.y(),
                    node=node or None,
                    tc_min=_num(f[tc_f]),
                    cn=_num(f[cn_f]) if cn_f else None,
                    imperv_pct=(_num(f[imp_f]) or 0.0) if imp_f else 0.0,
                    lat=lat,
                    lon=lon,
                    x_out=x_out,
                    y_out=y_out,
                    wkb=bytes(g.asWkb()),
                    sub_type=_text(f["sub_type"]) if "sub_type" in names else "",
                    bank=_text(f["bank"]) if "bank" in names else "",
                    attrs=_numeric_attrs(f, names),
                )
            )
            rings[nm] = [
                [(p.x(), p.y()) for p in poly[0]] for poly in g.asMultiPolygon() if poly
            ]
            wkts.append(g)
        if not subs:
            raise QgsProcessingException("The sub-catchment layer has no polygons.")
        if node_f and not any(x.node for x in subs):
            feedback.pushWarning(
                f"The node field '{node_f}' is empty: every sub-catchment drains to its "
                "own outlet."
            )
        return source, subs, rings, wkts, cn_f

    def _read_streams(self, parameters, context, feedback, crs):
        source = self.parameterAsSource(parameters, self.STREAMS, context)
        if source is None:
            return None, [], {}
        names = source.fields().names()
        nf = (
            self.parameterAsString(parameters, self.STREAM_NAME_FIELD, context)
            or "name"
        )
        df = (
            self.parameterAsString(parameters, self.STREAM_DS_FIELD, context)
            or "ds_name"
        )
        if nf not in names:
            raise QgsProcessingException(f"The stream layer has no field '{nf}'.")
        tr = self._transform(source, crs, context)
        reaches, lines = [], {}
        for f in source.getFeatures():
            nm = _text(f[nf])
            if not nm:
                continue
            g = QgsGeometry(f.geometry())
            if g.isNull() or g.isEmpty():
                continue
            if tr is not None:
                g.transform(tr)
            parts = g.asMultiPolyline() if g.isMultipart() else [g.asPolyline()]
            parts = [[(p.x(), p.y()) for p in part] for part in parts if part]
            xy = max(parts, key=len) if parts else []

            def val(key, feat=f):
                return _num(feat[key]) if key in names else None

            reaches.append(
                core.ReachIn(
                    name=nm,
                    ds_name=(_text(f[df]) if df in names else "") or None,
                    xy=xy,
                    length_m=val("length_m") or g.length(),
                    slope=val("slope"),
                    musk_k_h=val("musk_k_h"),
                    musk_x=val("musk_x"),
                    tt_min=val("tt_min"),
                    wkb=bytes(g.asWkb()),
                    attrs=_numeric_attrs(f, names),
                )
            )
            lines[nm] = parts
        if not reaches:
            feedback.pushWarning("The stream layer has no named reaches.")
        return source, reaches, lines

    def _read_nodes(self, parameters, context, crs):
        source = self.parameterAsSource(parameters, self.NODES, context)
        if source is None:
            return None, []
        nf = self.parameterAsString(parameters, self.NODE_NAME_FIELD, context) or "name"
        if nf not in source.fields().names():
            raise QgsProcessingException(f"The node layer has no field '{nf}'.")
        tr = self._transform(source, crs, context)
        out = []
        for f in source.getFeatures():
            nm = _text(f[nf])
            g = QgsGeometry(f.geometry())
            if not nm or g.isNull() or g.isEmpty():
                continue
            if tr is not None:
                g.transform(tr)
            p = g.centroid().asPoint()
            out.append(core.NodeIn(nm, p.x(), p.y()))
        return source, out

    def _arf_model(self, parameters, context, feedback, subs, reaches, nodes):
        """The ARF model: method, area (default: the largest outlet system)."""
        method = arfm.METHODS[
            self.parameterAsEnum(parameters, self.ARF_METHOD, context)
        ][0]
        area = self.parameterAsDouble(parameters, self.ARF_AREA, context)
        note = "given"
        if not area:
            try:
                areas = core.system_areas(subs, reaches, nodes, core.Settings())
            except core.HmsProjectError:
                areas = {"all": sum(x.area_km2 for x in subs)}
            area = next(iter(areas.values()))
            note = "total area draining to the outlet"
            if len(areas) > 1:
                note = f"largest of {len(areas)} outlet systems ({next(iter(areas))})"
                feedback.pushWarning(
                    f"ARF: {len(areas)} outlet systems; the largest "
                    f"({area:.4g} km2) is used for all. Give the ARF area to "
                    "change it."
                )
        region_i = self.parameterAsEnum(parameters, self.ARF_REGION, context)
        table = None
        if method == arfm.METHOD_TABLE:
            path = self.parameterAsFile(parameters, self.ARF_TABLE, context)
            if not path:
                raise QgsProcessingException("Give the user ARF table (CSV).")
            try:
                table = arfm.ArfTable.from_csv(path)
            except (arfm.ArfError, OSError) as e:
                raise QgsProcessingException(f"ARF table: {e}") from e
        try:
            model = arfm.ArfModel(
                method=method,
                area_km2=area,
                constant=self.parameterAsDouble(parameters, self.ARF, context),
                region=region_i or None,
                table=table,
                area_note=note,
            )
        except arfm.ArfError as e:
            raise QgsProcessingException(str(e)) from e
        feedback.pushInfo(f"Areal reduction: {model.label()} ({note})")
        return model

    def _method_settings(self, parameters, context, loss):
        return mth.MethodSettings(
            loss=loss,
            ksat_factor=self.parameterAsDouble(parameters, self.KSAT_FACTOR, context),
            antecedent=ANTECEDENTS[
                self.parameterAsEnum(parameters, self.ANTECEDENT, context)
            ][0],
            soil_depth_mm=self.parameterAsDouble(parameters, self.SOIL_DEPTH, context),
            initial_loss_mm=self.parameterAsDouble(
                parameters, self.IL_DEFAULT, context
            ),
            constant_rate_mmh=self.parameterAsDouble(
                parameters, self.CL_DEFAULT, context
            ),
            snyder_tp=SNYDER_TPS[
                self.parameterAsEnum(parameters, self.SNYDER_TP, context)
            ][0],
            snyder_ct=self.parameterAsDouble(parameters, self.SNYDER_CT, context),
            snyder_cp=self.parameterAsDouble(parameters, self.SNYDER_CP, context),
            lagk_lag_share=self.parameterAsDouble(parameters, self.LAGK_SHARE, context),
        )

    def _soil_values(self, parameters, context, feedback, crs, geoms, subs, loss):
        """Zonal soil parameters per sub-basin from the Regional Soil
        Parameterisation rasters (x.soil); returns the input label."""
        folder = self.parameterAsFile(parameters, self.SOIL_FOLDER, context)
        if not folder:
            if loss == mth.LOSS_GA:
                feedback.pushWarning(
                    "Green and Ampt without a soil raster folder: every sub-catchment "
                    "needs the ga_ks_mmh, ga_psi_mm, ga_ts and ga_ti fields."
                )
            return "-"
        if loss == mth.LOSS_SCS:
            feedback.pushInfo("Soil raster folder not used by the SCS loss method.")
            return f"{folder} (not used)"
        try:
            found, notes = mth.find_soil_rasters(folder)
        except mth.MethodError as e:
            raise QgsProcessingException(str(e)) from e
        for n in notes:
            feedback.pushInfo(n)
        if not found:
            raise QgsProcessingException(
                f"No Regional Soil Parameterisation rasters (bands such as "
                f"ksat_0-30cm_P50) in {folder}."
            )
        feedback.setProgressText("Soil parameters per sub-basin")
        from qgis.core import QgsRasterLayer

        for code, (path, band) in found.items():
            lyr = QgsRasterLayer(path, code)
            tr = QgsCoordinateTransform(crs, lyr.crs(), context.transformContext())
            wkts = []
            for g in geoms:
                gg = QgsGeometry(g)
                if lyr.crs() != crs:
                    gg.transform(tr)
                wkts.append(gg.asWkt())
            vals = mth.zonal_values(path, band, wkts, geometric=(code == "ksat"))
            for x, v in zip(subs, vals, strict=True):
                if x.soil is None:
                    x.soil = {}
                x.soil[code] = v
            empty = [x.name for x, v in zip(subs, vals, strict=True) if v is None]
            if empty:
                feedback.pushWarning(
                    f"No {code} value for {len(empty)} sub-catchment(s) "
                    f"({', '.join(empty[:8])}): outside the soil raster."
                )
        missing = [c for c in mth.SOIL_CODES if c not in found]
        if missing:
            feedback.pushInfo("Soil rasters not found: " + ", ".join(missing))
        return f"{folder} ({', '.join(sorted(found))})"

    def _storms(self, parameters, context, feedback, arf_model=None):
        chosen = [
            SOURCES[i][0]
            for i in self.parameterAsEnums(parameters, self.RAIN_SOURCES, context)
        ]
        if not chosen:
            raise QgsProcessingException("Choose at least one design storm source.")
        aris = stm.parse_number_list(
            self.parameterAsString(parameters, self.ARIS, context)
        )
        durs = stm.parse_duration_list(
            self.parameterAsString(parameters, self.DURATIONS, context)
        )
        peaks = [
            stm.PEAK_POSITIONS[i]
            for i in sorted(set(self.parameterAsEnums(parameters, self.PEAK, context)))
        ] or [50]
        arf = arf_model if arf_model is not None else 1.0
        ddf_path = self.parameterAsFile(parameters, self.DDF_CSV, context)
        curves = None
        if any(
            c in chosen for c in (stm.SOURCE_DDF, stm.SOURCE_HUFF, stm.SOURCE_ENSEMBLE)
        ):
            if not ddf_path:
                raise QgsProcessingException(
                    "Give the DDF table: the frequency, Huff and ensemble storms need it."
                )
            curves = stm.DdfCurves.from_csv(
                ddf_path, self.parameterAsString(parameters, self.DDF_SITE, context)
            )
            if not aris:
                aris = curves.aris
            if curves.info.get("fixed_day_durations"):
                feedback.pushInfo(
                    "DDF: fixed-day durations converted to continuous: "
                    + ", ".join(curves.info["fixed_day_durations"])
                )
        storms, warnings = [], []
        if stm.SOURCE_DDF in chosen:
            for t in aris:
                for d in durs or [60, 180, 360, 720, 1440]:
                    for peak in peaks:
                        storms.append(stm.frequency_storm(curves, t, d, peak, arf))
        if stm.SOURCE_AB in chosen:
            p = self.parameterAsFile(parameters, self.AB_CSV, context)
            if not p:
                raise QgsProcessingException("Give the Alternating Block CSV.")
            rows = stm.read_alternating_block_csv(p)
            s, w = stm.alternating_block_storms(rows, aris, durs, peaks, arf)
            storms += s
            warnings += w
        if stm.SOURCE_HUFF in chosen:
            p = self.parameterAsFile(parameters, self.HUFF_CSV, context)
            if not p:
                raise QgsProcessingException("Give the Huff curves CSV.")
            q = QUARTILES[
                self.parameterAsEnum(parameters, self.HUFF_QUARTILE, context)
            ][0]
            stat = stm.HUFF_STATS[
                self.parameterAsEnum(parameters, self.HUFF_STAT, context)
            ]
            s, w = stm.huff_storms(
                curves,
                stm.read_huff_csv(p),
                aris,
                durs or [60, 180, 360, 720, 1440],
                self.parameterAsDouble(parameters, self.HYETO_STEP, context),
                q,
                stat,
                arf,
            )
            storms += s
            warnings += w
        if stm.SOURCE_ENSEMBLE in chosen:
            p = self.parameterAsFile(parameters, self.ENS_CSV, context)
            if not p:
                raise QgsProcessingException("Give the design storm ensemble CSV.")
            shapes = [
                stm.SHAPES[i]
                for i in self.parameterAsEnums(parameters, self.ENS_SHAPES, context)
            ] or stm.SHAPES
            s, w = stm.ensemble_storms(
                curves,
                stm.read_ensemble_csv(p),
                aris,
                durs or [60, 180, 360, 720, 1440],
                shapes,
                arf,
            )
            storms += s
            warnings += w
        stm.add_burst_checks(storms, curves, arf)
        if arf_model is not None:
            warnings += arf_model.notes()
        flagged = sorted({f for st in storms for f in st.flags})
        if "extrap_short" in flagged:
            warnings.append(
                "DDF depths below the table's shortest duration were extrapolated with "
                "its first segment's power law (may underestimate short bursts)."
            )
        if "extrap_T" in flagged:
            warnings.append(
                "Some return periods lie outside the DDF table (extrapolated)."
            )
        if "extrap_long" in flagged:
            warnings.append(
                "Some durations are longer than the DDF table (extrapolated)."
            )
        return storms, warnings, chosen

    # ------------------------------------------------------------------ run

    def processAlgorithm(self, parameters, context, feedback):
        self._post_processors = []
        source = self.parameterAsSource(parameters, self.SUBCATCHMENTS, context)
        if source is None:
            raise QgsProcessingException("Choose the sub-catchment layer.")
        crs = source.sourceCrs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The sub-catchments are in a geographic CRS (degrees). Reproject them "
                "to a projected CRS in metres first."
            )
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        )
        project_name = self.parameterAsString(
            parameters, self.PROJECT_NAME, context
        ).strip()
        hms_name = self.parameterAsString(parameters, self.HMS_NAME, context).strip()
        folder_name = "_".join(
            p for p in (clean_name_part(project_number), clean_name_part(hms_name)) if p
        )
        if not folder_name:
            raise QgsProcessingException("Give the HEC-HMS project name.")
        hms_parent = self.parameterAsFileOutput(parameters, self.FOLDER_HMS, context)
        if not hms_parent:
            raise QgsProcessingException(
                "Choose the output folder for the HEC-HMS project."
            )
        selected = [
            s
            for s in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, s.key, context)
        ]
        folders = {}
        for kind in {s.kind for s in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {s.key: s.path(folders, project_number) for s in selected}

        feedback.setProgressText("Reading the sub-catchments")
        _, subs, rings, geoms, cn_field = self._read_subcatchments(
            parameters, context, feedback, crs
        )
        cn_layer = self.parameterAsRasterLayer(parameters, self.CN_RASTER, context)
        loss = LOSSES[self.parameterAsEnum(parameters, self.LOSS, context)][0]
        lam = LAMBDA_OPTIONS[self.parameterAsEnum(parameters, self.LAMBDA, context)][0]
        cn_method = CN_METHODS[
            self.parameterAsEnum(parameters, self.CN_METHOD, context)
        ][0]
        cn_depth = self.parameterAsDouble(parameters, self.CN_DEPTH, context)
        cn_label = f"field {cn_field}" if cn_field else "-"
        cn_band = 1
        if cn_layer is not None and not cn_field:
            feedback.setProgressText("Curve numbers per sub-basin")
            tr = QgsCoordinateTransform(crs, cn_layer.crs(), context.transformContext())
            wkts = []
            for g in geoms:
                gg = QgsGeometry(g)
                if cn_layer.crs() != crs:
                    gg.transform(tr)
                wkts.append(gg.asWkt())
            try:
                stats, band = core.zonal_cn_from_raster(
                    cn_layer.source(), wkts, cn_method, cn_depth, lam
                )
            except core.HmsProjectError as e:
                raise QgsProcessingException(str(e)) from e
            cn_label = f"{cn_layer.source()} ({band})"
            cn_band = int(band.split()[1])
            for x, st in zip(subs, stats, strict=True):
                if st.get("n_cells"):
                    x.cn = st["cn"]
                    x.cn_stats = st
                    if st.get("centroid_only"):
                        feedback.pushWarning(
                            f"{x.name}: CN from the cell at its centroid."
                        )
        elif not cn_field and loss == mth.LOSS_SCS:
            raise QgsProcessingException("Give the curve number raster or a CN field.")
        missing = [x.name for x in subs if x.cn is None]
        if missing and loss == mth.LOSS_SCS:
            raise QgsProcessingException(
                f"No curve number for {len(missing)} sub-catchment(s): "
                + ", ".join(missing[:10])
            )
        soil_label = self._soil_values(
            parameters, context, feedback, crs, geoms, subs, loss
        )

        feedback.setProgressText("Reading the streams and nodes")
        ssrc, reaches, lines = self._read_streams(parameters, context, feedback, crs)
        nsrc, nodes = self._read_nodes(parameters, context, crs)

        arf_model = self._arf_model(parameters, context, feedback, subs, reaches, nodes)
        try:
            storms, storm_warnings, chosen = self._storms(
                parameters, context, feedback, arf_model
            )
        except (stm.StormError, ValueError) as e:
            raise QgsProcessingException(str(e)) from e

        dt_opt = DT_OPTIONS[self.parameterAsEnum(parameters, self.DT, context)]
        tail = self.parameterAsDouble(parameters, self.TAIL_H, context)
        hms_version = core.HMS_VERSIONS[
            self.parameterAsEnum(parameters, self.HMS_VERSION, context)
        ]
        hms_dir = self.parameterAsFile(
            parameters, self.HMS_DIR, context
        ) or HMS_DIR_TEMPLATE.format(version=hms_version)
        s = core.Settings(
            project_name=folder_name,
            hms_version=hms_version,
            basin_name=self.parameterAsString(
                parameters, self.BASIN_NAME, context
            ).strip()
            or "Existing",
            description=(project_name or "")
            + (" - " if project_name else "")
            + "prepared with Mayim Tools",
            reach_prefix=self.parameterAsString(parameters, self.REACH_PREFIX, context),
            junction_prefix=self.parameterAsString(
                parameters, self.JUNCTION_PREFIX, context
            ),
            sink_prefix=self.parameterAsString(parameters, self.SINK_PREFIX, context),
            lambda_ia=lam,
            cn_method=cn_method,
            cn_design_depth_mm=cn_depth,
            transform=TRANSFORMS[
                self.parameterAsEnum(parameters, self.TRANSFORM, context)
            ][0],
            tc_floor_min=self.parameterAsDouble(parameters, self.TC_MIN, context),
            clark_ratio=self.parameterAsDouble(parameters, self.CLARK_RATIO, context),
            routing=ROUTINGS[self.parameterAsEnum(parameters, self.ROUTING, context)][
                0
            ],
            velocity_ms=self.parameterAsDouble(parameters, self.VELOCITY, context),
            musk_x=self.parameterAsDouble(parameters, self.MUSK_X, context),
            mc_n=self.parameterAsDouble(parameters, self.MC_N, context),
            mc_width_m=self.parameterAsDouble(parameters, self.MC_WIDTH, context),
            mc_side_slope=self.parameterAsDouble(parameters, self.MC_SIDE, context),
            dt_min=dt_opt,
            tail_h=tail if tail > 0 else None,
            ensembles=self.parameterAsBoolean(parameters, self.ENSEMBLES, context),
            methods=self._method_settings(parameters, context, loss),
        )
        try:
            r = core.assemble(subs, reaches, nodes, storms, s)
        except (core.HmsProjectError, stm.StormError) as e:
            raise QgsProcessingException(str(e)) from e
        r.arf = arf_model
        r.warnings = storm_warnings + r.warnings
        r.crs_name = f"{crs.authid()} - {crs.description()}"
        for e in r.network.of("Subbasin"):
            x = next(v for v in subs if v.name == e.source)
            e.params["lat"], e.params["lon"] = x.lat, x.lon

        feedback.setProgressText("Writing the HEC-HMS project")
        dem = self.parameterAsRasterLayer(parameters, self.DEM, context)
        ext = QgsGeometry.unaryUnion(geoms).boundingBox()
        srs_auth, srid = _srs_authority(crs)
        try:
            write_project(
                r,
                hms_parent,
                folder_name,
                crs_wkt=_wkt1(crs),
                srid=srid,
                srtext=_wkt1(crs),
                srs_auth=srs_auth,
                cn_raster=(
                    cn_layer.source() if cn_layer is not None and not cn_field else None
                ),
                cn_band=cn_band,
                dem_path=dem.source() if dem is not None else None,
                terrain_name="Terrain",
                bounds=(ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum()),
                hms_dir=hms_dir,
                log=feedback.pushWarning,
            )
        except Exception as e:
            from .dss import DssUnavailableError

            if isinstance(e, DssUnavailableError):
                raise QgsProcessingException(str(e)) from e
            raise QgsProcessingException(
                f"The HEC-HMS project could not be written: {e}"
            ) from e
        feedback.pushInfo(f"HEC-HMS project: {r.files['Project']}")

        results = None
        if self.parameterAsBoolean(parameters, self.RUN_HMS, context):
            from . import results as rs

            if rs.find_hms(hms_dir) is None:
                feedback.pushWarning(
                    f"HEC-HMS.cmd was not found in {hms_dir}; the runs were not computed."
                )
            else:
                feedback.setProgressText(f"Computing {len(r.runs)} run(s) in HEC-HMS")
                try:
                    code, out = rs.run_hms(hms_dir, r.folder)
                    for line in out.splitlines():
                        if "runs computed" in line or "failed" in line:
                            feedback.pushInfo(line.strip())
                except Exception as e:  # noqa: BLE001
                    feedback.pushWarning(f"HEC-HMS could not be run: {e}")
                results, missing = rs.read_results(
                    r.folder, [run.name for run in r.runs]
                )
                for run in r.runs:
                    for err in rs.run_log_errors(r.folder, run.dss_file):
                        r.warnings.append(f"HEC-HMS {run.name}: {err}")
                if missing:
                    r.warnings.append(
                        f"{len(missing)} run(s) have no HEC-HMS results: "
                        + ", ".join(missing[:10])
                    )

        outputs = {self.OUTPUT_PROJECT: r.files["Project"], self.FOLDER_HMS: r.folder}
        written = [(k, v) for k, v in r.files.items()]

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        inputs = [
            ("Sub-catchments", f"{source.sourceName()} ({len(subs)} polygons)"),
            ("CRS", r.crs_name),
            ("Streams", ssrc.sourceName() if ssrc is not None else "-"),
            ("Nodes", nsrc.sourceName() if nsrc is not None else "-"),
            ("Curve numbers", cn_label),
            ("Loss method", mth.LOSS_LABELS[loss]),
            ("Soil rasters", soil_label),
            (
                "Time of concentration",
                self.parameterAsString(parameters, self.TC_FIELD, context),
            ),
            ("DEM (terrain)", dem.source() if dem is not None else "-"),
            (
                "DDF table",
                self.parameterAsFile(parameters, self.DDF_CSV, context) or "-",
            ),
            ("Storm sources", ", ".join(stm.SOURCE_LABELS[c] for c in chosen)),
        ]
        if "OUTPUT_ELEMENTS" in paths or "OUTPUT_LINKS" in paths:
            from mayim_tools.hydrology.catchment_delineation.qgis_outputs import (
                load_on_completion,
                save_shapefile,
            )

            for key, layer in (
                ("OUTPUT_ELEMENTS", self._elements_layer(r, crs)),
                ("OUTPUT_LINKS", self._links_layer(r, crs)),
            ):
                if key not in paths:
                    continue
                try:
                    p = save_shapefile(layer, paths[key], context.transformContext())
                    load_on_completion(
                        context, p, Path(p).stem, None, self._post_processors
                    )
                    done(key, p)
                except Exception as e:  # noqa: BLE001
                    feedback.reportError(str(e), fatalError=False)
        if "OUTPUT_CSV" in paths:
            from .export import write_csv

            try:
                write_csv(paths["OUTPUT_CSV"], r, inputs, results)
                done("OUTPUT_CSV", paths["OUTPUT_CSV"])
            except OSError as e:
                feedback.reportError(
                    f"{Path(paths['OUTPUT_CSV']).name} could not be written ({e}); it "
                    "may be open in another program.",
                    fatalError=False,
                )
        if "OUTPUT_DOCX" in paths:
            feedback.setProgressText("Writing the report")
            geometry = {}
            for e in r.network.of("Subbasin"):
                geometry[("Subbasin", e.name)] = rings.get(e.source, [])
            for e in r.network.of("Reach"):
                geometry[("Reach", e.name)] = lines.get(e.source, [])
            try:
                from .report import write_report

                notes = write_report(
                    paths["OUTPUT_DOCX"],
                    r,
                    {
                        "project_number": project_number,
                        "project_name": project_name,
                        "software": _software_version(),
                        "inputs": inputs,
                        "outputs": written + [("This report", paths["OUTPUT_DOCX"])],
                        "warnings": r.warnings,
                    },
                    geometry,
                    results,
                )
                for msg in notes:
                    feedback.pushWarning(msg)
                done("OUTPUT_DOCX", paths["OUTPUT_DOCX"])
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs python-docx and matplotlib in QGIS's Python "
                    "(python -m pip install python-docx). All other outputs were written."
                )
            except Exception as e:  # noqa: BLE001 - the results must not be lost
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for w in r.warnings:
            feedback.pushWarning(w)
        feedback.pushInfo(
            f"{len(r.network.of('Subbasin'))} sub-basins, {len(r.network.of('Reach'))} "
            f"reaches, {len(r.network.of('Junction'))} junctions, "
            f"{len(r.network.of('Sink'))} outlet(s); time step {r.dt_min:g} min; "
            f"{len(r.runs)} run(s)."
        )
        if results:
            from .export import peak_table

            hdr, rows = peak_table(r, results)
            for row in rows:
                if row[0].startswith(s.sink_prefix):
                    vals = [v for v in row[1:] if v is not None]
                    if vals:
                        feedback.pushInfo(
                            f"{row[0]}: largest peak {max(vals):.3f} m3/s"
                        )
        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        return outputs

    # ------------------------------------------------------------------ layers

    def _elements_layer(self, r, crs):
        fields = QgsFields()
        for name, t in (
            ("name", QMetaType.Type.QString),
            ("kind", QMetaType.Type.QString),
            ("source", QMetaType.Type.QString),
            ("ds", QMetaType.Type.QString),
            ("area_km2", QMetaType.Type.Double),
            ("cn", QMetaType.Type.Double),
            ("ia_mm", QMetaType.Type.Double),
            ("tc_min", QMetaType.Type.Double),
            ("lag_min", QMetaType.Type.Double),
            ("route", QMetaType.Type.QString),
            ("k_h", QMetaType.Type.Double),
            ("steps", QMetaType.Type.Int),
        ):
            field = QgsField(name, t)
            if t == QMetaType.Type.QString:
                field.setLength(80)
            fields.append(field)
        layer = QgsMemoryProviderUtils.createMemoryLayer(
            "HMS elements", fields, Qgis.WkbType.Point, crs
        )
        feats = []
        for e in r.network.elements:
            f = QgsFeature(fields)
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(e.x, e.y)))
            p = e.params
            f["name"], f["kind"], f["source"], f["ds"] = (
                e.name,
                e.kind,
                e.source,
                e.downstream or "",
            )
            for k in ("area_km2", "cn", "ia_mm", "tc_min", "lag_min", "k_h"):
                f[k] = p.get(k)
            f["route"] = p.get("method") or ""
            f["steps"] = p.get("steps")
            feats.append(f)
        layer.dataProvider().addFeatures(feats)
        layer.updateExtents()
        return layer

    def _links_layer(self, r, crs):
        fields = QgsFields()
        for name in ("from_el", "to_el", "kind"):
            field = QgsField(name, QMetaType.Type.QString)
            field.setLength(80)
            fields.append(field)
        layer = QgsMemoryProviderUtils.createMemoryLayer(
            "HMS links", fields, Qgis.WkbType.LineString, crs
        )
        by = r.network.by_name()
        feats = []
        for e in r.network.elements:
            if not e.downstream or e.kind == "Junction":
                continue  # a junction's link is the reach that starts there
            d = by[e.downstream]
            start = e.from_xy if (e.kind == "Reach" and e.from_xy) else (e.x, e.y)
            end = (d.x, d.y) if e.kind != "Reach" else (e.x, e.y)
            if start == end:
                continue
            f = QgsFeature(fields)
            f.setGeometry(
                QgsGeometry.fromPolylineXY([QgsPointXY(*start), QgsPointXY(*end)])
            )
            f["from_el"], f["to_el"], f["kind"] = e.name, e.downstream, e.kind
            feats.append(f)
        layer.dataProvider().addFeatures(feats)
        layer.updateExtents()
        return layer
