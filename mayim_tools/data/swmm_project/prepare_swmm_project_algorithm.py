"""Processing Toolbox algorithm: Prepare SWMM project (Data Tools).

Sub-catchments + stream reaches + stream nodes (Catchment Delineation /
Stream Network outputs) + an SCS curve number raster + DDF / hyetograph /
Huff / ensemble rainfall + a DEM -> one EPA SWMM 5 input file with every
design storm as a rainfall time series, for EPA SWMM, PCSWMM and Autodesk
Storm and Sanitary Analysis (import); optionally every storm computed in the
EPA SWMM engine (swmm-toolkit). Thin wrapper: the network, curve numbers and
storms come from the Prepare HEC-HMS project code, the SWMM model from
core.py / inp.py / run.py (tests/test_swmm_project_core.py).
"""

import os

from qgis.core import (
    QgsCoordinateTransform,
    QgsProcessing,
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

from mayim_tools._common.output_files import CSV, WORD, OutputSpec, clean_name_part

from ..hec_hms_project import core as hms
from ..hec_hms_project import storms as stm
from ..hec_hms_project.prepare_hec_hms_project_algorithm import (
    CN_METHODS,
    LAMBDA_OPTIONS,
    QUARTILES,
    SOURCES,
    PrepareHecHmsProjectAlgorithm,
    _advanced,
    _num,
    _software_version,
    _wkt1,
    arf_parameters,
)
from . import core

WIDTHS = [
    (
        core.WIDTH_FLOW,
        "w_flow_m: A / mean overland flow length (Catchment Delineation)",
    ),
    (core.WIDTH_CHANNEL, "w_chan_m: 2 x waterway length (Catchment Delineation)"),
    (core.WIDTH_GUO, "w_gu_m: Guo and Urbonas 2009 (Catchment Delineation)"),
    (core.WIDTH_SQRT, "sqrt(area) (no width field)"),
]
ROUTINGS = [
    (core.ROUTE_DYNWAVE, "Dynamic wave"),
    (core.ROUTE_KINWAVE, "Kinematic wave"),
]
SECTIONS = [
    (core.XS_TRAPEZOID, "Trapezoid (bottom width, depth, side slope)"),
    (core.XS_TRANSECT, "Irregular sections cut from the DEM (HEC-2 transects)"),
]
OUTPUT_SPECS = [
    OutputSpec("OUTPUT_CSV", "Model set-up and peak flows (sectioned CSV)", "SWMM Model Setup", CSV),
    OutputSpec("OUTPUT_DOCX", "Model set-up report (Word)", "SWMM Model Report", WORD),
]  # fmt: skip
SPEC_BY_KEY = {s.key: s for s in OUTPUT_SPECS}
FOLDER_PARAMS = {
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


class PrepareSwmmProjectAlgorithm(PrepareHecHmsProjectAlgorithm):
    """Reuses the HEC-HMS tool's input readers (_read_subcatchments,
    _read_streams, _read_nodes, _storms)."""

    WIDTH_SOURCE = "WIDTH_SOURCE"
    SLOPE_FIELD = "SLOPE_FIELD"
    XS_METHOD = "XS_METHOD"
    SWMM_ROUTING = "SWMM_ROUTING"
    RUN_SWMM = "RUN_SWMM"
    SWMM_NAME = "SWMM_NAME"
    FREQ_STEP = "FREQ_STEP"
    DEFAULT_SLOPE = "DEFAULT_SLOPE"
    N_IMPERV = "N_IMPERV"
    N_PERV = "N_PERV"
    DS_IMPERV = "DS_IMPERV"
    DS_PERV = "DS_PERV"
    PCT_ZERO = "PCT_ZERO"
    DRY_DAYS = "DRY_DAYS"
    CHANNEL_N = "CHANNEL_N"
    BOTTOM_WIDTH = "BOTTOM_WIDTH"
    CHANNEL_DEPTH = "CHANNEL_DEPTH"
    SIDE_SLOPE = "SIDE_SLOPE"
    TRANSECT_WIDTH = "TRANSECT_WIDTH"
    OVERBANK_N = "OVERBANK_N"
    MIN_SLOPE = "MIN_SLOPE"
    SWMM_TAIL = "SWMM_TAIL"
    FOLDER_SWMM = "FOLDER_SWMM"
    OUTPUT_INP = "OUTPUT_INP"

    def createInstance(self):
        return PrepareSwmmProjectAlgorithm()

    def name(self):
        return "prepare_swmm_project"

    def displayName(self):
        return "Prepare SWMM project"

    def shortHelpString(self):
        return (
            f"Prepare SWMM project (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE\n"
            "Builds an EPA SWMM 5 model (.inp) from the GIS outputs of the catchment "
            "tools: subcatchments, junctions at the stream nodes, open-channel conduits "
            "along the reaches and outfalls, with every design storm in the file as a "
            "rainfall time series. The file opens in EPA SWMM 5 and PCSWMM and imports "
            "into Autodesk Storm and Sanitary Analysis (File > Import > EPA SWMM input "
            "file); the tool can also compute every storm in the EPA SWMM engine and "
            "report the critical storm per method and return period.\n"
            "\n"
            "METHOD\n"
            "Network and rainfall as in Prepare HEC-HMS project: each sub-catchment "
            "drains to the junction at its stream node; each reach is a conduit from "
            "the node at its top to the node below; nodes with no reach leaving them are "
            "outfalls (normal depth).\n"
            "Subcatchments: area, % impervious (optional field), width from the "
            "Catchment Delineation field chosen (w_flow_m by default; sqrt(area) when "
            "missing), slope from s_mean_pc (else a default), Manning n and depression "
            "storage, Curve Number infiltration from the CN raster (or a CN field).\n"
            "Nodes: invert = lowest DEM cell near the node, lowered where needed so every "
            "conduit falls at least the minimum slope; without a DEM, relative inverts "
            "from the reach slopes.\n"
            "Conduits: reach length, Manning n, and a trapezoid or an irregular section cut "
            "across the middle of the reach from the DEM (HEC-2 transect).\n"
            "Rainfall: every design storm (DDF frequency storms as nested curves with each "
            "chosen peak position, Alternating Block, Huff, ensembles) as a CUMULATIVE "
            "time series in mm on a common step; the rain gage RG1 uses the first storm - "
            "select another time series for the gage to run another storm.\n"
            "Areal reduction: the DDF depths are point rainfall (also Design Rainfall SA, which is gridded but holds point values, and gridded rainfall adjusted to a point DDF) and are reduced to catchment rainfall: SANRAL Drainage Manual Eq 3.10 (default), Pietersen et al. (2024) regional ARFs for 24 h - 7 days (South Africa; Eq 3.10 below 24 h), the UK Flood Estimation Handbook ARF (Keers and Wescott 1977), a user ARF table, a constant, or none (DDF already areal, e.g. unadjusted gridded rainfall). Nested frequency storms and alternating block storms are reduced per duration (each nested / cumulative depth with the ARF of its own duration, so they follow the areal DDF curve); Huff and ensemble storms with the ARF of the storm duration. The ARF area is the total area draining to the outlet (the largest system if there are several) unless given. Alternating Block CSVs already reduced by that tool (ArfMethod column) are not reduced again. The CSV and report list the ARF per duration and a burst check: the largest 15 min ... depth inside each hyetograph as a % of the areal DDF depth (well below 100 %: the pattern is smoother than the DDF, typical of ERA5 patterns on a small catchment).\n"
            "Run-out: 2 x the HEC-HMS run-out (5 x lag + routing), at least 6 h, unless set.\n"
            "\n"
            "BACKGROUND\n"
            "EPA SWMM 5 User's and Reference Manuals (Rossman 2015, 2017; Rossman and "
            "Huber 2016); NEH 630 Ch 10 (USDA-NRCS); Guo and Urbonas (2009) widths. "
            "The files were checked by computing them in EPA SWMM 5.2 and importing them "
            "into Storm and Sanitary Analysis 2024.\n"
            "\n"
            "PARAMETERS\n"
            "Sub-catchments: Catchment Delineation catchments (incremental mode): name "
            "outlet_id, node, Tc tc_usbr_mn, widths w_flow_m / w_chan_m / w_gu_m, slope "
            "s_mean_pc.\n"
            "Streams / nodes: Catchment Delineation or Stream Network layers (name, "
            "ds_name, length_m, slope).\n"
            "DEM: for the node inverts and DEM transects.\n"
            "Peak positions, return periods, durations: as in Prepare HEC-HMS project.\n"
            "Compute in SWMM: needs swmm-toolkit in QGIS's Python "
            "(python -m pip install swmm-toolkit).\n"
            "Areal reduction factor: method (advanced: ARF area in km2, 0 = automatic; Pietersen region 1 - 5; user table CSV with DurationMin, ARF and optionally AreaKm2; constant).\n"
            "Outputs: <project number>_<SWMM name>.inp in the SWMM folder, plus the "
            "ticked CSV and Word report named <project number>_<description>.\n"
        )

    def initAlgorithm(self, config=None):
        P = self.addParameter
        P(
            QgsProcessingParameterFeatureSource(
                self.SUBCATCHMENTS,
                "Sub-catchments (polygons)",
                [QgsProcessing.TypeVectorPolygon],
            )
        )
        P(
            QgsProcessingParameterField(
                self.SUB_NAME_FIELD,
                "Sub-catchment name field",
                defaultValue="outlet_id",
                parentLayerParameterName=self.SUBCATCHMENTS,
                optional=True,
            )
        )
        P(
            QgsProcessingParameterField(
                self.SUB_NODE_FIELD,
                "Field with the stream node each sub-catchment drains to",
                defaultValue="node",
                parentLayerParameterName=self.SUBCATCHMENTS,
                optional=True,
            )
        )
        P(
            QgsProcessingParameterField(
                self.TC_FIELD,
                "Time of concentration field (min; sets the run-out time)",
                defaultValue="tc_usbr_mn",
                parentLayerParameterName=self.SUBCATCHMENTS,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        P(
            QgsProcessingParameterEnum(
                self.WIDTH_SOURCE,
                "Subcatchment width",
                options=[lbl for _, lbl in WIDTHS],
                defaultValue=0,
            )
        )
        P(
            QgsProcessingParameterField(
                self.SLOPE_FIELD,
                "Subcatchment slope field (%)",
                defaultValue="s_mean_pc",
                parentLayerParameterName=self.SUBCATCHMENTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        P(
            QgsProcessingParameterFeatureSource(
                self.STREAMS,
                "Stream reaches (lines)",
                [QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )
        P(
            QgsProcessingParameterField(
                self.STREAM_NAME_FIELD,
                "Stream name field",
                defaultValue="name",
                parentLayerParameterName=self.STREAMS,
                optional=True,
            )
        )
        P(
            QgsProcessingParameterField(
                self.STREAM_DS_FIELD,
                "Downstream stream name field",
                defaultValue="ds_name",
                parentLayerParameterName=self.STREAMS,
                optional=True,
            )
        )
        P(
            QgsProcessingParameterFeatureSource(
                self.NODES,
                "Stream nodes (points; optional)",
                [QgsProcessing.TypeVectorPoint],
                optional=True,
            )
        )
        P(
            QgsProcessingParameterField(
                self.NODE_NAME_FIELD,
                "Node name field",
                defaultValue="name",
                parentLayerParameterName=self.NODES,
                optional=True,
            )
        )
        P(
            QgsProcessingParameterRasterLayer(
                self.DEM, "DEM (node inverts and channel sections)", optional=True
            )
        )
        P(
            QgsProcessingParameterRasterLayer(
                self.CN_RASTER,
                "SCS curve number raster (curve_numbers.tif)",
                optional=True,
            )
        )
        P(
            QgsProcessingParameterField(
                self.CN_FIELD,
                "Curve number field (instead of the raster; optional)",
                parentLayerParameterName=self.SUBCATCHMENTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        P(
            QgsProcessingParameterEnum(
                self.CN_METHOD,
                "Subcatchment curve number from the raster",
                options=[lbl for _, lbl in CN_METHODS],
                defaultValue=0,
            )
        )
        P(
            QgsProcessingParameterEnum(
                self.SWMM_ROUTING,
                "Flow routing",
                options=[lbl for _, lbl in ROUTINGS],
                defaultValue=0,
            )
        )
        P(
            QgsProcessingParameterEnum(
                self.XS_METHOD,
                "Channel cross-sections",
                options=[lbl for _, lbl in SECTIONS],
                defaultValue=0,
            )
        )
        P(
            QgsProcessingParameterEnum(
                self.RAIN_SOURCES,
                "Design storms from",
                options=[lbl for _, lbl in SOURCES],
                allowMultiple=True,
                defaultValue=[0],
            )
        )
        P(
            QgsProcessingParameterFile(
                self.DDF_CSV,
                "DDF table (CSV; Design Rainfall or Precipitation data to DDF)",
                extension="csv",
                optional=True,
            )
        )
        P(
            QgsProcessingParameterFile(
                self.AB_CSV,
                "Alternating Block incremental depth CSV (optional)",
                extension="csv",
                optional=True,
            )
        )
        P(
            QgsProcessingParameterFile(
                self.HUFF_CSV,
                "Huff curves CSV (optional)",
                extension="csv",
                optional=True,
            )
        )
        P(
            QgsProcessingParameterFile(
                self.ENS_CSV,
                "Design storm ensemble CSV (optional)",
                extension="csv",
                optional=True,
            )
        )
        for p in arf_parameters(self)[0]:
            P(p)
        P(
            QgsProcessingParameterString(
                self.ARIS,
                "Return periods (years; blank = all in the table)",
                defaultValue="2, 5, 10, 20, 50, 100",
                optional=True,
            )
        )
        P(
            QgsProcessingParameterString(
                self.DURATIONS,
                "Storm durations (blank = all Alternating Block rows)",
                defaultValue="1h, 3h, 6h, 12h, 24h",
                optional=True,
            )
        )
        P(
            QgsProcessingParameterEnum(
                self.PEAK,
                "Peak positions (% of the duration; frequency and Alternating Block storms - one storm per position)",
                options=[f"{p} %" for p in stm.PEAK_POSITIONS],
                allowMultiple=True,
                defaultValue=list(range(len(stm.PEAK_POSITIONS))),
            )
        )
        P(
            QgsProcessingParameterString(
                self.SWMM_NAME, "SWMM model name", defaultValue="SWMM Model"
            )
        )
        P(
            QgsProcessingParameterBoolean(
                self.RUN_SWMM,
                "Compute every storm in the EPA SWMM engine (swmm-toolkit)",
                defaultValue=False,
            )
        )
        P(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix and report, optional)",
                optional=True,
            )
        )
        P(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project / site name (report title page, optional)",
                optional=True,
            )
        )
        D = QgsProcessingParameterNumber.Double
        adv = [
            QgsProcessingParameterEnum(self.LAMBDA, "Ia / S for the runoff-equivalent CN", options=[lbl for _, lbl in LAMBDA_OPTIONS], defaultValue=0),
            QgsProcessingParameterNumber(self.CN_DEPTH, "CN design depth for the runoff-equivalent CN (mm)", D, defaultValue=50.0, minValue=1.0),
            QgsProcessingParameterField(self.IMPERV_FIELD, "Impervious area field (% of the sub-catchment; optional)", parentLayerParameterName=self.SUBCATCHMENTS, type=QgsProcessingParameterField.Numeric, optional=True),
            QgsProcessingParameterNumber(self.TC_MIN, "Minimum time of concentration (min)", D, defaultValue=5.0, minValue=0.5),
            QgsProcessingParameterNumber(self.DEFAULT_SLOPE, "Subcatchment slope where none is given (%)", D, defaultValue=2.0, minValue=0.01),
            QgsProcessingParameterNumber(self.N_IMPERV, "Manning n, impervious", D, defaultValue=0.015, minValue=0.005),
            QgsProcessingParameterNumber(self.N_PERV, "Manning n, pervious", D, defaultValue=0.15, minValue=0.01),
            QgsProcessingParameterNumber(self.DS_IMPERV, "Depression storage, impervious (mm)", D, defaultValue=1.5, minValue=0.0),
            QgsProcessingParameterNumber(self.DS_PERV, "Depression storage, pervious (mm)", D, defaultValue=5.0, minValue=0.0),
            QgsProcessingParameterNumber(self.PCT_ZERO, "Impervious area without depression storage (%)", D, defaultValue=25.0, minValue=0.0, maxValue=100.0),
            QgsProcessingParameterNumber(self.DRY_DAYS, "Curve Number drying time (days)", D, defaultValue=7.0, minValue=1.0),
            QgsProcessingParameterNumber(self.CHANNEL_N, "Channel Manning n", D, defaultValue=0.04, minValue=0.01),
            QgsProcessingParameterNumber(self.BOTTOM_WIDTH, "Trapezoid bottom width (m)", D, defaultValue=5.0, minValue=0.1),
            QgsProcessingParameterNumber(self.CHANNEL_DEPTH, "Trapezoid depth (m)", D, defaultValue=2.0, minValue=0.1),
            QgsProcessingParameterNumber(self.SIDE_SLOPE, "Trapezoid side slope (H:V)", D, defaultValue=2.0, minValue=0.0),
            QgsProcessingParameterNumber(self.TRANSECT_WIDTH, "DEM transect width (m)", D, defaultValue=100.0, minValue=5.0),
            QgsProcessingParameterNumber(self.OVERBANK_N, "Overbank Manning n (DEM transects)", D, defaultValue=0.06, minValue=0.01),
            QgsProcessingParameterNumber(self.MIN_SLOPE, "Minimum conduit slope (m/m)", D, defaultValue=0.001, minValue=0.00001),
            QgsProcessingParameterString(self.DDF_SITE, "DDF site (multi-site Design Rainfall CSV; blank = first)", optional=True),
            QgsProcessingParameterEnum(self.HUFF_QUARTILE, "Huff quartile", options=[lbl for _, lbl in QUARTILES], defaultValue=0),
            QgsProcessingParameterEnum(self.HUFF_STAT, "Huff curve", options=stm.HUFF_STATS, defaultValue=2),
            QgsProcessingParameterEnum(self.ENS_SHAPES, "Ensemble storm shapes", options=stm.SHAPES, allowMultiple=True, defaultValue=[0, 1, 2]),
            QgsProcessingParameterNumber(self.HYETO_STEP, "Huff hyetograph time step (min)", D, defaultValue=5.0, minValue=1.0),
            QgsProcessingParameterNumber(self.FREQ_STEP, "Frequency storm time step (min)", D, defaultValue=5.0, minValue=1.0),
            QgsProcessingParameterNumber(self.SWMM_TAIL, "Simulation time after the longest storm (h; 0 = automatic)", D, defaultValue=0.0, minValue=0.0),
            QgsProcessingParameterString(self.REACH_PREFIX, "Conduit name prefix", defaultValue="R-", optional=True),
            QgsProcessingParameterString(self.JUNCTION_PREFIX, "Junction name prefix", defaultValue="J-", optional=True),
            QgsProcessingParameterString(self.SINK_PREFIX, "Outfall name prefix", defaultValue="Outlet-", optional=True),
        ]  # fmt: skip
        for p in arf_parameters(self)[1] + adv:
            P(_advanced(p))
        for spec in OUTPUT_SPECS:
            P(
                QgsProcessingParameterBoolean(
                    spec.key, f"Output: {spec.label}", defaultValue=spec.default
                )
            )
        P(
            QgsProcessingParameterFolderDestination(
                self.FOLDER_SWMM, "Output folder for the SWMM input file"
            )
        )
        for name, label in FOLDER_PARAMS.values():
            P(QgsProcessingParameterFolderDestination(name, label, optional=True))

    # ------------------------------------------------------------------ run

    def processAlgorithm(self, parameters, context, feedback):
        self._post_processors = []
        source = self.parameterAsSource(parameters, self.SUBCATCHMENTS, context)
        if source is None:
            raise QgsProcessingException("Choose the sub-catchment layer.")
        crs = source.sourceCrs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The sub-catchments are in a geographic CRS (degrees). Reproject them to a "
                "projected CRS in metres first."
            )
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        )
        project_name = self.parameterAsString(
            parameters, self.PROJECT_NAME, context
        ).strip()
        model_name = self.parameterAsString(parameters, self.SWMM_NAME, context).strip()
        stem = "_".join(
            p
            for p in (clean_name_part(project_number), clean_name_part(model_name))
            if p
        )
        if not stem:
            raise QgsProcessingException("Give the SWMM model name.")
        swmm_folder = self.parameterAsFileOutput(parameters, self.FOLDER_SWMM, context)
        if not swmm_folder:
            raise QgsProcessingException(
                "Choose the output folder for the SWMM input file."
            )
        os.makedirs(swmm_folder, exist_ok=True)
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
        extra = self._sub_extra(parameters, context, source, subs)
        cn_layer = self.parameterAsRasterLayer(parameters, self.CN_RASTER, context)
        lam = LAMBDA_OPTIONS[self.parameterAsEnum(parameters, self.LAMBDA, context)][0]
        cn_method = CN_METHODS[
            self.parameterAsEnum(parameters, self.CN_METHOD, context)
        ][0]
        cn_depth = self.parameterAsDouble(parameters, self.CN_DEPTH, context)
        cn_label = f"field {cn_field}" if cn_field else "-"
        if cn_layer is not None and not cn_field:
            feedback.setProgressText("Curve numbers per subcatchment")
            tr = QgsCoordinateTransform(crs, cn_layer.crs(), context.transformContext())
            wkts = []
            for g in geoms:
                from qgis.core import QgsGeometry

                gg = QgsGeometry(g)
                if cn_layer.crs() != crs:
                    gg.transform(tr)
                wkts.append(gg.asWkt())
            try:
                stats, band = hms.zonal_cn_from_raster(
                    cn_layer.source(), wkts, cn_method, cn_depth, lam
                )
            except hms.HmsProjectError as e:
                raise QgsProcessingException(str(e)) from e
            cn_label = f"{cn_layer.source()} ({band})"
            for x, st in zip(subs, stats, strict=True):
                if st.get("n_cells"):
                    x.cn, x.cn_stats = st["cn"], st
        elif not cn_field:
            raise QgsProcessingException("Give the curve number raster or a CN field.")
        missing = [x.name for x in subs if x.cn is None]
        if missing:
            raise QgsProcessingException(
                f"No curve number for {len(missing)} sub-catchment(s): "
                + ", ".join(missing[:10])
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

        hs = hms.Settings(
            project_name=stem,
            reach_prefix=self.parameterAsString(parameters, self.REACH_PREFIX, context),
            junction_prefix=self.parameterAsString(
                parameters, self.JUNCTION_PREFIX, context
            ),
            sink_prefix=self.parameterAsString(parameters, self.SINK_PREFIX, context),
            lambda_ia=lam,
            cn_method=cn_method,
            cn_design_depth_mm=cn_depth,
            tc_floor_min=self.parameterAsDouble(parameters, self.TC_MIN, context),
            routing=hms.ROUTE_LAG,
            ensembles=False,
        )
        try:
            r = hms.assemble(subs, reaches, nodes, storms, hs)
        except (hms.HmsProjectError, stm.StormError) as e:
            raise QgsProcessingException(str(e)) from e
        r.arf = arf_model

        tail = self.parameterAsDouble(parameters, self.SWMM_TAIL, context)
        s = core.SwmmSettings(
            title=(project_name + " - " if project_name else "")
            + "prepared with Mayim Tools",
            routing=ROUTINGS[
                self.parameterAsEnum(parameters, self.SWMM_ROUTING, context)
            ][0],
            width_source=WIDTHS[
                self.parameterAsEnum(parameters, self.WIDTH_SOURCE, context)
            ][0],
            default_slope_pc=self.parameterAsDouble(
                parameters, self.DEFAULT_SLOPE, context
            ),
            n_imperv=self.parameterAsDouble(parameters, self.N_IMPERV, context),
            n_perv=self.parameterAsDouble(parameters, self.N_PERV, context),
            ds_imperv_mm=self.parameterAsDouble(parameters, self.DS_IMPERV, context),
            ds_perv_mm=self.parameterAsDouble(parameters, self.DS_PERV, context),
            pct_zero=self.parameterAsDouble(parameters, self.PCT_ZERO, context),
            dry_days=self.parameterAsDouble(parameters, self.DRY_DAYS, context),
            xs_method=SECTIONS[
                self.parameterAsEnum(parameters, self.XS_METHOD, context)
            ][0],
            channel_n=self.parameterAsDouble(parameters, self.CHANNEL_N, context),
            bottom_width_m=self.parameterAsDouble(
                parameters, self.BOTTOM_WIDTH, context
            ),
            channel_depth_m=self.parameterAsDouble(
                parameters, self.CHANNEL_DEPTH, context
            ),
            side_slope=self.parameterAsDouble(parameters, self.SIDE_SLOPE, context),
            transect_half_width_m=self.parameterAsDouble(
                parameters, self.TRANSECT_WIDTH, context
            )
            / 2.0,
            overbank_n=self.parameterAsDouble(parameters, self.OVERBANK_N, context),
            min_slope=self.parameterAsDouble(parameters, self.MIN_SLOPE, context),
            hyetograph_step_min=self.parameterAsDouble(
                parameters, self.FREQ_STEP, context
            ),
            tail_h=tail if tail > 0 else None,
        )
        dem = self.parameterAsRasterLayer(parameters, self.DEM, context)
        sampler = None
        if dem is not None:
            feedback.setProgressText("Reading the DEM")
            from qgis.core import QgsGeometry

            ext = QgsGeometry.unaryUnion(geoms).boundingBox()
            try:
                sampler = core.DemSampler.from_file(
                    dem.source(),
                    _wkt1(crs),
                    (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum()),
                )
            except Exception as e:  # noqa: BLE001
                feedback.pushWarning(
                    f"The DEM could not be read ({e}); relative inverts are used."
                )
        try:
            m = core.build_model(
                r,
                s,
                sub_extra=extra,
                dem=sampler,
                reach_points={k: max(v, key=len) for k, v in lines.items() if v},
                reach_slopes={x.name: x.slope for x in reaches},
            )
        except (core.SwmmProjectError, hms.HmsProjectError, stm.StormError) as e:
            raise QgsProcessingException(str(e)) from e
        m.warnings = storm_warnings + r.warnings + m.warnings
        m.crs_name = f"{crs.authid()} - {crs.description()}"

        from .inp import write_inp

        inp_path = os.path.join(swmm_folder, stem + ".inp")
        try:
            write_inp(inp_path, m)
        except OSError as e:
            raise QgsProcessingException(
                f"{inp_path} could not be written ({e})."
            ) from e
        m.files["SWMM input file"] = inp_path
        feedback.pushInfo(f"SWMM input file: {inp_path}")

        results = None
        if self.parameterAsBoolean(parameters, self.RUN_SWMM, context):
            from . import run

            feedback.setProgressText(f"Computing {len(m.series)} storm(s) in EPA SWMM")
            try:
                results = run.run_all(
                    m, os.path.join(swmm_folder, stem + "_runs"), feedback=feedback
                )
            except run.SwmmUnavailableError as e:
                feedback.pushWarning(str(e))
            for res in results or []:
                for err in res.errors[:3]:
                    m.warnings.append(f"SWMM {res.run}: {err}")
                if (
                    res.routing_error_pct is not None
                    and abs(res.routing_error_pct) > 10
                ):
                    m.warnings.append(
                        f"SWMM {res.run}: flow routing continuity error "
                        f"{res.routing_error_pct:g} % (large; often a storm with very "
                        "little runoff, else shorten the routing step or lengthen the run-out)."
                    )

        outputs = {self.OUTPUT_INP: inp_path, self.FOLDER_SWMM: swmm_folder}
        written = [("SWMM input file", inp_path)]
        inputs = [
            ("Sub-catchments", f"{source.sourceName()} ({len(subs)} polygons)"),
            ("CRS", m.crs_name),
            ("Streams", ssrc.sourceName() if ssrc is not None else "-"),
            ("Nodes", nsrc.sourceName() if nsrc is not None else "-"),
            ("Curve numbers", cn_label),
            ("DEM", dem.source() if dem is not None else "-"),
            (
                "DDF table",
                self.parameterAsFile(parameters, self.DDF_CSV, context) or "-",
            ),
            ("Storm sources", ", ".join(stm.SOURCE_LABELS[c] for c in chosen)),
        ]
        if "OUTPUT_CSV" in paths:
            from .export import write_csv

            try:
                write_csv(paths["OUTPUT_CSV"], m, inputs, results)
                outputs["OUTPUT_CSV"] = paths["OUTPUT_CSV"]
                written.append((SPEC_BY_KEY["OUTPUT_CSV"].label, paths["OUTPUT_CSV"]))
            except OSError as e:
                feedback.reportError(
                    f"The CSV could not be written ({e}); it may be open.",
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
                    m,
                    {
                        "project_number": project_number,
                        "project_name": project_name,
                        "software": _software_version(),
                        "inputs": inputs,
                        "outputs": written + [("This report", paths["OUTPUT_DOCX"])],
                        "warnings": m.warnings,
                    },
                    geometry,
                    results,
                )
                for msg in notes:
                    feedback.pushWarning(msg)
                outputs["OUTPUT_DOCX"] = paths["OUTPUT_DOCX"]
                written.append((SPEC_BY_KEY["OUTPUT_DOCX"].label, paths["OUTPUT_DOCX"]))
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs python-docx and matplotlib in QGIS's Python. All "
                    "other outputs were written."
                )
            except Exception as e:  # noqa: BLE001
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for w in m.warnings:
            feedback.pushWarning(w)
        feedback.pushInfo(
            f"{len(m.subcatchments)} subcatchments, {len(m.conduits)} conduits, "
            f"{sum(1 for n in m.nodes if n.kind == 'junction')} junctions, "
            f"{sum(1 for n in m.nodes if n.kind == 'outfall')} outfall(s); "
            f"{len(m.series)} storm(s) on a {m.step_min:g} min step."
        )
        if results:
            from .export import critical_rows

            for row in critical_rows(m, results):
                feedback.pushInfo(
                    f"{row[0]} {row[1]} at {row[3]}: {row[4]:.3f} m3/s ({row[5]})"
                )
        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        return outputs

    def _sub_extra(self, parameters, context, source, subs):
        """Catchment Delineation width and slope fields per sub-catchment."""
        names = source.fields().names()
        name_f = self.parameterAsString(parameters, self.SUB_NAME_FIELD, context)
        slope_f = self.parameterAsString(parameters, self.SLOPE_FIELD, context)
        wanted = [f for f in core.WIDTH_FIELDS if f in names]
        out = {}
        for f in source.getFeatures():
            nm = (
                str(f[name_f]).strip()
                if name_f and name_f in names and f[name_f] is not None
                else ""
            ) or f"SB{f.id()}"
            d = {k: _num(f[k]) for k in wanted}
            if slope_f and slope_f in names:
                d["s_mean_pc"] = _num(f[slope_f])
            out[nm] = d
        return out
