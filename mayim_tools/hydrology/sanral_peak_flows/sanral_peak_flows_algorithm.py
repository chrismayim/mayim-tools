"""Processing Toolbox algorithm: catchment polygons ('Catchments: 1. Delineation'
output or any layer with the same attributes) + design rainfall DDF + runoff
coefficient CSV in -> design peak flows by the SANRAL Drainage Manual
Chapter 3 methods and DWS CAPA out (catchment layer, sectioned CSV,
results.json and a Word annexure).

Outputs follow the Mayim output standard: a tick box per output, shapefile,
CSV and Word folders, files named <project number>_<description>.

Thin wrapper - all real logic lives in core.py / methods.py (zero QGIS
dependency; see tests/test_sanral_peak_flows_core.py).
"""

import os
from datetime import datetime
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsMemoryProviderUtils,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    SHAPEFILE,
    WORD,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import (
    STYLE_CATCHMENTS,
    load_on_completion,
    save_shapefile,
)

from . import tables as tb
from .core import TOOL_VERSION, CatchmentInput, Settings, field_names, run
from .export import field_definition, write_csv, write_results_json
from .inputs import InputError, read_c_csv, read_tr102_csv
from .methods import METHOD_ORDER, METHODS

RP_OPTIONS = [f"1:{T} year" for T in tb.RETURN_PERIODS]
NOT_SET = "Not set (use the field, or skip)"
BASIN_OPTIONS = [NOT_SET] + [f"{b} - {v[1]} ({v[0]})" for b, v in tb.SDF_BASINS.items()]
BASIN_KEYS = [None] + list(tb.SDF_BASINS)
VELD_OPTIONS = [NOT_SET] + [f"{k} - {v[0]}" for k, v in tb.VELD_ZONES.items()]
VELD_KEYS = [None] + list(tb.VELD_ZONES)
KOVACS_LABELS = {
    "K1": "K1 (K = 2.8)",
    "K2": "K2 (K = 3.4)",
    "K3": "K3 (K = 4.0)",
    "K4": "K4 (K = 4.6)",
    "K5": "K5 (K = 5.0)",
    "K5_SWC": "K5 in the SW Cape - G, H drainage regions (K = 5.0)",
    "K6": "K6 (K = 5.2)",
    "K7": "K7 (K = 5.4)",
    "K8": "K8 (K = 5.6)",
}
KOVACS_OPTIONS = [NOT_SET] + list(KOVACS_LABELS.values())
KOVACS_KEYS = [None] + list(KOVACS_LABELS)
ARF_METHOD_OPTIONS = [
    "DM Eq 3.10 (UK FSR / Alexander) - Manual default",
    "Pietersen et al (2024) regional ARF for D >= 24 h (else Eq 3.10)",
]
ARF_METHOD_KEYS = ["eq310", "pietersen"]
ARF_REGION_OPTIONS = [NOT_SET] + [f"ARF region {r}" for r in range(1, 6)]
ARF_REGION_KEYS = [None, 1, 2, 3, 4, 5]

# Selectable outputs: key, dialog label, file-name description, folder kind,
# ticked by default. results.json is written to the CSV folder.
OUTPUT_SPECS = [
    OutputSpec(
        "OUTPUT", "Catchments with design peak flows (shapefile)",
        "Peak Flow Catchments", SHAPEFILE,
    ),
    OutputSpec("OUTPUT_CSV", "Peak flow calculations (CSV)", "Peak Flow Calculations", CSV),
    OutputSpec("OUTPUT_DOCX", "Peak flow annexure (Word)", "Peak Flow Annexure", WORD),
    OutputSpec(
        "OUTPUT_SHEETS", "Peak flow calculation sheets per method (Word)",
        "Peak Flow Calculation Sheets", WORD,
    ),
    OutputSpec(
        "OUTPUT_JSON", "Results for the report compiler (results.json)",
        "Peak Flow Results", CSV, False,
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files (and results.json)"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


def output_path(spec, folders, project_number):
    path = spec.path(folders, project_number)
    if spec.key == "OUTPUT_JSON":
        path = str(Path(path).with_suffix(".json"))
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


def _advanced(param):
    param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
    return param


def _num(value):
    if value is None:
        return None
    try:
        text = str(value).strip().replace(",", ".")
        if text in ("", "NULL", "None"):
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _text(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ("", "NULL", "None") else text


class SanralPeakFlowsAlgorithm(QgsProcessingAlgorithm):
    CATCHMENTS = "CATCHMENTS"
    ID_FIELD = "ID_FIELD"
    AREA_FIELD = "AREA_FIELD"
    LFP_FIELD = "LFP_FIELD"
    S1085_FIELD = "S1085_FIELD"
    LCA_FIELD = "LCA_FIELD"
    SMEAN_FIELD = "SMEAN_FIELD"
    TC_FIELD = "TC_FIELD"
    INEFF_FIELD = "INEFF_FIELD"
    MODE_FIELD = "MODE_FIELD"
    METHODS = "METHODS"
    RETURN_PERIODS = "RETURN_PERIODS"
    DDF = "DDF"
    C_CSV = "C_CSV"
    MAP = "MAP"
    MAP_FIELD = "MAP_FIELD"
    SDF_BASIN = "SDF_BASIN"
    SDF_FIELD = "SDF_FIELD"
    VELD = "VELD"
    VELD_FIELD = "VELD_FIELD"
    KOVACS = "KOVACS"
    KOVACS_FIELD = "KOVACS_FIELD"
    TC_FLOOR = "TC_FLOOR"
    APPLY_ARF = "APPLY_ARF"
    ARF_METHOD = "ARF_METHOD"
    ARF_REGION = "ARF_REGION"
    ARF_FIELD = "ARF_FIELD"
    TR102_CSV = "TR102_CSV"
    R_DAYS = "R_DAYS"
    K_CUSTOM = "K_CUSTOM"
    OUTPUT = "OUTPUT"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_JSON = "OUTPUT_JSON"
    OUTPUT_DOCX = "OUTPUT_DOCX"
    OUTPUT_SHEETS = "OUTPUT_SHEETS"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return SanralPeakFlowsAlgorithm()

    def name(self):
        return "sanral_peak_flows"

    def displayName(self):
        return "SANRAL Deterministic Peak Flows"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        methods = "\n".join(f"  {v[0]}: {v[1]}" for v in METHODS.values())
        files = "\n".join(
            f"  {s.label}: <project number>_{clean_name_part(s.description)}"
            f"{'.json' if s.key == 'OUTPUT_JSON' else EXTENSIONS[s.kind]}"
            for s in OUTPUT_SPECS
        )
        notes = "\n".join(f"  - {n}" for n in tb.SOURCE_NOTES)
        return (
            f"SANRAL Deterministic Peak Flows (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDesign peak flows for each catchment by the deterministic "
            "and empirical methods of the SANRAL Drainage Manual (2013) Chapter 3 "
            "and the DWS (2025) Catchment Parameter method. Every catchment is an "
            "independent design point (its own area and Tc, also for the ARF); no "
            "routing. The Manual asks for the methods to be compared, not "
            "weighted, and for the SDF always to be included - choosing the "
            "design value is left to the engineer.\n"
            "\n"
            "INPUTS:\tCatchment polygons with area, longest flow path, 10-85 "
            "slope, Lc and mean slope (%) - the 'Catchments: 1. Delineation' field "
            "names are the defaults - and Tc (min), by default tc_sel_mn of the "
            "'Catchments: 2. Time of Concentration' tool. Rational: a C CSV (catchment_id, alpha, "
            "C1, Ft_class, beta, C2, gamma, C3; the C1 breakdown columns of the "
            "Rational C tool are shown on the calculation sheets) and, for "
            "Alternative 3, the "
            "design rainfall DDF CSV (Design Rainfall Estimation or Precipitation "
            "data to DDF). Regions (SDF basin, veld zone, Kovacs region, ARF "
            "region) and MAP apply to all catchments, or per catchment from a "
            "field.\n"
            "\n"
            "RATIONAL:\tQ = C_T I A / 3.6. C_T = alpha C1 Ft + beta C2 + gamma C3 "
            "(Ft from Table 3.8; 1.0 at 1:200). Alternative 2: TR102 + modified "
            "Hershfield (the SDF basin's station unless a TR102 CSV is given). "
            "Alternative 3: the DDF table. Tc below 15 min is raised to 15 min "
            "(switchable). ARF by Eq 3.10, or Pietersen et al (2024) for "
            "durations of 24 h and longer.\n"
            "\n"
            "SDF:\tUsed exactly as published: its own Tc (Eq 3.17), the basin's "
            "TR102 station (Hershfield to 6 h, TR102 interpolation beyond, per "
            "Step 6), Eq 3.19 ARF and the calibrated C2/C100.\n"
            "\n"
            "EMPIRICAL:\tMidgley & Pitman Eq 3.34 (1:10 - 1:100); Kovacs RMF "
            "(Table 3.18) and 1:50 - 1:200 peaks from the QT/QRMF ratios (Table "
            "3D.1); CAPA with the digitised M diagram (DWS 2025 Figure 5-2) and "
            "Table 5-2.\n"
            "\n"
            "FLAGS:\tMethods outside their range of application are calculated "
            "and flagged in the notes field; missing inputs skip only that "
            "method.\n"
            "\n"
            "OUTPUTS:\tTick the outputs wanted and choose the folders. The "
            "catchments shapefile has one Q field per method and return period "
            "(loaded with the catchment style); the sectioned CSV (UTF-8 with a "
            "byte-order mark) starts with the abbreviations; the Word annexure "
            "follows the Mayim report standard (needs python-docx and "
            "matplotlib); the calculation sheets (Word) give one sheet per "
            "catchment and method after the Drainage Manual Appendix 3C forms "
            "(R2, R3, S and E, plus CAPA); results.json (schema mayim.results/1, for the report "
            "compiler, not ticked by default) goes to the CSV folder. Files are "
            "named <project number>_<description> and replace files of the same "
            "name; a file that is open cannot be replaced and is reported, and "
            "the other outputs are still written.\n" + files + "\n"
            "\n"
            "FIELD CODES:\n" + methods + "\n\nSOURCE NOTES:\n" + notes
        )

    # ------------------------------------------------------------------
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CATCHMENTS,
                "Catchments (polygons)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
            )
        )
        numeric = Qgis.ProcessingFieldParameterDataType.Numeric
        for key, label, default, dtype, optional in (
            (self.ID_FIELD, "Catchment ID field", "outlet_id", None, True),
            (self.AREA_FIELD, "Area field (km²)", "area_km2", numeric, False),
            (self.LFP_FIELD, "Longest watercourse length field (km)", "lfp_km", numeric, True),
            (self.S1085_FIELD, "10-85 slope field (m/m)", "s_1085", numeric, True),
            (self.LCA_FIELD, "Length to centroid Lc field (km)", "lca_km", numeric, True),
            (self.SMEAN_FIELD, "Mean catchment slope field (%), for CAPA", "s_mean_pc", numeric, True),
            (self.TC_FIELD, "Tc field for the Rational method (min)", "tc_sel_mn", numeric, True),
        ):  # fmt: skip
            kwargs = {"parentLayerParameterName": self.CATCHMENTS, "defaultValue": default,
                      "optional": optional}  # fmt: skip
            if dtype is not None:
                kwargs["type"] = dtype
            self.addParameter(QgsProcessingParameterField(key, label, **kwargs))
        self.addParameter(
            _advanced(
                QgsProcessingParameterField(
                    self.INEFF_FIELD,
                    "Ineffective area field (km², pans/marshes; Ae = A - this)",
                    parentLayerParameterName=self.CATCHMENTS,
                    type=numeric,
                    optional=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterField(
                    self.MODE_FIELD,
                    "Catchment mode field (total / incremental)",
                    parentLayerParameterName=self.CATCHMENTS,
                    defaultValue="mode",
                    optional=True,
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHODS,
                "Methods",
                options=[METHODS[m][1] for m in METHOD_ORDER],
                allowMultiple=True,
                defaultValue=list(range(len(METHOD_ORDER))),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RETURN_PERIODS,
                "Return periods",
                options=RP_OPTIONS,
                allowMultiple=True,
                defaultValue=list(range(len(RP_OPTIONS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.DDF,
                "Design rainfall DDF CSV (Rational Alternative 3)",
                extension="csv",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.C_CSV,
                "Runoff coefficient CSV (Rational)",
                extension="csv",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAP,
                "Mean annual precipitation, MAP (mm; empty = from the DDF file)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.MAP_FIELD,
                "MAP field (mm, per catchment - overrides MAP)",
                parentLayerParameterName=self.CATCHMENTS,
                type=numeric,
                optional=True,
            )
        )
        for key, fkey, label, options in (
            (self.SDF_BASIN, self.SDF_FIELD, "SDF drainage basin (DM Figure 3.30)", BASIN_OPTIONS),
            (self.VELD, self.VELD_FIELD, "Generalised veld-type zone (DM Figure 3.20)", VELD_OPTIONS),
            (self.KOVACS, self.KOVACS_FIELD, "Kovacs region at the site (DM Figure 3.37)", KOVACS_OPTIONS),
        ):  # fmt: skip
            self.addParameter(
                QgsProcessingParameterEnum(key, label, options=options, defaultValue=0)
            )
            self.addParameter(
                QgsProcessingParameterField(
                    fkey,
                    label.split(" (")[0] + " field (per catchment - overrides)",
                    parentLayerParameterName=self.CATCHMENTS,
                    optional=True,
                )
            )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.TC_FLOOR,
                    "Raise Tc below 15 min to 15 min (Rational, DM p 3-20)",
                    defaultValue=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.APPLY_ARF,
                    "Apply the areal reduction factor (Rational)",
                    defaultValue=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterEnum(
                    self.ARF_METHOD,
                    "ARF method (Rational)",
                    options=ARF_METHOD_OPTIONS,
                    defaultValue=0,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterEnum(
                    self.ARF_REGION,
                    "Pietersen ARF region",
                    options=ARF_REGION_OPTIONS,
                    defaultValue=0,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterField(
                    self.ARF_FIELD,
                    "ARF region field (1 - 5, per catchment - overrides)",
                    parentLayerParameterName=self.CATCHMENTS,
                    optional=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterFile(
                    self.TR102_CSV,
                    "TR102 station CSV (Rational Alternative 2; default: SDF basin station)",
                    extension="csv",
                    optional=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.R_DAYS,
                    "Thunder days per year R (Rational Alternative 2; empty = basin/CSV value)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    minValue=0.0,
                    optional=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.K_CUSTOM,
                    "Custom Kovacs K (overrides the region; ratios then skipped)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    minValue=0.0,
                    maxValue=6.0,
                    optional=True,
                )
            )
        )
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
    def _optional_number(self, parameters, key, context):
        value = parameters.get(key)
        if value is None or str(value).strip() == "":
            return None
        return self.parameterAsDouble(parameters, key, context)

    def _field(self, parameters, key, context, fields):
        name = self.parameterAsString(parameters, key, context)
        if not name:
            return None
        if fields.indexOf(name) < 0:
            return None
        return name

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        source = self.parameterAsSource(parameters, self.CATCHMENTS, context)
        if source is None:
            raise QgsProcessingException("No catchment layer.")
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()
        project_name = (
            self.parameterAsString(parameters, self.PROJECT_NAME, context) or ""
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
                raise QgsProcessingException(
                    f"Choose the {label[0].lower() + label[1:]}."
                )
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {s.key: output_path(s, folders, project_number) for s in selected}
        fields = source.fields()
        f = {
            k: self._field(parameters, k, context, fields)
            for k in (
                self.ID_FIELD, self.AREA_FIELD, self.LFP_FIELD, self.S1085_FIELD,
                self.LCA_FIELD, self.SMEAN_FIELD, self.TC_FIELD, self.INEFF_FIELD,
                self.MODE_FIELD, self.MAP_FIELD, self.SDF_FIELD, self.VELD_FIELD,
                self.KOVACS_FIELD, self.ARF_FIELD,
            )
        }  # fmt: skip
        if f[self.AREA_FIELD] is None:
            raise QgsProcessingException("The area field was not found on the layer.")

        chosen = self.parameterAsEnums(parameters, self.METHODS, context)
        rps = self.parameterAsEnums(parameters, self.RETURN_PERIODS, context)
        if not chosen or not rps:
            raise QgsProcessingException(
                "Choose at least one method and return period."
            )
        settings = Settings(
            methods=tuple(METHOD_ORDER[i] for i in sorted(chosen)),
            return_periods=tuple(tb.RETURN_PERIODS[i] for i in sorted(rps)),
            tc_floor=self.parameterAsBoolean(parameters, self.TC_FLOOR, context),
            apply_arf=self.parameterAsBoolean(parameters, self.APPLY_ARF, context),
            arf_method=ARF_METHOD_KEYS[
                self.parameterAsEnum(parameters, self.ARF_METHOD, context)
            ],
            k_custom=self._optional_number(parameters, self.K_CUSTOM, context),
            r_days=self._optional_number(parameters, self.R_DAYS, context),
        )

        ddf = ddf_info = None
        ddf_path = self.parameterAsFile(parameters, self.DDF, context)
        if ddf_path:
            from mayim_tools.rainfall._common.ddf import DDFTable, read_ddf_csv

            try:
                long_df, ddf_info = read_ddf_csv(ddf_path)
                ddf = DDFTable(long_df)
            except Exception as e:
                raise QgsProcessingException(f"Could not read the DDF CSV: {e}") from e
            ddf_info = dict(ddf_info, path=ddf_path)
        c_values = None
        c_path = self.parameterAsFile(parameters, self.C_CSV, context)
        tr_path = self.parameterAsFile(parameters, self.TR102_CSV, context)
        try:
            if c_path:
                c_values = read_c_csv(c_path)
            tr102 = read_tr102_csv(tr_path, settings.r_days) if tr_path else None
        except InputError as e:
            raise QgsProcessingException(str(e)) from e

        map_default = self._optional_number(parameters, self.MAP, context)
        if map_default is None and ddf_info and ddf_info.get("map_mm"):
            map_default = float(ddf_info["map_mm"])
            feedback.pushInfo(f"MAP {map_default:g} mm taken from the DDF file.")
        basin_default = BASIN_KEYS[
            self.parameterAsEnum(parameters, self.SDF_BASIN, context)
        ]
        veld_default = VELD_KEYS[self.parameterAsEnum(parameters, self.VELD, context)]
        kov_default = KOVACS_KEYS[
            self.parameterAsEnum(parameters, self.KOVACS, context)
        ]
        arf_default = ARF_REGION_KEYS[
            self.parameterAsEnum(parameters, self.ARF_REGION, context)
        ]

        def attr(feat, key):
            name = f[key]
            return None if name is None else feat[name]

        catchments, features = [], []
        for feat in source.getFeatures():
            if feedback.isCanceled():
                return {}
            area = _num(attr(feat, self.AREA_FIELD))
            if area is None:
                feedback.pushWarning(f"Feature {feat.id()} has no area - skipped.")
                continue
            cid = _text(attr(feat, self.ID_FIELD)) or str(feat.id())
            s_mean = _num(attr(feat, self.SMEAN_FIELD))
            basin = _num(attr(feat, self.SDF_FIELD))
            arf_region = _num(attr(feat, self.ARF_FIELD))
            veld = _text(attr(feat, self.VELD_FIELD))
            kov = _text(attr(feat, self.KOVACS_FIELD))
            if veld is not None:
                veld = veld.upper()
                if veld not in tb.VELD_ZONES:
                    feedback.pushWarning(
                        f"{cid}: veld zone '{veld}' unknown - ignored."
                    )
                    veld = None
            if kov is not None:
                kov = kov.upper().replace(" ", "_")
                if kov not in tb.KOVACS_REGIONS:
                    feedback.pushWarning(
                        f"{cid}: Kovacs region '{kov}' unknown - ignored."
                    )
                    kov = None
            basin = int(basin) if basin is not None else basin_default
            if basin is not None and basin not in tb.SDF_BASINS:
                feedback.pushWarning(f"{cid}: SDF basin {basin} unknown - ignored.")
                basin = None
            catchments.append(
                CatchmentInput(
                    catchment_id=cid,
                    area_km2=area,
                    lfp_km=_num(attr(feat, self.LFP_FIELD)),
                    s1085=_num(attr(feat, self.S1085_FIELD)),
                    lca_km=_num(attr(feat, self.LCA_FIELD)),
                    s_mean=None if s_mean is None else s_mean / 100.0,
                    tc_min=_num(attr(feat, self.TC_FIELD)),
                    ineffective_km2=_num(attr(feat, self.INEFF_FIELD)) or 0.0,
                    mode=_text(attr(feat, self.MODE_FIELD)),
                    map_mm=_num(attr(feat, self.MAP_FIELD)) or map_default,
                    sdf_basin=basin,
                    veld_zone=veld or veld_default,
                    kovacs_region=kov or kov_default,
                    arf_region=(
                        int(arf_region) if arf_region is not None else arf_default
                    ),
                )
            )
            features.append(feat)
        if not catchments:
            raise QgsProcessingException("No catchment with an area to process.")

        feedback.setProgressText("Calculating peak flows")
        result = run(
            catchments,
            settings,
            ddf=ddf,
            ddf_info=ddf_info,
            c_values=c_values,
            tr102_override=tr102,
            feedback=feedback,
        )
        if feedback.isCanceled():
            return {}
        for w in result.warnings:
            feedback.pushWarning(w)

        names = field_names(settings)
        out_fields = QgsFields()
        clash = set(names) | {"notes"}
        for fld in fields:
            if fld.name() not in clash:
                out_fields.append(fld)
        for n in names:
            out_fields.append(QgsField(n, QMetaType.Type.Double))
        out_fields.append(QgsField("notes", QMetaType.Type.QString))
        layer = QgsMemoryProviderUtils.createMemoryLayer(
            "Peak flow catchments", out_fields, source.wkbType(), source.sourceCrs()
        )
        feats = []
        for feat, item in zip(features, result.items, strict=True):
            out = QgsFeature(out_fields)
            out.setGeometry(feat.geometry())
            for fld in out_fields:
                n = fld.name()
                if fields.indexOf(n) >= 0:
                    out[n] = feat[n]
            q = item.by_field()
            for n in names:
                v = q.get(n)
                out[n] = None if v is None else round(v, 3)
            out["notes"] = item.notes_text()
            feats.append(out)
        layer.dataProvider().addFeatures(feats)
        layer.updateExtents()

        run_info = {
            "tool": self.displayName(),
            "version": TOOL_VERSION,
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "catchment layer": source.sourceName(),
            "DDF CSV": ddf_path or "",
            "C CSV": c_path or "",
            "TR102 CSV": tr_path or "",
        }
        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []
        self._post_processors = []

        def save(key, writer):
            path = paths[key]
            try:
                writer(path)
            except ImportError as e:
                feedback.pushWarning(
                    f"{SPEC_BY_KEY[key].label}: not written - needs python-docx "
                    f"and matplotlib in QGIS's Python ({e})."
                )
                return
            except Exception as e:
                feedback.reportError(
                    f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be "
                    f"written ({e}). If it is open in QGIS, Excel or Word, close it "
                    "and run again.",
                    fatalError=False,
                )
                return
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))
            feedback.pushInfo(f"Written: {path}")

        if self.OUTPUT in paths:

            def write_layer(p):
                save_shapefile(layer, p, context.transformContext())
                load_on_completion(
                    context, p, Path(p).stem, STYLE_CATCHMENTS, self._post_processors
                )

            save(self.OUTPUT, write_layer)
        if self.OUTPUT_CSV in paths:
            save(self.OUTPUT_CSV, lambda p: write_csv(p, result, run_info))
        if self.OUTPUT_JSON in paths:
            save(
                self.OUTPUT_JSON,
                lambda p: write_results_json(
                    p,
                    result,
                    {"id": self.name(), "name": self.displayName(),
                     "version": TOOL_VERSION, "run_time": run_info["run_time"]},
                    {"files": {"catchments": source.sourceName(),
                               "ddf": ddf_path or None, "c_csv": c_path or None,
                               "tr102_csv": tr_path or None},
                     "parameters": {"methods": list(settings.methods),
                                    "return_periods": list(settings.return_periods),
                                    "tc_floor": settings.tc_floor,
                                    "apply_arf": settings.apply_arf,
                                    "arf_method": settings.arf_method,
                                    "k_custom": settings.k_custom,
                                    "r_days": settings.r_days,
                                    "field_definitions": {
                                        n: field_definition(n) for n in names}}},
                ),
            )  # fmt: skip
        if self.OUTPUT_SHEETS in paths:
            feedback.setProgressText("Writing the calculation sheets")

            def write_sheets_docx(p):
                from .sheets import write_sheets

                for note in write_sheets(
                    p,
                    result,
                    run_info,
                    context={
                        "project_number": project_number,
                        "project_name": project_name,
                        "software": _software_version(),
                    },
                ):
                    feedback.pushInfo(note)

            save(self.OUTPUT_SHEETS, write_sheets_docx)
        if self.OUTPUT_DOCX in paths:
            feedback.setProgressText("Writing the Word annexure")
            docx_path = paths[self.OUTPUT_DOCX]

            def write_docx(p):
                from .report import write_report

                notes = write_report(
                    p,
                    result,
                    run_info,
                    context={
                        "project_number": project_number,
                        "project_name": project_name,
                        "software": _software_version(),
                        "outputs": written + [("This report", docx_path)],
                    },
                )
                for note in notes:
                    feedback.pushInfo(note)

            save(self.OUTPUT_DOCX, write_docx)

        if not written:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )
        for item in result.items:
            q = item.by_field()
            shown = ", ".join(
                f"{n} {q[n]:.1f}"
                for n in names
                if n.endswith(("_50", "_100")) and q.get(n)
            )
            feedback.pushInfo(f"{item.catchment.catchment_id}: {shown or 'no results'}")
        return outputs
