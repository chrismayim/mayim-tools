"""DesignStormEnsembleAlgorithm - the Processing algorithm.

Thin wrapper - all real logic lives in core.py / report.py (zero QGIS
dependency, independently testable; see
tests/test_design_storm_ensembles_core.py). QGIS4/Qt6-safe: this
algorithm only uses File/Number/String/Boolean/FolderDestination
Processing parameters (no vector/raster geometry enums).

Outputs follow the Mayim output standard: a tick box per output, one
CSV folder, files named <project number>_<description>.csv.
"""

import os
from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    OutputSpec,
    clean_name_part,
)

from . import core, report

TOOL_NAME = "Design Storm Ensembles"
TOOL_VERSION = "1.0.1"

# Selectable outputs: key, dialog label, file-name description, folder
# kind, ticked by default. All three share the CSV folder.
OUTPUT_SPECS = [
    OutputSpec(
        "OUTPUT_ENSEMBLE_CSV", "Design storm ensemble (CSV)",
        "Design Storm Ensemble", CSV,
    ),
    OutputSpec(
        "OUTPUT_CATALOGUE_CSV", "Storm catalogue (CSV, diagnostic)",
        "Storm Catalogue", CSV,
    ),
    OutputSpec(
        "OUTPUT_BIN_SUMMARY_CSV", "Bin summary (CSV, diagnostic)",
        "Storm Bin Summary", CSV,
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {CSV: ("FOLDER_CSV", "Output folder for CSV files")}


class DesignStormEnsembleAlgorithm(QgsProcessingAlgorithm):
    INPUT_TIMESERIES = "INPUT_TIMESERIES"
    INPUT_DDF = "INPUT_DDF"
    DATETIME_COLUMN = "DATETIME_COLUMN"
    VALUE_COLUMN = "VALUE_COLUMN"
    IETD_HOURS = "IETD_HOURS"
    MIN_EVENT_DEPTH = "MIN_EVENT_DEPTH"
    MIN_SAMPLE_SIZE = "MIN_SAMPLE_SIZE"
    MC_SIMS = "MC_SIMS"
    MC_ALPHA = "MC_ALPHA"
    RANDOM_SEED = "RANDOM_SEED"
    POOL_SPARSE_BINS = "POOL_SPARSE_BINS"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    OUTPUT_ENSEMBLE_CSV = "OUTPUT_ENSEMBLE_CSV"
    OUTPUT_CATALOGUE_CSV = "OUTPUT_CATALOGUE_CSV"
    OUTPUT_BIN_SUMMARY_CSV = "OUTPUT_BIN_SUMMARY_CSV"

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return DesignStormEnsembleAlgorithm()

    def name(self):
        return "generate_design_storm_ensembles"

    def displayName(self):
        return "Design Storm Ensembles"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        files = "\n".join(
            f"  {spec.label}: <project number>_"
            f"{clean_name_part(spec.description)}{EXTENSIONS[spec.kind]}"
            for spec in OUTPUT_SPECS
        )
        return (
            f"{TOOL_NAME} (version {TOOL_VERSION})\n\n"
            "PURPOSE\n"
            "Builds an ensemble of representative temporal storm patterns "
            "(Huff/ARR-style) for a fixed set of design durations and AEP "
            "bands, for use as design temporal patterns in hydrological "
            "modelling.\n\n"
            "METHOD\n"
            "1. Independent storm events are identified in the input "
            "rainfall time series using an Inter-Event Time Definition "
            "(IETD) and a minimum event depth threshold.\n"
            "2. For each of 24 target durations, the highest-depth "
            "sub-window near each event's peak is extracted.\n"
            "3. Each extracted storm is assigned an Annual Exceedance "
            "Probability (AEP) by inverting the supplied DDF table at "
            "that duration (log-duration x ln(ARI) interpolation, AEP = "
            "1-exp(-1/ARI)); storms whose depth falls outside the DDF "
            "table's covered range are assigned an AEP empirically "
            "instead, by rank within the sample (Weibull plotting "
            "position). Storms are then binned Frequent (>=20% AEP) / "
            "Intermediate (5-20% AEP) / Rare (<5% AEP).\n"
            "4. Each storm is also classified Early/Middle/Late by which "
            "third of its duration contains the peak burst.\n"
            "5. For each Duration x AEP x Shape bin where the required "
            "output time step is no finer than the input series' native "
            "time step, a representative pattern is built directly from "
            "data: storms are normalized to 0-100% time / 0-100% depth "
            "and the median curve across the bin's storms is taken "
            "(Huff's original method). If a bin has too few storms, and "
            "pooling is enabled, storms with the same AEP band and Shape "
            "from the nearest neighbouring durations are added in until "
            "the minimum sample size is reached (flagged "
            "direct_from_data_pooled in the bin summary) or the whole "
            "duration range is exhausted (flagged "
            "insufficient_data_even_pooled, left blank).\n"
            "6. For bins whose required time step is FINER than the "
            "input series' native time step (unavoidable for short "
            "durations unless the input is itself very fine), a Monte "
            "Carlo multiplicative-cascade disaggregation is used "
            "instead: the coarsest directly-built pattern for that "
            "AEP/Shape combination is cropped to the relevant sub-window "
            "and split down to the target time step using randomly "
            "sampled Dirichlet weights, run over many simulations and "
            "summarized by their median. Because no observed data exists "
            "at that finer resolution, this sub-step's variability is a "
            "generic statistical assumption, not one calibrated to this "
            "catchment - treat disaggregated durations as indicative "
            "rather than observed, and see the bin-summary output to "
            "know which rows were built which way.\n\n"
            "BACKGROUND\n"
            "Methodology follows Huff (1967, ISWS Circular 173) for the "
            "event separation, quartile/third classification, and "
            "median-curve aggregation approach, and Australian Rainfall "
            "& Runoff (Project 3, Book 1 s2.2.5) for the AEP/ARI "
            "relationship and the Early/Middle/Late (front/middle/back) "
            "loading classification. This is a single-representative-"
            "pattern-per-bin simplification of ARR's newer "
            "full-ensemble approach.\n\n"
            "PARAMETERS\n"
            "Input rainfall time series: CSV with a datetime column and "
            "an INCREMENTAL (not cumulative) rainfall depth column, any "
            "sub-daily interval.\n"
            "Input DDF table: CSV with a duration column (e.g. '3 h') "
            "and one or more '<ARI>yr Depth (mm)' columns. Where the "
            "table gives both a continuous '24 h' and a fixed-interval "
            "'1 day' value (Design Rainfall tables), the 24 h is used and "
            "the 1 day dropped.\n"
            "Datetime/value column names: leave blank to auto-detect.\n"
            "IETD: minimum dry-period hours separating independent "
            "storms (default 6h, per Huff 1967).\n"
            "Minimum event depth: events below this total depth (mm) "
            "are discarded as noise (default 2mm).\n"
            "Minimum sample size: bins with fewer storms than this are "
            "left blank in the output rather than built from a thin "
            "sample (default 5).\n"
            "Monte Carlo simulations / random seed / Dirichlet alpha: "
            "control the sub-native-resolution disaggregation (default "
            "500 sims, seed 42 for reproducibility, alpha 3.0 - lower "
            "alpha gives peakier/more variable synthesized "
            "sub-structure).\n"
            "Project number: optional prefix for the output file names.\n\n"
            "OUTPUTS\n"
            "Tick the outputs wanted (all three by default) and choose the "
            "CSV output folder. Files are named <project number>_<description> "
            "with underscores (just <description> without a project number) "
            "and replace files of the same name; a file that is open in Excel "
            "or QGIS cannot be replaced - close it and run again (the other "
            "outputs are still written). The CSVs are UTF-8 with a byte-order "
            "mark, so Excel opens them cleanly, and each starts with a column "
            "key ('#' lines) explaining the headings. The ensemble CSV can be "
            "read directly by Prepare HEC-HMS project and Prepare SWMM "
            "project.\n" + files
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_TIMESERIES, "Rainfall time series CSV", extension="csv"
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(self.INPUT_DDF, "DDF table CSV", extension="csv")
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DATETIME_COLUMN,
                "Datetime column name (blank = auto-detect)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.VALUE_COLUMN,
                "Rainfall depth column name (blank = auto-detect)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IETD_HOURS,
                "Inter-event time definition (hours)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_IETD_HOURS,
                minValue=0.1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_EVENT_DEPTH,
                "Minimum event depth (mm)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_MIN_EVENT_DEPTH_MM,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_SAMPLE_SIZE,
                "Minimum storms per bin",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=core.DEFAULT_MIN_SAMPLE_SIZE,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.POOL_SPARSE_BINS,
                "Pool sparse bins from neighbouring durations (same AEP/Shape) "
                "before giving up",
                defaultValue=core.DEFAULT_POOL_SPARSE_BINS,
            )
        )

        mc_sims_param = QgsProcessingParameterNumber(
            self.MC_SIMS,
            "Monte Carlo simulations per disaggregated bin",
            type=QgsProcessingParameterNumber.Type.Integer,
            defaultValue=core.DEFAULT_MC_SIMS,
            minValue=10,
        )
        mc_sims_param.setFlags(
            mc_sims_param.flags() | QgsProcessingParameterNumber.Flag.FlagAdvanced
        )
        self.addParameter(mc_sims_param)

        mc_alpha_param = QgsProcessingParameterNumber(
            self.MC_ALPHA,
            "Dirichlet cascade alpha (lower = peakier)",
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=core.DEFAULT_MC_ALPHA,
            minValue=0.1,
        )
        mc_alpha_param.setFlags(
            mc_alpha_param.flags() | QgsProcessingParameterNumber.Flag.FlagAdvanced
        )
        self.addParameter(mc_alpha_param)

        seed_param = QgsProcessingParameterNumber(
            self.RANDOM_SEED,
            "Random seed",
            type=QgsProcessingParameterNumber.Type.Integer,
            defaultValue=core.DEFAULT_RANDOM_SEED,
        )
        seed_param.setFlags(
            seed_param.flags() | QgsProcessingParameterNumber.Flag.FlagAdvanced
        )
        self.addParameter(seed_param)

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

    def processAlgorithm(self, parameters, context, feedback):
        ts_path = self.parameterAsFile(parameters, self.INPUT_TIMESERIES, context)
        ddf_path = self.parameterAsFile(parameters, self.INPUT_DDF, context)
        datetime_col = (
            self.parameterAsString(parameters, self.DATETIME_COLUMN, context) or None
        )
        value_col = (
            self.parameterAsString(parameters, self.VALUE_COLUMN, context) or None
        )
        ietd_hours = self.parameterAsDouble(parameters, self.IETD_HOURS, context)
        min_event_depth = self.parameterAsDouble(
            parameters, self.MIN_EVENT_DEPTH, context
        )
        min_sample_size = self.parameterAsInt(parameters, self.MIN_SAMPLE_SIZE, context)
        pool_sparse_bins = self.parameterAsBoolean(
            parameters, self.POOL_SPARSE_BINS, context
        )
        mc_sims = self.parameterAsInt(parameters, self.MC_SIMS, context)
        mc_alpha = self.parameterAsDouble(parameters, self.MC_ALPHA, context)
        random_seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
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

        feedback.pushInfo(
            "Loading time series and DDF table, extracting storm events..."
        )

        engine = core.DesignStormEngine(
            ietd_hours=ietd_hours,
            min_event_depth_mm=min_event_depth,
            min_sample_size=min_sample_size,
            pool_sparse_bins=pool_sparse_bins,
            mc_sims=mc_sims,
            mc_alpha=mc_alpha,
            random_seed=random_seed,
        )
        result = engine.run(
            ts_path, ddf_path, datetime_col=datetime_col, value_col=value_col
        )

        feedback.pushInfo(
            f"Native resolution: {result['native_minutes']:.0f} min. "
            f"{result['n_events']} independent storm events identified."
        )
        for w in result["warnings"]:
            feedback.pushWarning(w)

        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        writers = {
            self.OUTPUT_ENSEMBLE_CSV: (
                report.write_ensemble_csv,
                result["ensemble_rows"],
            ),
            self.OUTPUT_CATALOGUE_CSV: (
                report.write_catalogue_csv,
                result["catalogue_rows"],
            ),
            self.OUTPUT_BIN_SUMMARY_CSV: (
                report.write_bin_summary_csv,
                result["bin_summary_rows"],
            ),
        }
        written = []
        for key, (writer, rows) in writers.items():
            if key not in paths:
                continue
            path = paths[key]
            try:
                writer(rows, path)
            except Exception as e:
                feedback.reportError(
                    f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be "
                    f"written ({e}). If it is open in QGIS or Excel, close it and "
                    "run again.",
                    fatalError=False,
                )
                continue
            outputs[key] = path
            written.append(path)
            feedback.pushInfo(f"Written: {path}")
        if not written:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )

        n_blank = sum(1 for r in result["ensemble_rows"] if not r["Increments"])
        if n_blank:
            feedback.pushWarning(
                f"{n_blank} of {len(result['ensemble_rows'])} Duration x AEP x Shape "
                "bins had too few storms and were left blank - see the bin "
                "summary CSV."
            )

        return outputs
