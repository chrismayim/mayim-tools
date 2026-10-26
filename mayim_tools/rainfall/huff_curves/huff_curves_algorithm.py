"""Processing Toolbox algorithm: CSV rainfall record in -> Huff curve
outputs (event inventory, normalized event curves, percentile curve
bundles, metadata).

Thin wrapper - all real logic lives in huffrain/ (zero QGIS dependency,
independently testable outside QGIS; see tests/test_huffrain.py).

Outputs follow the Mayim output standard: a tick box per output, one
CSV folder, files named <project number>_<description>.csv.
"""

import os
from pathlib import Path

import pandas as pd
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
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

from .huffrain.export import (
    write_event_curves,
    write_event_inventory,
    write_huff_curves,
    write_metadata,
)
from .huffrain.io import build_huff_curves

TOOL_NAME = "Precipitation data to Huff Curves"
TOOL_VERSION = "1.1.0"

# Selectable outputs: key, dialog label, file-name description, folder
# kind, ticked by default. All share the CSV folder.
OUTPUT_SPECS = [
    OutputSpec("OUTPUT_HUFF", "Percentile Huff curves (CSV)", "Huff Curves", CSV),
    OutputSpec(
        "OUTPUT_EVENTS", "Event inventory (CSV)", "Huff Event Inventory", CSV,
    ),
    OutputSpec(
        "OUTPUT_METADATA", "Metadata and diagnostics (CSV)", "Huff Curves Metadata",
        CSV,
    ),
    OutputSpec(
        "OUTPUT_CURVES", "Normalised event curves (CSV; long, one row per event "
        "per time point)", "Huff Event Curves", CSV, False,
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {CSV: ("FOLDER_CSV", "Output folder for CSV files")}
WRITERS = {
    "OUTPUT_HUFF": write_huff_curves,
    "OUTPUT_EVENTS": write_event_inventory,
    "OUTPUT_METADATA": write_metadata,
    "OUTPUT_CURVES": write_event_curves,
}


class HuffCurvesAlgorithm(QgsProcessingAlgorithm):

    INPUT_CSV = "INPUT_CSV"
    TIMESTAMP_COL = "TIMESTAMP_COL"
    DEPTH_COL = "DEPTH_COL"
    TIMESTAMP_FORMAT = "TIMESTAMP_FORMAT"
    TIMESTAMP_SEMANTICS = "TIMESTAMP_SEMANTICS"
    MIT_HOURS = "MIT_HOURS"
    WET_THRESHOLD_MM = "WET_THRESHOLD_MM"
    EVENT_DEPTH_THRESHOLD_MM = "EVENT_DEPTH_THRESHOLD_MM"
    MIN_DURATION_HOURS = "MIN_DURATION_HOURS"
    QUALITY_MODE = "QUALITY_MODE"
    NORMALIZED_STEP = "NORMALIZED_STEP"
    MIN_SAMPLE = "MIN_SAMPLE"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    OUTPUT_EVENTS = "OUTPUT_EVENTS"
    OUTPUT_CURVES = "OUTPUT_CURVES"
    OUTPUT_HUFF = "OUTPUT_HUFF"
    OUTPUT_METADATA = "OUTPUT_METADATA"

    _SEMANTICS_OPTIONS = ["interval_end", "interval_start"]
    _QUALITY_OPTIONS = ["strict", "lenient"]

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return HuffCurvesAlgorithm()

    def name(self):
        return "huff_curves"

    def displayName(self):
        return "Precipitation data to Huff Curves"

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
            "Derives Huff curves from a CSV rainfall time series (timestamp + "
            "precipitation depth columns). Delineates storm events using a "
            "minimum inter-event time (MIT - recommended automatically if left "
            "blank, with the sensitivity scan recorded in the metadata output), "
            "classifies each event into a quartile (1-4, by which quarter of "
            "the event contains its single largest reading), and produces "
            "10th/25th/50th/75th/90th percentile curves per quartile using "
            "Hyndman-Fan type 8 empirical quantiles. Missing rainfall is never "
            "treated as zero - events with interior missing data are excluded "
            "from the curves but kept in the event inventory for auditability. "
            "This is a scoped v1: bootstrap uncertainty bands, stratification "
            "by duration/season, regional pooling, and design hyetograph "
            "generation are not yet implemented.\n\n"
            "OUTPUTS\n"
            "Tick the outputs wanted (Huff curves, event inventory and metadata "
            "by default; the long normalised event curves table is optional) "
            "and choose the CSV output folder. Files are named <project "
            "number>_<description> with underscores (just <description> "
            "without a project number) and replace files of the same name; a "
            "file that is open in Excel or QGIS cannot be replaced - close it "
            "and run again (the other outputs are still written). The CSVs are "
            "UTF-8 with a byte-order mark, so Excel opens them cleanly; the Huff "
            "curves, event inventory and event curves CSVs start with a column "
            "key ('#' lines) explaining the headings. The Huff curves CSV can "
            "be read directly by Prepare HEC-HMS project and Prepare SWMM "
            "project.\n" + files
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(self.INPUT_CSV, "Rainfall CSV", extension="csv")
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.TIMESTAMP_COL, "Timestamp column name", defaultValue="Date/Time"
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DEPTH_COL,
                "Precipitation depth column name",
                defaultValue="Precipitation",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.TIMESTAMP_FORMAT,
                "Timestamp format (leave blank to auto-detect, e.g. %Y-%m-%d %H:%M:%S)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.TIMESTAMP_SEMANTICS,
                "Timestamp represents",
                options=[
                    "Interval end (value is rainfall from previous timestamp to this one)",
                    "Interval start (value is rainfall from this timestamp to the next one)",
                ],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIT_HOURS,
                "Minimum inter-event time, hours (leave blank to auto-recommend)",
                type=QgsProcessingParameterNumber.Type.Double,
                optional=True,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WET_THRESHOLD_MM,
                "Wet-interval threshold, mm",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.EVENT_DEPTH_THRESHOLD_MM,
                "Minimum event total depth to retain, mm (optional)",
                type=QgsProcessingParameterNumber.Type.Double,
                optional=True,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_DURATION_HOURS,
                "Minimum event duration to retain, hours (optional)",
                type=QgsProcessingParameterNumber.Type.Double,
                optional=True,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.QUALITY_MODE,
                "Quality mode",
                options=[
                    "Strict (exclude events with missing interior data)",
                    "Lenient (keep them in the inventory, still excluded from curves)",
                ],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NORMALIZED_STEP,
                "Normalized time grid step (e.g. 0.05 = 21 points from 0 to 1)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.05,
                minValue=0.01,
                maxValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_SAMPLE,
                "Minimum events per quartile before flagging insufficient sample",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=5,
                minValue=1,
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

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        csv_path = self.parameterAsFile(parameters, self.INPUT_CSV, context)
        timestamp_col = self.parameterAsString(parameters, self.TIMESTAMP_COL, context)
        depth_col = self.parameterAsString(parameters, self.DEPTH_COL, context)
        timestamp_format = (
            self.parameterAsString(parameters, self.TIMESTAMP_FORMAT, context) or None
        )
        semantics = self._SEMANTICS_OPTIONS[
            self.parameterAsEnum(parameters, self.TIMESTAMP_SEMANTICS, context)
        ]
        mit_hours = (
            self.parameterAsDouble(parameters, self.MIT_HOURS, context)
            if parameters.get(self.MIT_HOURS) is not None
            else None
        )
        wet_threshold = self.parameterAsDouble(
            parameters, self.WET_THRESHOLD_MM, context
        )
        event_depth_threshold = (
            self.parameterAsDouble(parameters, self.EVENT_DEPTH_THRESHOLD_MM, context)
            if parameters.get(self.EVENT_DEPTH_THRESHOLD_MM) is not None
            else None
        )
        min_duration_hours = (
            self.parameterAsDouble(parameters, self.MIN_DURATION_HOURS, context)
            if parameters.get(self.MIN_DURATION_HOURS) is not None
            else None
        )
        quality_mode = self._QUALITY_OPTIONS[
            self.parameterAsEnum(parameters, self.QUALITY_MODE, context)
        ]
        normalized_step = self.parameterAsDouble(
            parameters, self.NORMALIZED_STEP, context
        )
        min_sample = self.parameterAsInt(parameters, self.MIN_SAMPLE, context)
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

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise QgsProcessingException(f"Could not read CSV: {e}") from e

        feedback.pushInfo(f"Loaded {len(df)} rows from {csv_path}")
        feedback.pushInfo(f"Columns found: {list(df.columns)}")

        try:
            result = build_huff_curves(
                df,
                timestamp_col=timestamp_col,
                depth_col=depth_col,
                timestamp_format=timestamp_format,
                timestamp_semantics=semantics,
                mit_hours=mit_hours,
                wet_threshold_mm=wet_threshold,
                event_depth_threshold_mm=event_depth_threshold,
                min_duration_hours=min_duration_hours,
                quality_mode=quality_mode,
                normalized_step=normalized_step,
                min_sample=min_sample,
            )
        except ValueError as e:
            raise QgsProcessingException(str(e)) from e

        for w in result.warnings:
            feedback.pushWarning(w)

        feedback.pushInfo(
            f"{result.diagnostics['n_events_raw']} events delineated, "
            f"{result.diagnostics['n_events_retained']} retained, "
            f"{result.diagnostics['n_curves']} contributed to curve sets."
        )
        for cs in result.curve_sets:
            feedback.pushInfo(
                f"  Quartile {cs.quartile}: {cs.n_events} events"
                f"{' (INSUFFICIENT SAMPLE)' if cs.insufficient_sample else ''}"
            )

        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []
        for key, writer in WRITERS.items():
            if key not in paths:
                continue
            path = paths[key]
            try:
                writer(result, path)
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

        return outputs
