"""Processing Toolbox algorithm: DDF/IDF table in -> three design
hyetograph CSVs out (incremental depth, cumulative depth, incremental
intensity), via the Generalized Alternating Block Method.

Outputs follow the Mayim report and figure standard: the user ticks the
outputs wanted and chooses the CSV folder; files are named <project
number>_<description> with underscores.

Thin wrapper - all real logic lives in core.py/export.py (zero QGIS
dependency, independently testable; see tests/test_core.py).
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

from .core import PEAK_RATIOS, parse_idf_table
from .export import (
    write_cumulative_depth,
    write_incremental_depth,
    write_incremental_intensity,
)

TOOL_VERSION = "0.4.0"

# Selectable outputs: key, dialog label, file-name description, folder
# kind, ticked by default. All three share the CSV folder.
OUTPUT_SPECS = [
    OutputSpec(
        "OUTPUT_INCREMENTAL_DEPTH", "Incremental depth vs time (CSV)",
        "Incremental Depth", CSV,
    ),
    OutputSpec(
        "OUTPUT_CUMULATIVE_DEPTH", "Cumulative depth vs time (CSV)",
        "Cumulative Depth", CSV,
    ),
    OutputSpec(
        "OUTPUT_INCREMENTAL_INTENSITY", "Incremental intensity vs time (CSV)",
        "Incremental Intensity", CSV,
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {CSV: ("FOLDER_CSV", "Output folder for CSV files")}

WRITERS = {
    "OUTPUT_INCREMENTAL_DEPTH": write_incremental_depth,
    "OUTPUT_CUMULATIVE_DEPTH": write_cumulative_depth,
    "OUTPUT_INCREMENTAL_INTENSITY": write_incremental_intensity,
}


class DdfToHyetographsAlgorithm(QgsProcessingAlgorithm):

    INPUT_CSV = "INPUT_CSV"
    DURATION_COL = "DURATION_COL"
    TARGET_DURATIONS = "TARGET_DURATIONS"
    TIMESTEP_MIN = "TIMESTEP_MIN"
    ALLOW_SHORT_DURATION_EXTRAPOLATION = "ALLOW_SHORT_DURATION_EXTRAPOLATION"
    PROJECT_NUMBER = "PROJECT_NUMBER"

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return DdfToHyetographsAlgorithm()

    def name(self):
        return "ddf_to_hyetographs"

    def displayName(self):
        return "DDF to Alternating Block Design Hyetographs"

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
            f"DDF to Alternating Block Design Hyetographs (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tConverts a DDF/IDF table (Duration x Return Period "
            "depths) into design storm hyetographs at 5 peak-position "
            "ratios (25/33/50/66/75% of storm duration before the peak), "
            "for one or more target storm durations, across every return "
            "period in the table. Three outputs: incremental depth (the "
            "original hyetograph), cumulative depth, and incremental "
            "intensity - all versus time, all sharing the same row/column "
            "layout so they're directly comparable.\n"
            "\n"
            "METHOD:\t\tFor each return period, interpolates a continuous "
            "cumulative-depth-vs-duration curve from the input table "
            "(piecewise log-log, exact at every tabulated point, no "
            "extrapolation beyond the table's own duration range). Divides "
            "each target storm duration into equal-width timesteps and "
            "computes each step's incremental depth from that curve. "
            "Reorders the incremental blocks via the Generalized "
            "Alternating Block Method: sorts descending, places the "
            "largest block at the peak position, then alternates placing "
            "the next-largest blocks immediately before and after it, "
            "working outward - pure reordering, total depth is conserved "
            "by construction, never altered. This is NOT the classical "
            "SCS/NRCS Type I/IA/II/III design storm distributions, despite "
            "this plugin's original 'SCS Design Storm' working name - "
            "renamed specifically to avoid that confusion.\n"
            "\n"
            "BACKGROUND:\tInput accepts either the '{RT}yr Depth (mm)' "
            "header format (matching the Design Rainfall and Rainfall "
            "Frequency Analysis plugins' recommended-DDF output - feeds in "
            "with zero reformatting) or bare AEP-percent column headers "
            "(e.g. '50%', '1%'). Duration values accept a bare number "
            "(assumed minutes) or an explicit unit ('5 min', '1 h'). A "
            "target duration shorter than the table's shortest tabulated "
            "duration, or longer than its longest, will fail with a clear "
            "error rather than extrapolate silently. SHORT-DURATION "
            "EXTRAPOLATION: an hourly-only source (e.g. an ERA5-derived "
            "DDF table) has no tabulated data below 60 min, but a finer "
            "timestep still needs cumulative-depth values within that "
            "first hour - enable the option below to extend the curve "
            "toward zero using the first tabulated segment's own power-law "
            "slope (the same log-log form already used between every other "
            "pair of tabulated points, just continued down to zero rather "
            "than an externally-imposed coefficient). This may "
            "UNDERESTIMATE short-duration intensity if the true DDF curve "
            "steepens at short durations, which is common - real sub-"
            "hourly convective bursts are often more intense, relative to "
            "duration, than an hourly curve's own slope suggests. Prefer "
            "actual sub-hourly data or a recognized short-duration formula "
            "when available; this is a fallback for when neither exists, "
            "not a general substitute for one. Never used above the "
            "table's longest tabulated duration - there's no equivalent "
            "'curve continues toward infinity' assumption that makes "
            "sense the way 'curve continues toward zero at zero duration' "
            "does.\n"
            "\n"
            "PARAMETERS:\n"
            "  Duration column: leave blank to auto-detect ('Duration', "
            "'Time', or similar).\n"
            "  Target storm duration(s): comma-separated minutes (e.g. "
            "'60,120,1440' for 1h/2h/24h design storms) - one hyetograph "
            "set is generated per (return period, target duration) "
            "combination across every return period found in the table.\n"
            "  Timestep: the hyetograph's block width in minutes.\n"
            "  Allow short-duration extrapolation: off by default - see "
            "BACKGROUND above.\n"
            "\n"
            "OUTPUTS:\tTick the outputs wanted (all three by default) and "
            "choose the CSV output folder. Files are named <project "
            "number>_<description> with underscores (just <description> "
            "without a project number) and replace files of the same name; "
            "a file that is open in Excel or QGIS cannot be replaced - "
            "close it and run again (the other outputs are still written). "
            "Written as UTF-8 with a byte-order mark, so Excel opens them "
            "cleanly.\n" + files
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_CSV, "DDF/IDF table CSV", extension="csv"
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DURATION_COL,
                "Duration column name (blank = auto-detect)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.TARGET_DURATIONS,
                "Target storm duration(s) in minutes, comma-separated",
                defaultValue="60",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TIMESTEP_MIN,
                "Timestep (minutes)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=5.0,
                minValue=0.1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ALLOW_SHORT_DURATION_EXTRAPOLATION,
                "Allow short-duration extrapolation\n"
                "(for timesteps finer than the table's\n"
                "shortest duration - see description)",
                defaultValue=False,
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
        duration_col = (
            self.parameterAsString(parameters, self.DURATION_COL, context) or None
        )
        target_durations_str = self.parameterAsString(
            parameters, self.TARGET_DURATIONS, context
        )
        timestep_min = self.parameterAsDouble(parameters, self.TIMESTEP_MIN, context)
        allow_short_duration_extrapolation = self.parameterAsBoolean(
            parameters, self.ALLOW_SHORT_DURATION_EXTRAPOLATION, context
        )
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
            target_durations = [
                float(x.strip()) for x in target_durations_str.split(",") if x.strip()
            ]
        except ValueError:
            raise QgsProcessingException(
                f"Could not parse target durations: {target_durations_str!r}"
            ) from None
        if not target_durations:
            raise QgsProcessingException(
                "At least one target storm duration is required."
            )

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise QgsProcessingException(f"Could not read CSV: {e}") from e
        feedback.pushInfo(
            f"Loaded {len(df)} rows from {csv_path}. Columns: {list(df.columns)}"
        )

        try:
            tables = parse_idf_table(df, duration_col=duration_col)
        except ValueError as e:
            raise QgsProcessingException(str(e)) from e
        feedback.pushInfo(f"Return periods found: {sorted(tables.keys())}")
        feedback.pushInfo(
            f"Target duration(s): {target_durations} min. Timestep: {timestep_min} min. "
            f"Peak ratios: {PEAK_RATIOS}"
        )

        return_periods_and_durations = [
            (rt, dur) for rt in sorted(tables.keys()) for dur in target_durations
        ]

        for rt, dur in return_periods_and_durations:
            table_min = tables[rt][0][0]
            table_max = tables[rt][-1][0]
            if dur < table_min or dur > table_max:
                raise QgsProcessingException(
                    f"Target duration {dur:g} min is outside the {rt:g}yr table's range "
                    f"[{table_min:g}, {table_max:g}] min - no extrapolation. Adjust the target "
                    f"duration or supply a table that covers it."
                )
            if timestep_min < table_min and not allow_short_duration_extrapolation:
                raise QgsProcessingException(
                    f"The {timestep_min:g}-min timestep needs cumulative-depth values below the "
                    f"{rt:g}yr table's shortest tabulated duration ({table_min:g} min) - e.g. an "
                    f"hourly-only DDF table (such as one derived from ERA5) with a finer timestep. "
                    f"Enable 'Allow short-duration extrapolation' to extend the curve toward zero "
                    f"using the first tabulated segment's own power-law slope, or use a timestep no "
                    f"finer than {table_min:g} min."
                )
        if (
            timestep_min < min(tables[rt][0][0] for rt in tables)
            and allow_short_duration_extrapolation
        ):
            feedback.pushWarning(
                f"Short-duration extrapolation is active: at least one return period's table has no "
                f"tabulated data below {min(tables[rt][0][0] for rt in tables):g} min, but the "
                f"{timestep_min:g}-min timestep needs values there. Extrapolated using the first "
                f"tabulated segment's own power-law slope - this may UNDERESTIMATE short-duration "
                f"intensity if the true DDF curve steepens at short durations (a common, real "
                f"behaviour), particularly for an hourly-only source like ERA5. Use actual sub-hourly "
                f"data or a recognized short-duration formula instead if available."
            )

        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []  # (label, path) for the log

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        def failed(key, path, error):
            feedback.reportError(
                f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be written "
                f"({error}). If it is open in QGIS or Excel, close it and run "
                "again.",
                fatalError=False,
            )

        n_rows = len(return_periods_and_durations)
        for key, writer in WRITERS.items():
            if key not in want:
                continue
            path = paths[key]
            try:
                writer(
                    return_periods_and_durations,
                    timestep_min,
                    tables,
                    path,
                    allow_short_duration_extrapolation,
                )
            except ValueError as e:
                raise QgsProcessingException(str(e)) from e
            except Exception as e:
                failed(key, path, e)
                continue
            done(key, path)

        if not written:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )
        feedback.pushInfo(
            f"{n_rows} row(s) written to each of {len(written)} output(s) "
            f"({len(tables)} return period(s) x {len(target_durations)} target duration(s))."
        )
        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")

        return outputs
