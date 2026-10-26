"""AdjustToDdfAlgorithm - the Processing algorithm.

Thin wrapper - all real logic lives in core.py / adjust.py / export.py /
report.py and the shared mayim_tools.rainfall._common package (zero QGIS
dependency, independently testable; see tests/test_storm_library_core.py).
QGIS4/Qt6-safe: only File/Number/String/Boolean/Enum/FileDestination
Processing parameters are used.

(The package keeps its original name, storm_library, because the storm
library is still one of its outputs.)
"""

import os
from pathlib import Path

import pandas as pd
from qgis.core import (
    Qgis,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import CSV, WORD, OutputSpec, clean_name_part
from mayim_tools.rainfall._common.ddf import DDFTable, read_ddf_csv

from . import core, export

# Selectable outputs (Mayim output standard): key, dialog label, file-name
# description, folder kind, ticked by default. The chart PNG goes to the
# Word folder, beside the report it belongs to.
OUTPUT_SPECS = [
    OutputSpec("OUTPUT_ADJUSTED", "Adjusted rainfall series (CSV)", "Adjusted Rainfall Series", CSV),
    OutputSpec("OUTPUT_REPORT", "Report (Word)", "DDF Adjustment Report", WORD),
    OutputSpec("OUTPUT_CONSISTENCY", "Before/after DDF consistency (CSV)", "DDF Consistency", CSV),
    OutputSpec("OUTPUT_METADATA", "Metadata (CSV)", "DDF Adjustment Metadata", CSV),
    OutputSpec("OUTPUT_ENSEMBLE", "Ensemble series (CSV; needs realisations > 0)", "Adjusted Rainfall Ensemble", CSV, False),
    OutputSpec("OUTPUT_LIBRARY", "Storm library (CSV)", "Storm Library", CSV, False),
    OutputSpec("OUTPUT_EVENTS", "Storm audit (CSV)", "Storm Audit", CSV, False),
    OutputSpec("OUTPUT_PNG", "Before/after chart (PNG)", "DDF Adjustment Chart", WORD, False),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports (and the chart PNG)"),
}


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


def output_path(spec, folders, project_number, site=None):
    """Standard file name <project number>_<description>[_<site>].<ext>;
    the chart is a .png in the Word folder."""
    desc = spec.description + (f" {clean_name_part(site)}" if site else "")
    path = OutputSpec(spec.key, spec.label, desc, spec.kind).path(
        folders, project_number
    )
    if spec.key == "OUTPUT_PNG":
        path = str(Path(path).with_suffix(".png"))
    return path


class AdjustToDdfAlgorithm(QgsProcessingAlgorithm):
    INPUT_SERIES = "INPUT_SERIES"
    SITE = "SITE"
    TIME_COLUMN = "TIME_COLUMN"
    DEPTH_COLUMN = "DEPTH_COLUMN"
    TIMEZONE_OFFSET_H = "TIMEZONE_OFFSET_H"
    INPUT_DDF = "INPUT_DDF"
    DDF_SITE = "DDF_SITE"
    MAP_SOURCE = "MAP_SOURCE"
    MAP_VALUE = "MAP_VALUE"
    REFERENCE_TIER = "REFERENCE_TIER"
    REFERENCE_IS_AREAL = "REFERENCE_IS_AREAL"
    ANCHOR_DURATION = "ANCHOR_DURATION"
    TEST_DURATIONS = "TEST_DURATIONS"
    CALIBRATION_DURATIONS = "CALIBRATION_DURATIONS"
    RARITY_DEPENDENT = "RARITY_DEPENDENT"
    N_REALISATIONS = "N_REALISATIONS"
    ENSEMBLE_SIGMA = "ENSEMBLE_SIGMA"
    IETD_HOURS = "IETD_HOURS"
    MIN_EVENT_DEPTH = "MIN_EVENT_DEPTH"
    MIN_COMPLETENESS = "MIN_COMPLETENESS"
    FIXED_INTERVAL_CORRECTION = "FIXED_INTERVAL_CORRECTION"
    DISTRIBUTION = "DISTRIBUTION"
    TOLERANCE_PCT = "TOLERANCE_PCT"
    N_BOOTSTRAP = "N_BOOTSTRAP"
    RANDOM_SEED = "RANDOM_SEED"
    OUTPUT_ADJUSTED = "OUTPUT_ADJUSTED"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_ENSEMBLE = "OUTPUT_ENSEMBLE"
    OUTPUT_CONSISTENCY = "OUTPUT_CONSISTENCY"
    OUTPUT_LIBRARY = "OUTPUT_LIBRARY"
    OUTPUT_EVENTS = "OUTPUT_EVENTS"
    OUTPUT_METADATA = "OUTPUT_METADATA"
    OUTPUT_PNG = "OUTPUT_PNG"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

    _TIER_OPTIONS = [core.REFERENCE_TIERS[k] for k in (1, 2, 3)]
    _ANCHOR_OPTIONS = [
        ("12 h", 720.0),
        ("24 h", 1440.0),
        ("48 h", 2880.0),
        ("72 h", 4320.0),
    ]
    _DIST_OPTIONS = list(core.DISTRIBUTION_CHOICES)
    _MAP_OPTIONS = [
        "Value entered below",
        "From the reference DDF CSV (Design Rainfall 'MAP (mm)')",
        "Retain the input series' own MAP",
    ]

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return AdjustToDdfAlgorithm()

    def name(self):
        return "adjust_subdaily_to_ddf"

    def displayName(self):
        return "Adjust Sub-daily Rainfall to DDF"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            f"{core.TOOL_NAME} (version {core.TOOL_VERSION})\n\n"
            "PURPOSE\n"
            "Adjusts a sub-daily gridded rainfall series (ERA5, IMERG, CMORPH, "
            "PERSIANN-CCS...) so that its annual-maximum depths at every tested "
            "duration match a reference, gauge-based DDF table, while keeping the "
            "timing and sequence of the recorded storms. Outputs the adjusted series "
            "and a Word report documenting the method and the before/after fit.\n\n"
            "METHOD\n"
            "1. The series is regularised (missing data stays missing, never zero) "
            "and incomplete years are excluded.\n"
            "2. Pass 1 - magnitude and annual total: the product's own "
            "annual-maximum curve G is fitted at the anchor duration (default 24 h). "
            "Storms (IETD separation) in the annual-maximum range are scaled so "
            "their anchor-duration depth equals the reference DDF depth at the "
            "storm's own return period in G. Smaller storms and light rain follow a "
            "power curve f(x) = f(x0)(x/x0)^gamma joined continuously to it, with "
            "gamma solved so the adjusted series reproduces the target mean annual "
            "precipitation (MAP) exactly. gamma > 1 moves the annual total out of "
            "light rain into fewer, heavier storms - correcting the reanalysis "
            "tendency to spread rain as frequent drizzle with flattened peaks.\n"
            "3. Pass 2 - shape: working down a ladder of calibration durations (e.g. "
            "12 h -> 3 h -> 1 h), the peak window of each storm at that duration is "
            "multiplied by alpha = exp(a + c(lnT - ln10)) and the rest of its parent "
            "window reduced so the parent total is unchanged (mass-preserving, "
            "zeros stay zero, timing kept). (a, c) are fitted per rung so the "
            "adjusted series' implied DDF matches the reference.\n"
            "4. Durations between the rungs, and longer than the anchor, are not "
            "fitted - they are the independent validation. Before/after ratios "
            "F = implied / reference are reported with 5-95% year-bootstrap "
            "intervals.\n"
            "5. Optional stochastic ensemble: N realisations with lognormal "
            "variability (sigma) on each storm's peak concentration, re-fitted so "
            "the ensemble median matches the reference.\n"
            "6. Optional storm library (normalised patterns of the adjusted storms, "
            "Early/Middle/Late, AEP band, ARR-style embedded-burst flag) for Monte "
            "Carlo design storms.\n\n"
            "BACKGROUND\n"
            "Reanalysis and satellite products under-represent short-duration "
            "convective bursts through grid-scale averaging and parameterised "
            "convection, increasingly so for rarer events (e.g. Guerreiro et al. "
            "2024; Lavers et al. 2022). The gauge-based DDF is taken as the "
            "reference for magnitude at every duration; the gridded product "
            "contributes storm occurrence, timing and sequence. The adjusted series "
            "is a statistical reconstruction for design and continuous simulation, "
            "not a corrected historical record, and cannot create detail finer than "
            "the native time step. Fixed-interval maxima use the Weiss (1964) factor "
            "1/(1-1/(8n)); return periods are annual-maximum values (AEP = 1/T). "
            "Tier 3 (gridded-only) references are stamped INDICATIVE - NOT "
            "VALIDATED in every output. Supply an ARF-applied reference if the series "
            "must represent a catchment rather than a point.\n\n"
            "PARAMETERS\n"
            "Rainfall series CSV (sub-daily): incremental (not cumulative) depths "
            "at a fixed time step, e.g. straight from Extract: ERA5 / IMERG / "
            "CMORPH precipitation (Site, ValidTime, PrecipitationMM) or any "
            "time/depth CSV. Missing values are kept as missing, never zero.\n"
            "\n"
            "Site: only for multi-site CSVs - the label exactly as it appears in "
            "the Site column (e.g. BH01). One site per run. Blank = first site in "
            "the file (a warning is given if there are others). Ignored if there "
            "is no Site column.\n"
            "\n"
            "Time column / Depth column: blank = auto-detected "
            "(ValidTime/Time/Date..., PrecipitationMM/Depth/Rainfall...). Name them"
            " only if auto-detection fails.\n"
            "\n"
            "Time-zone offset (h): hours added to the timestamps before calendar "
            "years and months are assigned. Gridded products are in UTC: +2 for "
            "South Africa (SAST), +8 for Western Australia (AWST); 0 if the series "
            "is already in local time. Durations are sliding windows and are not "
            "affected; the adjusted series is written back in the input time base. "
            "Effect on results is usually small.\n"
            "\n"
            "Reference DDF CSV: Design Rainfall (South Africa) output (single- or "
            "multi-site), Precipitation data to DDF output, or any table with a "
            "'Duration' column and '<T>yr Depth (mm)' columns (T = annual-maximum "
            "return period, AEP = 1/T). Lower/Upper bound columns are carried "
            "separately; continuous '24 h' is preferred over fixed '1 day'; day-"
            "labelled durations are converted with the Weiss factor.\n"
            "\n"
            "Reference DDF site: for multi-site Design Rainfall CSVs - must be the "
            "same location as the series. Blank = first.\n"
            "\n"
            "Target MAP source / value (mm/yr): the mean annual precipitation the "
            "adjusted series must reproduce. 'Value entered below' (default) uses "
            "the number you type, e.g. from a gauge record for the same period as "
            "the series, or from a gauge-adjusted product such as CHIRPS over that "
            "period. 'From the reference DDF CSV' reads the 'MAP (mm)' value that "
            "Design Rainfall (South Africa) writes into its output (note: a "
            "long-term value from the gauge period behind the DDF). 'Retain the "
            "input series' own MAP' keeps the gridded product's total. Use a MAP "
            "for the same period as the series where possible; the report states "
            "the source. If the DDF-matched storms alone exceed the target MAP, the "
            "tool stops with a diagnostic - the DDF, MAP and series are then not "
            "consistent (check site, period and MAP source).\n"
            "\n"
            "Reference tier: Tier 1 = published gauge-based DDF (Design Rainfall "
            "SA, BoM 2016 IFD); Tier 2 = GSDR-IDF, literature IDF or gauge-derived "
            "DDF; Tier 3 = gridded-only reference (every output is then stamped "
            "INDICATIVE - NOT VALIDATED). Recorded in the report.\n"
            "\n"
            "Reference already areal: tick only if the DDF already has an areal "
            "reduction factor applied, i.e. the adjusted series is meant to "
            "represent a catchment rather than a point.\n"
            "\n"
            "Anchor duration (default 24 h): the duration at which pass 1 matches "
            "each storm's magnitude to the DDF (storm depth over the anchor -> its "
            "return period in the product's own record -> scaled to the DDF depth "
            "at that return period). Shorter durations are then corrected by pass 2"
            " without changing the anchor totals. 24 h is recommended: gridded "
            "products are most reliable near daily totals, it is the best-supported"
            " part of gauge DDFs, and most storms fit within it. Use 12 h for short"
            " convective storms with well-supported 12 h values; 48-72 h for large "
            "catchments or long frontal/tropical events. Choose a duration the DDF "
            "table actually contains (otherwise it is interpolated and a warning is"
            " given). The report's self-check should be within ~3%.\n"
            "\n"
            "Test durations (min): comma-separated durations to evaluate, e.g. "
            "60,120,180,360,720,1440,2880. Blank = every reference duration that is"
            " a whole multiple of the native step, up to 72 h.\n"
            "\n"
            "Calibration durations (min): durations pass 2 is fitted to, e.g. "
            "60,180,720. Blank = every other test duration below the anchor, "
            "starting from the shortest; the durations in between, and those longer"
            " than the anchor, are then independent validation. Keep at least one "
            "duration unfitted between rungs so the result can be validated.\n"
            "\n"
            "Peak concentration varies with return period (default on): lets rarer "
            "storms be sharpened more than frequent ones, which matches the known "
            "behaviour of reanalysis products. Switch off only for a simpler "
            "single-factor adjustment.\n"
            "\n"
            "Stochastic ensemble realisations (default 0): number of additional, "
            "equally likely adjusted series. Each matches the DDF on average but "
            "differs storm by storm in peak concentration. 0 = main adjusted series"
            " only (most work); 10-20 = enough to show an uncertainty band in the "
            "report; 50-100 = for Monte Carlo or HEC-RAS ensemble runs. Run time "
            "and ensemble CSV size grow roughly in proportion.\n"
            "\n"
            "Ensemble within-storm variability sigma (default 0.35): standard "
            "deviation of the mean-one lognormal factor applied to each storm's "
            "peak concentration in the ensemble (0 = no variation; 0.35 = typical "
            "+/-35% spread). A DDF table cannot constrain this, so it is a stated "
            "judgement: use 0.35, and for important work test 0.2, 0.35 and 0.5 and"
            " report the sensitivity. Only used when realisations > 0.\n"
            "\n"
            "Inter-event time definition, IETD (default 6 h): the dry gap that "
            "separates two storms; rain less than this apart belongs to the same "
            "storm. Shorter (3 h) splits bursts into separate storms, longer (12-24"
            " h) merges multi-burst events. 6 h suits summer convective rainfall "
            "(SA); 6-12 h suits frontal/winter rainfall (e.g. WA). Keep it well "
            "below the anchor duration; for important work check that 3 h and 12 h "
            "give similar after-ratios.\n"
            "\n"
            "Minimum storm depth (default 2 mm): rain events smaller than this are "
            "not treated as storms - they are only scaled by the constant "
            "background factor, not reshaped. 2 mm suits most sites; 3-5 mm for "
            "ERA5 (many drizzle hours) speeds up the run and removes noise. Has "
            "almost no effect on annual maxima.\n"
            "\n"
            "Minimum year completeness (default 0.90): fraction of a calendar year "
            "that must be present for that year to be used. 0.90 is conventional; "
            "lower it only for short records, and state it.\n"
            "\n"
            "Apply fixed-interval correction (default on): clock-hour or clock-day "
            "totals under-record the true maximum, which usually straddles two "
            "intervals; annual maxima are multiplied by the Weiss (1964) factor "
            "1/(1-1/(8n)) for an n-step window (about 1.14 for 1 step, 1.07 for 2, "
            "1.005 for 24). Leave ON when the reference is a published gauge DDF "
            "(Design Rainfall SA, BoM IFD - continuous-duration depths). Turn OFF "
            "when the reference was produced with Precipitation data to DDF with its "
            "fixed-interval correction switched off. Rule: use the same convention "
            "on both sides.\n"
            "\n"
            "Distribution: should match how the reference DDF was derived - GEV "
            "(L-moments) for Design Rainfall SA and ARR 2016 IFDs (default). The "
            "others are for references built differently.\n"
            "\n"
            "Tolerance (default 10%): how close the ratio F = implied / reference "
            "must be to 1 to be labelled 'consistent'. Affects only the verdict "
            "labels, not the adjustment. 10% is comparable with the uncertainty of "
            "published DDFs; 5% is a strict check. Design Rainfall lower/upper "
            "bounds are usually much wider.\n"
            "\n"
            "Bootstrap replicates (default 1000): number of whole-year resamples "
            "used for the 5-95% intervals. 100-200 for trial runs; 1000-2000 for "
            "final runs. Affects only the width and stability of the intervals and "
            "the run time, not the adjusted series.\n"
            "\n"
            "Random seed (default 42): fixes the random numbers of the bootstrap "
            "and ensemble so a rerun with the same inputs gives identical results "
            "(important for review). Leave it and quote it (it is recorded in the "
            "report); change it only to show that results are not an artefact of "
            "one random draw.\n"
            "\n"
            "Suggested settings: first trial - realisations 0, bootstrap 200, other"
            " defaults. Final - bootstrap 1000; realisations 20 at sigma 0.35 if an"
            " uncertainty band is wanted.\n"
            "\n"
            "Project number / project name (optional): the project number is the "
            "file-name prefix; both go on the report's title page.\n"
            "\n"
            "OUTPUTS\n"
            "Tick the outputs wanted; they are written to the CSV and Word folders "
            "as <project number>_<description>_<site>, with underscores: adjusted "
            "series CSV (Site, ValidTime, PrecipitationMM = adjusted, plus input "
            "depth and adjustment factors; usable directly in other Mayim Tools), "
            "Word report in the Mayim report standard (requires python-docx in "
            "QGIS's Python), before/after consistency CSV and metadata CSV (ticked "
            "by default); ensemble CSV (one column per realisation), storm library "
            "CSV (for Monte Carlo design storms), storm audit CSV and the "
            "before/after chart PNG (Word folder). CSVs are UTF-8 with BOM; the "
            "consistency, storm library and storm audit CSVs start with a column "
            "key ('#' lines). Existing files are overwritten."
        )

    def initAlgorithm(self, config=None):
        add = self.addParameter
        add(
            QgsProcessingParameterFile(
                self.INPUT_SERIES, "Rainfall series CSV (sub-daily)", extension="csv"
            )
        )
        add(
            QgsProcessingParameterString(
                self.SITE, "Site (multi-site CSVs; blank = first)", optional=True
            )
        )
        add(
            QgsProcessingParameterString(
                self.TIME_COLUMN, "Time column (blank = auto)", optional=True
            )
        )
        add(
            QgsProcessingParameterString(
                self.DEPTH_COLUMN, "Depth column (blank = auto)", optional=True
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.TIMEZONE_OFFSET_H,
                "Time-zone offset added to timestamps (h), e.g. 2 for UTC -> SAST",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=-14.0,
                maxValue=14.0,
            )
        )
        add(
            QgsProcessingParameterFile(
                self.INPUT_DDF, "Reference DDF CSV", extension="csv"
            )
        )
        add(
            QgsProcessingParameterString(
                self.DDF_SITE,
                "Reference DDF site (multi-site Design Rainfall CSVs; blank = first)",
                optional=True,
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.MAP_SOURCE,
                "Target MAP source",
                options=self._MAP_OPTIONS,
                defaultValue=0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MAP_VALUE,
                "Target MAP (mm/yr) - used when the source is 'Value entered below'",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.REFERENCE_TIER,
                "Reference tier",
                options=self._TIER_OPTIONS,
                defaultValue=0,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.REFERENCE_IS_AREAL,
                "Reference DDF already has an areal reduction factor applied",
                defaultValue=False,
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.ANCHOR_DURATION,
                "Anchor duration (pass-1 magnitude mapping)",
                options=[a[0] for a in self._ANCHOR_OPTIONS],
                defaultValue=1,
            )
        )
        add(
            QgsProcessingParameterString(
                self.TEST_DURATIONS,
                "Test durations in minutes, comma-separated (blank = automatic)",
                optional=True,
            )
        )
        add(
            QgsProcessingParameterString(
                self.CALIBRATION_DURATIONS,
                "Calibration durations in minutes, comma-separated (blank = automatic)",
                optional=True,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.RARITY_DEPENDENT,
                "Let peak concentration vary with return period",
                defaultValue=True,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.N_REALISATIONS,
                "Stochastic ensemble realisations (0 = deterministic only)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=0,
                minValue=0,
                maxValue=500,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.ENSEMBLE_SIGMA,
                "Ensemble within-storm variability sigma (lognormal)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.35,
                minValue=0.0,
                maxValue=2.0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.IETD_HOURS,
                "Inter-event time definition, IETD (h)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_IETD_HOURS,
                minValue=0.5,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MIN_EVENT_DEPTH,
                "Minimum storm depth (mm)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_MIN_EVENT_DEPTH_MM,
                minValue=0.0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MIN_COMPLETENESS,
                "Minimum year completeness (0-1)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_MIN_COMPLETENESS,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.FIXED_INTERVAL_CORRECTION,
                "Apply fixed-interval (Weiss 1964) correction to annual maxima",
                defaultValue=True,
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.DISTRIBUTION,
                "Distribution (match the reference DDF's method)",
                options=self._DIST_OPTIONS,
                defaultValue=0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.TOLERANCE_PCT,
                "Tolerance |F-1| treated as consistent (%)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_TOLERANCE_PCT,
                minValue=0.0,
                maxValue=100.0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.N_BOOTSTRAP,
                "Bootstrap replicates",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=core.DEFAULT_N_BOOTSTRAP,
                minValue=0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.RANDOM_SEED,
                "Random seed",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=core.DEFAULT_RANDOM_SEED,
            )
        )
        add(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix and report, optional)",
                optional=True,
            )
        )
        add(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project / site name (report title page, optional)",
                optional=True,
            )
        )
        for spec in OUTPUT_SPECS:
            add(
                QgsProcessingParameterBoolean(
                    spec.key, f"Output: {spec.label}", defaultValue=spec.default
                )
            )
        for name, label in FOLDER_PARAMS.values():
            add(QgsProcessingParameterFolderDestination(name, label, optional=True))

    def processAlgorithm(self, parameters, context, feedback):
        series_path = self.parameterAsFile(parameters, self.INPUT_SERIES, context)
        ddf_path = self.parameterAsFile(parameters, self.INPUT_DDF, context)

        def s(name):
            return self.parameterAsString(parameters, name, context).strip() or None

        def minutes_list(name):
            txt = s(name)
            if not txt:
                return None
            try:
                return tuple(float(x) for x in txt.split(",") if x.strip())
            except ValueError:
                raise QgsProcessingException(
                    f"Could not parse durations: {txt!r}"
                ) from None

        selected = [
            spec
            for spec in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, spec.key, context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {spec.kind for spec in selected}:
            fname, flabel = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, fname, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {flabel.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        want = {spec.key for spec in selected}
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        ).strip()
        project_name = self.parameterAsString(
            parameters, self.PROJECT_NAME, context
        ).strip()
        want_library = self.OUTPUT_LIBRARY in want
        cfg = core.StormLibraryConfig(
            anchor_min=self._ANCHOR_OPTIONS[
                self.parameterAsEnum(parameters, self.ANCHOR_DURATION, context)
            ][1],
            test_durations_min=minutes_list(self.TEST_DURATIONS),
            calibration_durations_min=minutes_list(self.CALIBRATION_DURATIONS),
            rarity_dependent_sharpening=self.parameterAsBoolean(
                parameters, self.RARITY_DEPENDENT, context
            ),
            n_realisations=self.parameterAsInt(
                parameters, self.N_REALISATIONS, context
            ),
            ensemble_sigma=self.parameterAsDouble(
                parameters, self.ENSEMBLE_SIGMA, context
            ),
            ietd_hours=self.parameterAsDouble(parameters, self.IETD_HOURS, context),
            min_event_depth_mm=self.parameterAsDouble(
                parameters, self.MIN_EVENT_DEPTH, context
            ),
            min_completeness=self.parameterAsDouble(
                parameters, self.MIN_COMPLETENESS, context
            ),
            fixed_interval_correction=self.parameterAsBoolean(
                parameters, self.FIXED_INTERVAL_CORRECTION, context
            ),
            distribution=self._DIST_OPTIONS[
                self.parameterAsEnum(parameters, self.DISTRIBUTION, context)
            ],
            tolerance_pct=self.parameterAsDouble(
                parameters, self.TOLERANCE_PCT, context
            ),
            n_bootstrap=self.parameterAsInt(parameters, self.N_BOOTSTRAP, context),
            random_seed=self.parameterAsInt(parameters, self.RANDOM_SEED, context),
            reference_tier=self.parameterAsEnum(
                parameters, self.REFERENCE_TIER, context
            )
            + 1,
            reference_is_areal=self.parameterAsBoolean(
                parameters, self.REFERENCE_IS_AREAL, context
            ),
            timezone_offset_h=self.parameterAsDouble(
                parameters, self.TIMEZONE_OFFSET_H, context
            ),
            build_library=want_library,
        )

        try:
            df = pd.read_csv(series_path)
            grid, native, site_used, warns = core.prepare_series(
                df,
                s(self.TIME_COLUMN),
                s(self.DEPTH_COLUMN),
                s(self.SITE),
                timezone_offset_h=cfg.timezone_offset_h,
            )
            ref_long, ref_info = read_ddf_csv(ddf_path, site=s(self.DDF_SITE))
            ref = DDFTable(ref_long)
        except (ValueError, OSError) as e:
            raise QgsProcessingException(str(e)) from e

        map_choice = self.parameterAsEnum(parameters, self.MAP_SOURCE, context)
        if map_choice == 0:
            v = self.parameterAsDouble(parameters, self.MAP_VALUE, context)
            if v <= 0:
                raise QgsProcessingException(
                    "Enter the target MAP (mm/yr), or choose another MAP source."
                )
            cfg.target_map_mm, cfg.map_source = v, "value entered by user"
        elif map_choice == 1:
            v = ref_info.get("map_mm")
            if not v:
                raise QgsProcessingException(
                    "The reference DDF CSV carries no 'MAP (mm)' value - enter the "
                    "target MAP instead."
                )
            cfg.target_map_mm, cfg.map_source = v, "reference DDF CSV (MAP (mm))"
        else:
            cfg.target_map_mm, cfg.map_source = None, "input series (retained)"
        feedback.pushInfo(
            "Target MAP: "
            + (f"{cfg.target_map_mm:.1f} mm/yr" if cfg.target_map_mm else "input MAP")
            + f" ({cfg.map_source})"
        )

        for w in warns:
            feedback.pushWarning(w)
        feedback.pushInfo(
            f"Series: {site_used or 'single site'}, native step {native:g} min, "
            f"{grid.index.min()} to {grid.index.max()}"
        )
        feedback.pushInfo(
            f"Reference DDF: {len(ref.durations)} durations, return periods "
            f"{', '.join(f'{t:g}' for t in ref.aris)} yr"
        )
        for conv in ref_info.get("fixed_day_durations", []):
            feedback.pushInfo(f"Reference fixed-day duration converted: {conv}")
        for drop in ref_info.get("dropped_durations", []):
            feedback.pushInfo(
                f"Reference duration '{drop}' not used (continuous duration of the "
                "same length preferred)"
            )

        def progress(pct):
            feedback.setProgress(int(pct))
            return not feedback.isCanceled()

        try:
            result = core.StormLibraryEngine(cfg).run(
                grid,
                native,
                ref,
                progress=progress,
                ref_long=ref_long,
                ref_info=ref_info,
            )
        except InterruptedError:
            raise QgsProcessingException("Cancelled by user.") from None
        except ValueError as e:
            raise QgsProcessingException(str(e)) from e
        result.metadata["input_series"] = series_path
        result.metadata["input_site"] = site_used or ""
        result.metadata["input_ddf"] = ddf_path

        for w in result.warnings:
            feedback.pushWarning(w)
        if cfg.reference_tier == 3:
            feedback.pushWarning(core.INDICATIVE_STAMP)
        mi = result.map_info
        feedback.pushInfo(
            f"MAP: input {mi['map_input_mm']:.1f} -> target {mi['map_target_mm']:.1f}"
            f" -> adjusted {mi['map_output_mm']:.1f} mm/yr "
            f"(gamma {mi['transfer_gamma']:.3f}; wet steps/yr "
            f"{mi['wet_steps_per_year_input']:.0f} -> "
            f"{mi['wet_steps_per_year_adjusted']:.0f})"
        )
        feedback.pushInfo(f"BEFORE: {result.overall_verdict}")
        feedback.pushInfo(f"AFTER:  {result.verdict_after}")
        for d in sorted({r["duration_min"] for r in result.after_rows}):
            rows = [r for r in result.after_rows if r["duration_min"] == d]
            fb = sum(r["F_before"] for r in rows) / len(rows)
            fa = sum(r["F_after"] for r in rows) / len(rows)
            feedback.pushInfo(
                f"  {rows[0]['duration']:>6} ({rows[0]['role']}): "
                f"mean F {fb:.2f} -> {fa:.2f}"
            )

        paths = {
            key: output_path(SPEC_BY_KEY[key], folders, project_number, site_used)
            for key in want
        }
        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []  # (label, path), for the report and the log

        def save(key, writer):
            path = paths[key]
            try:
                writer(path)
            except OSError as e:
                feedback.reportError(
                    f"{Path(path).name} could not be written ({e}) - it is probably "
                    "open in another program; close it and run again.",
                    fatalError=False,
                )
                return False
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))
            return True

        if self.OUTPUT_ADJUSTED in want:
            save(
                self.OUTPUT_ADJUSTED,
                lambda p: export.write_adjusted_series_csv(result, p, site=site_used),
            )
        for key, fn in (
            (self.OUTPUT_CONSISTENCY, export.write_consistency_csv),
            (self.OUTPUT_METADATA, export.write_metadata_csv),
            (self.OUTPUT_ENSEMBLE, export.write_ensemble_csv),
            (self.OUTPUT_LIBRARY, export.write_library_csv),
            (self.OUTPUT_EVENTS, export.write_events_csv),
        ):
            if key in want:
                save(key, lambda p, fn=fn: fn(result, p))
        if self.OUTPUT_ENSEMBLE in want and not result.ensemble:
            feedback.pushWarning(
                "Ensemble CSV requested but realisations = 0 - file is header-only."
            )

        chart = None
        chart_notes = []
        if self.OUTPUT_PNG in want or self.OUTPUT_REPORT in want:
            try:
                chart, chart_notes = export.flatness_figure(result)
            except ImportError:
                feedback.pushWarning("matplotlib not available - chart not produced.")
            except Exception as e:  # noqa: BLE001 - a chart must not lose the results
                feedback.pushWarning(f"The chart could not be drawn: {e}")
            for note in chart_notes:
                feedback.pushWarning(note)
        if self.OUTPUT_PNG in want and chart is not None:

            def write_png(p):
                with open(p, "wb") as fh:
                    fh.write(chart.getvalue())

            save(self.OUTPUT_PNG, write_png)

        if self.OUTPUT_REPORT in want:
            path = paths[self.OUTPUT_REPORT]
            try:
                from . import report

                if chart is not None:
                    chart.seek(0)
                notes = report.write_docx(
                    result,
                    path,
                    chart=chart,
                    inputs={"series": series_path, "ddf": ddf_path, "site": site_used},
                    context={
                        "project_number": project_number,
                        "project_name": project_name,
                        "software": _software_version(),
                        "tool_version": core.TOOL_VERSION,
                        "settings": [
                            ("Target MAP (mm/yr)", f"{cfg.target_map_mm:.1f}"
                             if cfg.target_map_mm else "input series' own MAP"),
                        ],
                        "outputs": written + [("This report", path)],
                        "warnings": list(warns) + chart_notes,
                    },
                )  # fmt: skip
                for note in notes:
                    if note not in chart_notes:
                        feedback.pushWarning(note)
                outputs[self.OUTPUT_REPORT] = path
                written.append((SPEC_BY_KEY[self.OUTPUT_REPORT].label, path))
            except ImportError:
                feedback.pushWarning(
                    "python-docx is not installed in QGIS's Python - report not "
                    "written. Install it from the OSGeo4W Shell: python -m pip "
                    "install python-docx"
                )
            except OSError as e:
                feedback.reportError(
                    f"{Path(path).name} could not be written ({e}) - it is probably "
                    "open in Word; close it and run again.",
                    fatalError=False,
                )

        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        return outputs
