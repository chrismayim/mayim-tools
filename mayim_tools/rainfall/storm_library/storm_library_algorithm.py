"""StormLibraryAlgorithm - the Processing algorithm.

Thin wrapper - all real logic lives in core.py / export.py and the
shared mayim_tools.rainfall._common package (zero QGIS dependency,
independently testable; see tests/test_storm_library_core.py).
QGIS4/Qt6-safe: only File/Number/String/Boolean/Enum/FileDestination
Processing parameters are used.
"""

from pathlib import Path

import pandas as pd
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools.rainfall._common.ddf import DDFTable, read_ddf_csv

from . import core, export


class StormLibraryAlgorithm(QgsProcessingAlgorithm):
    INPUT_SERIES = "INPUT_SERIES"
    SITE = "SITE"
    TIME_COLUMN = "TIME_COLUMN"
    DEPTH_COLUMN = "DEPTH_COLUMN"
    TIMEZONE_OFFSET_H = "TIMEZONE_OFFSET_H"
    INPUT_DDF = "INPUT_DDF"
    DDF_SITE = "DDF_SITE"
    REFERENCE_TIER = "REFERENCE_TIER"
    REFERENCE_IS_AREAL = "REFERENCE_IS_AREAL"
    ANCHOR_DURATION = "ANCHOR_DURATION"
    TEST_DURATIONS = "TEST_DURATIONS"
    IETD_HOURS = "IETD_HOURS"
    MIN_EVENT_DEPTH = "MIN_EVENT_DEPTH"
    MIN_COMPLETENESS = "MIN_COMPLETENESS"
    FIXED_INTERVAL_CORRECTION = "FIXED_INTERVAL_CORRECTION"
    DISTRIBUTION = "DISTRIBUTION"
    TOLERANCE_PCT = "TOLERANCE_PCT"
    N_BOOTSTRAP = "N_BOOTSTRAP"
    RANDOM_SEED = "RANDOM_SEED"
    OUTPUT_CONSISTENCY = "OUTPUT_CONSISTENCY"
    OUTPUT_LIBRARY = "OUTPUT_LIBRARY"
    OUTPUT_EVENTS = "OUTPUT_EVENTS"
    OUTPUT_METADATA = "OUTPUT_METADATA"
    OUTPUT_PNG = "OUTPUT_PNG"

    _TIER_OPTIONS = [core.REFERENCE_TIERS[k] for k in (1, 2, 3)]
    _ANCHOR_OPTIONS = [
        ("12 h", 720.0),
        ("24 h", 1440.0),
        ("48 h", 2880.0),
        ("72 h", 4320.0),
    ]
    _DIST_OPTIONS = list(core.DISTRIBUTION_CHOICES)

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return StormLibraryAlgorithm()

    def name(self):
        return "storm_library_ddf_check"

    def displayName(self):
        return "Storm Library & DDF Consistency Check"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "PURPOSE\n"
            "Tests whether the within-storm structure of a sub-daily gridded "
            "rainfall series (ERA5, IMERG, CMORPH, PERSIANN-CCS...) is consistent "
            "with a reference, gauge-based DDF table, and builds a normalised "
            "storm library from it for Monte Carlo design-storm simulation. Answers: "
            "'if these storms are scaled to my DDF at one anchor duration, do the "
            "shorter bursts inside them also match my DDF?'\n\n"
            "METHOD\n"
            "1. The series is regularised (missing data stays missing, never zero) "
            "and incomplete years are excluded.\n"
            "2. The product's own annual-maximum curve G is fitted at the anchor "
            "duration (default 24 h).\n"
            "3. Independent storms are identified (IETD). Each storm is rescaled so "
            "its anchor-duration depth equals the reference DDF depth at the storm's "
            "own return period in G (annual-maximum quantile mapping). Below the "
            "table's most frequent return period the scale factor is held constant; "
            "above its rarest it extrapolates in ln(T) and is flagged.\n"
            "4. Annual maxima of the rescaled series are extracted at every test "
            "duration and fitted with the same distribution, giving an implied DDF.\n"
            "5. Flatness index F = implied / reference per duration and return "
            "period, with 5/50/95% intervals from a year-block bootstrap. F < 1 "
            "means the product's short bursts are flatter than the reference.\n"
            "6. Self-check: F at the anchor duration must be ~1 by construction.\n"
            "7. Each duration is classed consistent / inconclusive / flatter "
            "(sharpening indicated) / peakier (review); the overall verdict decides "
            "whether the cascade sharpening step is needed before Monte Carlo use.\n"
            "8. Storm library: best window per storm for each design duration the "
            "native step resolves (>= 3 steps), as % increments, with Early/Middle/"
            "Late class, AEP band against the reference DDF, month, and an ARR-style "
            "embedded-burst flag (a sub-burst rarer than the burst itself).\n\n"
            "BACKGROUND\n"
            "Reanalysis and satellite products under-represent short-duration "
            "convective bursts through grid-scale averaging and parameterised "
            "convection (e.g. Guerreiro et al. 2024; Lavers et al. 2022). Following "
            "the ARR Book 4 Monte Carlo principle, magnitude is taken from the gauge-"
            "based DDF and only structure from the gridded product; F quantifies how "
            "much structural correction is still needed. For point design both the "
            "areal-smoothing and model-physics parts of F < 1 must be corrected; for "
            "catchment design supply an ARF-applied reference and tick 'reference is "
            "areal'. Fixed-interval maxima are corrected with the Weiss (1964) factor "
            "1/(1-1/(8n)); day-labelled reference durations (fixed 08:00 readings) "
            "are converted the same way. Return periods are annual-maximum series "
            "values (AEP = 1/T). Tier 3 (gridded-only) references are stamped "
            "INDICATIVE - NOT VALIDATED in every output.\n\n"
            "PARAMETERS\n"
            "Rainfall series: CSV from any Mayim Tools extractor (Site, ValidTime, "
            "PrecipitationMM) or any time/depth CSV; incremental depths. Site / "
            "column names: optional (auto-detected). Time-zone offset: hours added "
            "to the timestamps (e.g. +2 for UTC -> SAST).\n"
            "Reference DDF: Design Rainfall (South Africa) or Precipitation data to "
            "DDF CSV, or any 'Duration' + '<T>yr Depth (mm)' table. Reference tier "
            "and 'reference is areal' are recorded and stamped on outputs.\n"
            "Anchor duration: where magnitude is mapped (24 h recommended). Test "
            "durations: comma-separated minutes, blank = every reference duration "
            "that is a multiple of the native step, up to 72 h.\n"
            "IETD / minimum event depth: storm separation. Minimum completeness: "
            "fraction of a year that must be present. Distribution: should match how "
            "the reference was derived (GEV L-moments for DRESA and ARR 2016). "
            "Tolerance: |F-1| accepted as consistent. Bootstrap replicates and "
            "random seed: uncertainty and reproducibility.\n"
            "Outputs: consistency CSV (headline), storm library CSV (input to the "
            "Monte Carlo step), event audit CSV, metadata CSV and an optional PNG "
            "chart of F against duration."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_SERIES, "Rainfall series CSV (sub-daily)", extension="csv"
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.SITE, "Site (multi-site CSVs; blank = first)", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.TIME_COLUMN, "Time column (blank = auto)", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DEPTH_COLUMN, "Depth column (blank = auto)", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TIMEZONE_OFFSET_H,
                "Time-zone offset added to timestamps (h), e.g. 2 for UTC -> SAST",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=-14.0,
                maxValue=14.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_DDF, "Reference DDF CSV", extension="csv"
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DDF_SITE,
                "Reference DDF site (multi-site Design Rainfall CSVs; blank = first)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.REFERENCE_TIER,
                "Reference tier",
                options=self._TIER_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REFERENCE_IS_AREAL,
                "Reference DDF already has an areal reduction factor applied",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ANCHOR_DURATION,
                "Anchor duration (magnitude mapped to the reference here)",
                options=[a[0] for a in self._ANCHOR_OPTIONS],
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.TEST_DURATIONS,
                "Test durations in minutes, comma-separated (blank = automatic)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IETD_HOURS,
                "Inter-event time definition, IETD (h)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_IETD_HOURS,
                minValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_EVENT_DEPTH,
                "Minimum storm depth (mm)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_MIN_EVENT_DEPTH_MM,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_COMPLETENESS,
                "Minimum year completeness (0-1)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_MIN_COMPLETENESS,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FIXED_INTERVAL_CORRECTION,
                "Apply fixed-interval (Weiss 1964) correction to annual maxima",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DISTRIBUTION,
                "Distribution (match the reference DDF's method)",
                options=self._DIST_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TOLERANCE_PCT,
                "Tolerance |F-1| treated as consistent (%)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=core.DEFAULT_TOLERANCE_PCT,
                minValue=0.0,
                maxValue=100.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_BOOTSTRAP,
                "Bootstrap replicates",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=core.DEFAULT_N_BOOTSTRAP,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANDOM_SEED,
                "Random seed",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=core.DEFAULT_RANDOM_SEED,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CONSISTENCY,
                "Output: DDF consistency (flatness index) CSV",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_LIBRARY,
                "Output: storm library CSV",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_EVENTS,
                "Output: storm event audit CSV (optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_METADATA,
                "Output: metadata / diagnostics CSV (optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_PNG,
                "Output: flatness chart PNG (optional)",
                fileFilter="PNG files (*.png)",
                optional=True,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        series_path = self.parameterAsFile(parameters, self.INPUT_SERIES, context)
        ddf_path = self.parameterAsFile(parameters, self.INPUT_DDF, context)

        def s(name):
            return self.parameterAsString(parameters, name, context).strip() or None

        test_str = s(self.TEST_DURATIONS)
        try:
            tests = (
                tuple(float(x) for x in test_str.split(",") if x.strip())
                if test_str
                else None
            )
        except ValueError:
            raise QgsProcessingException(
                f"Could not parse test durations: {test_str!r}"
            ) from None

        cfg = core.StormLibraryConfig(
            anchor_min=self._ANCHOR_OPTIONS[
                self.parameterAsEnum(parameters, self.ANCHOR_DURATION, context)
            ][1],
            test_durations_min=tests,
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
        feedback.pushInfo(
            f"Anchor self-check: {'PASS' if result.self_check_passed else 'WARN'} "
            f"(max |F-1| = {result.metadata['self_check_max_deviation']:.2%})"
        )
        for row in result.duration_summary:
            feedback.pushInfo(f"  {row['duration']:>6}: {row['decision']}")
        feedback.pushInfo(f"OVERALL VERDICT: {result.overall_verdict}")

        outputs = {}
        out = self.parameterAsFileOutput(parameters, self.OUTPUT_CONSISTENCY, context)
        export.write_consistency_csv(result, out)
        outputs[self.OUTPUT_CONSISTENCY] = out
        out = self.parameterAsFileOutput(parameters, self.OUTPUT_LIBRARY, context)
        n = export.write_library_csv(result, out)
        outputs[self.OUTPUT_LIBRARY] = out
        feedback.pushInfo(f"Storm library: {n} patterns written.")
        out = self.parameterAsFileOutput(parameters, self.OUTPUT_EVENTS, context)
        if out:
            export.write_events_csv(result, out)
            outputs[self.OUTPUT_EVENTS] = out
        out = self.parameterAsFileOutput(parameters, self.OUTPUT_METADATA, context)
        if out:
            export.write_metadata_csv(result, out)
            outputs[self.OUTPUT_METADATA] = out
        out = self.parameterAsFileOutput(parameters, self.OUTPUT_PNG, context)
        if out:
            if export.write_flatness_png(result, out):
                outputs[self.OUTPUT_PNG] = out
            else:
                feedback.pushWarning("matplotlib not available - PNG not written.")
        return outputs
