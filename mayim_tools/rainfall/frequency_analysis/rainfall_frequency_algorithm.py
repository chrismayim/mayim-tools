"""Processing Toolbox algorithm: Precipitation Data to DDF.

Time/Depth precipitation CSV in -> annual maximum series, per-duration
distribution fits (diagnostic), a duration-consistent DDF model
(recommended DDF with bootstrap bounds) and a Word report out.

Thin wrapper - all real logic lives in rainfall/_common/rfa/ (zero QGIS
dependency, independently testable).
"""

from pathlib import Path

import pandas as pd
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools.rainfall._common.rfa.analysis import (
    DEFAULT_DISTRIBUTIONS,
    DEFAULT_EXCEEDANCE_PROBABILITIES,
    run_frequency_analysis,
)
from mayim_tools.rainfall._common.rfa.export import (
    write_ams,
    write_metadata,
    write_parameters,
    write_quantiles,
    write_recommendations,
    write_recommended_ddf,
    write_year_completeness,
)
from mayim_tools.rainfall._common.rfa.timebase import EXTENDED_DURATIONS_MIN

_ADVANCED = QgsProcessingParameterDefinition.Flag.FlagAdvanced


class RainfallFrequencyAlgorithm(QgsProcessingAlgorithm):

    INPUT_CSV = "INPUT_CSV"
    TIMESTAMP_COL = "TIMESTAMP_COL"
    DEPTH_COL = "DEPTH_COL"
    TIMESTAMP_FORMAT = "TIMESTAMP_FORMAT"
    TIMEZONE_OFFSET = "TIMEZONE_OFFSET"
    DURATIONS = "DURATIONS"
    FIXED_INTERVAL_CORRECTION = "FIXED_INTERVAL_CORRECTION"
    EXCEEDANCE_PROBABILITIES = "EXCEEDANCE_PROBABILITIES"
    MIN_COMPLETENESS = "MIN_COMPLETENESS"
    MIN_YEARS_WARNING = "MIN_YEARS_WARNING"
    MODEL_DISTRIBUTION = "MODEL_DISTRIBUTION"
    SMOOTH_GROWTH = "SMOOTH_GROWTH"
    N_BOOTSTRAP = "N_BOOTSTRAP"
    RANDOM_SEED = "RANDOM_SEED"
    DISTRIBUTIONS = "DISTRIBUTIONS"
    GEV_METHOD = "GEV_METHOD"
    OUTPUT_RECOMMENDED_DDF = "OUTPUT_RECOMMENDED_DDF"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_DDF_PNG = "OUTPUT_DDF_PNG"
    OUTPUT_PARAMETERS = "OUTPUT_PARAMETERS"
    OUTPUT_QUANTILES = "OUTPUT_QUANTILES"
    OUTPUT_RECOMMENDATIONS = "OUTPUT_RECOMMENDATIONS"
    OUTPUT_AMS = "OUTPUT_AMS"
    OUTPUT_YEAR_COMPLETENESS = "OUTPUT_YEAR_COMPLETENESS"
    OUTPUT_METADATA = "OUTPUT_METADATA"

    _DISTRIBUTION_OPTIONS = list(DEFAULT_DISTRIBUTIONS)  # GEV, Gumbel, GLO, LP3
    _GEV_METHOD_OPTIONS = ["L-moments", "MLE"]
    _MODEL_OPTIONS = ["GEV", "GLO", "GUMBEL"]

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return RainfallFrequencyAlgorithm()

    def name(self):
        # Kept for backwards compatibility with saved models/scripts.
        return "rainfall_frequency_stage1"

    def displayName(self):
        return "Precipitation data to DDF"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "PURPOSE\n"
            "Derives a Depth-Duration-Frequency (DDF) table from a recorded "
            "precipitation time series - a gauge record, or a gridded product "
            "extracted at a point (ERA5, IMERG, CMORPH, CHIRPS ...). The recommended "
            "DDF is duration-consistent: depths always increase with duration and "
            "with return period, so the curves never cross. Outputs the recommended "
            "DDF CSV (Design Rainfall column layout, feeds straight into DDF to "
            "Hyetographs, Design Storm Ensembles and Adjust Sub-daily Rainfall to "
            "DDF), a Word report and optional diagnostic CSVs.\n\n"
            "METHOD\n"
            "1. The series is parsed and regularised to its native time step. "
            "Missing data stay missing (never zero); years below the completeness "
            "threshold are excluded from every duration.\n"
            "2. Annual maximum series (AMS) are extracted per calendar year with "
            "sliding windows at each duration. Optionally the Weiss (1964) factor "
            "1/(1-1/(8n)) corrects fixed-interval maxima to true maxima.\n"
            "3. Diagnostics: GEV, Gumbel, GLO and LP3 are fitted to each duration by "
            "L-moments, and the L-moment ratio diagram ranks them per duration.\n"
            "4. Recommended DDF: one model is fitted to all durations together - "
            "X_d = l1(d) x Y_d, with the mean l1(d) = lambda1 (d/60) "
            "((d+theta)/(60+theta))^-eta increasing smoothly with duration "
            "(Koutsoyiannis et al. 1998; Overeem et al. 2008) and a single growth "
            "distribution Y_d (GEV by default) whose L-CV and L-skewness vary "
            "smoothly with log-duration (weighted least squares across durations). "
            "Any residual crossings are removed by isotonic regression (Roksvag et "
            "al. 2021).\n"
            "5. Uncertainty: whole years are bootstrap-resampled (the same years for "
            "every duration) and the full model refitted, giving 5-95% bounds.\n\n"
            "BACKGROUND\n"
            "Fitting each duration separately, and especially picking a different "
            "distribution per duration, can give a longer duration a smaller design "
            "depth than a shorter one at the same return period - physically "
            "impossible and not defensible in review. Published DDF/IFD products "
            "(Smithers & Schulze 2003 for South Africa; ARR 2016/2019 IFDs for "
            "Australia) avoid this by treating all durations as one consistent "
            "family. The per-duration fits are kept as diagnostics and compared with "
            "the model in the report. A DDF from a gridded product is an areal, "
            "product-specific DDF: it is typically low at short durations and rare "
            "return periods compared with gauge DDFs.\n\n"
            "PARAMETERS\n"
            "Rainfall CSV: incremental (not cumulative) depths at a fixed time step, "
            "e.g. straight from the Extract tools. At least 10 complete years; 20+ "
            "recommended, 30+ for 100-year estimates.\n"
            "\n"
            "Timestamp / depth column names: exact CSV headers (e.g. Time / Depth, "
            "or ValidTime / PrecipitationMM for Extract-tool outputs).\n"
            "\n"
            "Timestamp format: blank = auto-detect (works for ISO dates). Give a "
            "format only if dates are ambiguous, e.g. %d/%m/%Y %H:%M.\n"
            "\n"
            "Time-zone offset (hours, default 0): added to every timestamp before "
            "the calendar-year split. Gridded products are in UTC; enter +2 for "
            "South Africa (SAST) or +8 for Western Australia (AWST) so that storms "
            "around midnight on 31 December fall in the right year and daily "
            "windows follow local days. Leave 0 for gauge data already in local "
            "time. Effect on DDF depths is usually small.\n"
            "\n"
            "Durations (minutes, comma-separated): default 5 min to 7 days. Durations "
            "finer than, or not whole multiples of, the data's time step are "
            "skipped automatically (hourly data: 1 h upwards; daily: 1 day "
            "upwards). Use at least 5 durations spanning the range you need.\n"
            "\n"
            "Apply fixed-interval correction (default on): clock-interval totals "
            "under-record the true maximum, which usually straddles two intervals. "
            "Maxima are multiplied by 1/(1-1/(8n)) for an n-step window (1.143 for "
            "1 step, 1.067 for 2, 1.005 for 24). Leave ON to compare with published "
            "gauge DDFs (continuous durations). Turn OFF only to reproduce a DDF "
            "made without it. Use the same setting when this DDF is later used as "
            "a reference in Adjust Sub-daily Rainfall to DDF.\n"
            "\n"
            "Exceedance probabilities (AEP, comma-separated): default 0.99 ... 0.002 "
            "(1.01 to 500 years). T = 1/AEP.\n"
            "\n"
            "Minimum year completeness (0-1, default 0.90): years with less valid "
            "data are excluded. Typical 0.85-0.95; lower values risk missing the "
            "annual maximum.\n"
            "\n"
            "Minimum years before low-confidence warning (default 10): per-duration "
            "fits with fewer years are flagged.\n"
            "\n"
            "DDF model distribution (default GEV): the growth distribution of the "
            "duration-consistent model. GEV matches Design Rainfall SA (Smithers & "
            "Schulze 2003) and ARR IFDs. GLO for heavier-tailed data; Gumbel (no "
            "shape) only if the ratio diagram clearly supports it.\n"
            "\n"
            "Smooth L-CV and L-skewness with duration (default on): lets the "
            "variability and skewness of the annual maxima change smoothly from "
            "short to long durations (short durations are usually more variable). "
            "Off = one growth curve for all durations (simple scaling); this can "
            "under-estimate rare short-duration depths.\n"
            "\n"
            "Bootstrap replicates (default 200; 0 = off): 200 gives stable 5-95% "
            "bounds; 500-1000 for final reports (a few seconds to a minute).\n"
            "\n"
            "Random seed (default 42): fixes the bootstrap so results are "
            "reproducible. Change only to test sensitivity.\n"
            "\n"
            "Advanced - distributions / GEV method: which distributions are fitted "
            "per duration for the diagnostics (all four recommended) and GEV "
            "L-moments (default) or MLE.\n"
            "\n"
            "OUTPUTS\n"
            "Recommended DDF CSV: one row per duration; '{T}yr Depth (mm)' plus "
            "'{T}yr Lower/Upper (mm)' bounds. Word report: inputs, method, model "
            "parameters, recommended DDF table and chart, AMS, ratio diagram, "
            "model-vs-independent comparison, warnings, references. Optional: DDF "
            "chart PNG, per-duration parameters and quantiles (all distributions), "
            "ratio-diagram ranking, AMS, year completeness and metadata CSVs."
        )

    def initAlgorithm(self, config=None):
        def add(p, advanced=False):
            if advanced:
                p.setFlags(p.flags() | _ADVANCED)
            self.addParameter(p)

        add(
            QgsProcessingParameterFile(
                self.INPUT_CSV, "Rainfall CSV (Time/Depth)", extension="csv"
            )
        )
        add(
            QgsProcessingParameterString(
                self.TIMESTAMP_COL, "Timestamp column name", defaultValue="Time"
            )
        )
        add(
            QgsProcessingParameterString(
                self.DEPTH_COL, "Depth column name", defaultValue="Depth"
            )
        )
        add(
            QgsProcessingParameterString(
                self.TIMESTAMP_FORMAT,
                "Timestamp format (blank = auto-detect, e.g. %Y-%m-%d %H:%M:%S)",
                optional=True,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.TIMEZONE_OFFSET,
                "Time-zone offset added to timestamps (hours; +2 SAST, +8 AWST, 0 = none)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=-14.0,
                maxValue=14.0,
            )
        )
        add(
            QgsProcessingParameterString(
                self.DURATIONS,
                "Durations (minutes, comma-separated; unsuitable ones skipped)",
                defaultValue=",".join(str(d) for d in EXTENDED_DURATIONS_MIN),
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
            QgsProcessingParameterString(
                self.EXCEEDANCE_PROBABILITIES,
                "Exceedance probabilities (AEP, comma-separated)",
                defaultValue=",".join(str(p) for p in DEFAULT_EXCEEDANCE_PROBABILITIES),
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MIN_COMPLETENESS,
                "Minimum year completeness to include in AMS (0-1)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.90,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.MIN_YEARS_WARNING,
                "Minimum years before flagging a duration's fit as low-confidence",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=10,
                minValue=3,
            )
        )
        add(
            QgsProcessingParameterEnum(
                self.MODEL_DISTRIBUTION,
                "DDF model distribution (duration-consistent recommended DDF)",
                options=self._MODEL_OPTIONS,
                defaultValue=0,
            )
        )
        add(
            QgsProcessingParameterBoolean(
                self.SMOOTH_GROWTH,
                "Smooth L-CV and L-skewness with duration (recommended)",
                defaultValue=True,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.N_BOOTSTRAP,
                "Bootstrap replicates for 5-95% bounds (0 = off)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=200,
                minValue=0,
                maxValue=5000,
            )
        )
        add(
            QgsProcessingParameterNumber(
                self.RANDOM_SEED,
                "Random seed (bootstrap)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=42,
                minValue=0,
            ),
            advanced=True,
        )
        add(
            QgsProcessingParameterEnum(
                self.DISTRIBUTIONS,
                "Distributions fitted per duration (diagnostics; all four recommended)",
                options=self._DISTRIBUTION_OPTIONS,
                allowMultiple=True,
                defaultValue=list(range(len(self._DISTRIBUTION_OPTIONS))),
            ),
            advanced=True,
        )
        add(
            QgsProcessingParameterEnum(
                self.GEV_METHOD,
                "GEV fitting method for per-duration diagnostics",
                options=self._GEV_METHOD_OPTIONS,
                defaultValue=0,
            ),
            advanced=True,
        )
        add(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_RECOMMENDED_DDF,
                "Output: recommended DDF CSV (duration-consistent, with bounds)",
                fileFilter="CSV files (*.csv)",
            )
        )
        add(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_REPORT,
                "Output: Word report (.docx, optional)",
                fileFilter="Word documents (*.docx)",
                optional=True,
            )
        )
        for key, label in ((self.OUTPUT_DDF_PNG, "Output: DDF chart PNG (optional)"),):
            add(
                QgsProcessingParameterFileDestination(
                    key, label, fileFilter="PNG files (*.png)", optional=True
                )
            )
        for key, label in (
            (
                self.OUTPUT_PARAMETERS,
                "Output: per-duration distribution parameters CSV (diagnostic, optional)",
            ),
            (
                self.OUTPUT_QUANTILES,
                "Output: per-duration quantiles CSV, all distributions (diagnostic, optional)",
            ),
            (
                self.OUTPUT_RECOMMENDATIONS,
                "Output: L-moment ratio diagram ranking CSV (diagnostic, optional)",
            ),
            (self.OUTPUT_AMS, "Output: annual maximum series CSV (optional)"),
            (
                self.OUTPUT_YEAR_COMPLETENESS,
                "Output: year completeness / audit CSV (optional)",
            ),
            (
                self.OUTPUT_METADATA,
                "Output: metadata / model parameters CSV (optional)",
            ),
        ):
            add(
                QgsProcessingParameterFileDestination(
                    key, label, fileFilter="CSV files (*.csv)", optional=True
                )
            )

    @staticmethod
    def _floats(text, what):
        try:
            return tuple(float(x.strip()) for x in (text or "").split(",") if x.strip())
        except ValueError:
            raise QgsProcessingException(f"Could not parse {what}: {text!r}") from None

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        csv_path = self.parameterAsFile(parameters, self.INPUT_CSV, context)
        timestamp_col = self.parameterAsString(parameters, self.TIMESTAMP_COL, context)
        depth_col = self.parameterAsString(parameters, self.DEPTH_COL, context)
        timestamp_format = (
            self.parameterAsString(parameters, self.TIMESTAMP_FORMAT, context) or None
        )
        dist_indices = self.parameterAsEnums(parameters, self.DISTRIBUTIONS, context)
        distributions = (
            tuple(self._DISTRIBUTION_OPTIONS[i] for i in dist_indices)
            or DEFAULT_DISTRIBUTIONS
        )
        gev_method = self._GEV_METHOD_OPTIONS[
            self.parameterAsEnum(parameters, self.GEV_METHOD, context)
        ]
        model_dist = self._MODEL_OPTIONS[
            self.parameterAsEnum(parameters, self.MODEL_DISTRIBUTION, context)
        ]
        aeps = self._floats(
            self.parameterAsString(parameters, self.EXCEEDANCE_PROBABILITIES, context),
            "exceedance probabilities",
        )
        if not aeps or any(not 0 < a < 1 for a in aeps):
            raise QgsProcessingException(
                "Exceedance probabilities must be between 0 and 1 (e.g. 0.5,0.1,0.01)."
            )
        durations = self._floats(
            self.parameterAsString(parameters, self.DURATIONS, context), "durations"
        ) or tuple(EXTENDED_DURATIONS_MIN)
        n_boot = self.parameterAsInt(parameters, self.N_BOOTSTRAP, context)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise QgsProcessingException(f"Could not read CSV: {e}") from e
        feedback.pushInfo(f"Loaded {len(df)} rows from {csv_path}")
        if n_boot:
            feedback.pushInfo(
                f"Fitting the DDF model with {n_boot} bootstrap replicates ..."
            )

        try:
            result = run_frequency_analysis(
                df,
                timestamp_col=timestamp_col,
                depth_col=depth_col,
                timestamp_format=timestamp_format,
                distributions=distributions,
                gev_method=gev_method,
                min_completeness=self.parameterAsDouble(
                    parameters, self.MIN_COMPLETENESS, context
                ),
                min_years_warning=self.parameterAsInt(
                    parameters, self.MIN_YEARS_WARNING, context
                ),
                exceedance_probabilities=aeps,
                requested_durations_min=tuple(sorted(set(durations))),
                fixed_interval_correction=self.parameterAsBool(
                    parameters, self.FIXED_INTERVAL_CORRECTION, context
                ),
                timezone_offset_h=self.parameterAsDouble(
                    parameters, self.TIMEZONE_OFFSET, context
                ),
                model_distribution=model_dist,
                vary_cv=self.parameterAsBool(parameters, self.SMOOTH_GROWTH, context),
                n_bootstrap=n_boot,
                random_seed=self.parameterAsInt(parameters, self.RANDOM_SEED, context),
                whole_multiples_only=True,
            )
        except ValueError as e:
            raise QgsProcessingException(str(e)) from e

        for w in result.warnings:
            feedback.pushWarning(w)
        md = result.metadata
        feedback.pushInfo(f"Native interval: {md.get('native_interval_min')} min")
        feedback.pushInfo(
            "Durations: " + ", ".join(d.duration_label for d in result.duration_series)
        )
        m = result.ddf_model
        if m is not None:
            feedback.pushInfo(
                f"DDF model ({m.distribution}): theta={m.theta:.1f} min, "
                f"eta={m.eta:.3f}, L-CV(60 min)={m.cv60:.3f}, beta={m.beta:.3f}, "
                f"tau3(60 min)={m.tau3:.3f}, g={m.tau3_slope:.3f}"
            )
            for note in m.notes:
                feedback.pushInfo(f"  {note}")

        outputs = {}
        ddf_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_RECOMMENDED_DDF, context
        )
        write_recommended_ddf(result, ddf_path)
        outputs[self.OUTPUT_RECOMMENDED_DDF] = ddf_path

        for key, fn in (
            (self.OUTPUT_PARAMETERS, write_parameters),
            (self.OUTPUT_QUANTILES, write_quantiles),
            (self.OUTPUT_RECOMMENDATIONS, write_recommendations),
            (self.OUTPUT_AMS, write_ams),
            (self.OUTPUT_YEAR_COMPLETENESS, write_year_completeness),
            (self.OUTPUT_METADATA, write_metadata),
        ):
            path = self.parameterAsFileOutput(parameters, key, context)
            if path:
                fn(result, path)
                outputs[key] = path

        png = self.parameterAsFileOutput(parameters, self.OUTPUT_DDF_PNG, context)
        if png:
            from . import charts

            if charts.write_ddf_chart(result, png):
                outputs[self.OUTPUT_DDF_PNG] = png
            else:
                feedback.pushWarning("matplotlib not available - chart not produced.")

        rep = self.parameterAsFileOutput(parameters, self.OUTPUT_REPORT, context)
        if rep:
            try:
                from . import report

                report.write_docx(
                    result,
                    rep,
                    inputs={"series": csv_path},
                )
                outputs[self.OUTPUT_REPORT] = rep
            except ImportError:
                feedback.pushWarning(
                    "python-docx is not installed - Word report not produced "
                    "(install with: python -m pip install python-docx)."
                )
        return outputs
