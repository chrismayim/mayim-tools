"""Processing Toolbox algorithm: Time/Depth rainfall CSV in -> Stage 1
frequency analysis results out (AMS, distribution parameters for all
fitted distributions, design quantiles, and an L-moment ratio diagram
recommendation per duration).

Thin wrapper - all real logic lives in rfa/ (zero QGIS dependency,
independently testable; see rfa/tests/test_core.py).
"""

from pathlib import Path

import pandas as pd
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from .rfa.analysis import (
    DEFAULT_DISTRIBUTIONS,
    DEFAULT_EXCEEDANCE_PROBABILITIES,
    run_frequency_analysis,
)
from .rfa.export import (
    write_ams,
    write_metadata,
    write_parameters,
    write_quantiles,
    write_recommendations,
    write_recommended_ddf,
    write_year_completeness,
)


class RainfallFrequencyAlgorithm(QgsProcessingAlgorithm):

    INPUT_CSV = "INPUT_CSV"
    TIMESTAMP_COL = "TIMESTAMP_COL"
    DEPTH_COL = "DEPTH_COL"
    TIMESTAMP_FORMAT = "TIMESTAMP_FORMAT"
    DISTRIBUTIONS = "DISTRIBUTIONS"
    GEV_METHOD = "GEV_METHOD"
    MIN_COMPLETENESS = "MIN_COMPLETENESS"
    MIN_YEARS_WARNING = "MIN_YEARS_WARNING"
    EXCEEDANCE_PROBABILITIES = "EXCEEDANCE_PROBABILITIES"
    OUTPUT_PARAMETERS = "OUTPUT_PARAMETERS"
    OUTPUT_QUANTILES = "OUTPUT_QUANTILES"
    OUTPUT_RECOMMENDED_DDF = "OUTPUT_RECOMMENDED_DDF"
    OUTPUT_RECOMMENDATIONS = "OUTPUT_RECOMMENDATIONS"
    OUTPUT_AMS = "OUTPUT_AMS"
    OUTPUT_YEAR_COMPLETENESS = "OUTPUT_YEAR_COMPLETENESS"
    OUTPUT_METADATA = "OUTPUT_METADATA"

    _DISTRIBUTION_OPTIONS = list(DEFAULT_DISTRIBUTIONS)  # GEV, Gumbel, GLO, LP3
    _GEV_METHOD_OPTIONS = ["L-moments", "MLE"]

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return RainfallFrequencyAlgorithm()

    def name(self):
        return "rainfall_frequency_stage1"

    def displayName(self):
        return "Precipitation data to DDF"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "Stage 1 of a rainfall frequency analysis workflow (Stage 2 - DDF "
            "table construction - is a planned future addition). Given a CSV "
            "with a timestamp column and a precipitation-depth column (any "
            "interval from 30 min to daily), extracts the annual maximum "
            "series for durations from 30 min to 24 h - whichever are coarser "
            "than the data's native interval - and fits EVERY selected "
            "distribution (GEV, Gumbel, GLO, LP3 by default) to EVERY "
            "duration, reporting Location (Xi)/Scale (Alpha)/Shape (Kappa) "
            "plus design depths at the requested exceedance probabilities "
            "for all of them - nothing is hidden behind a single choice. "
            "An L-moment ratio diagram diagnostic also runs per duration: "
            "each candidate distribution's theoretical L-kurtosis (given the "
            "data's own L-skewness) is compared against the data's actual "
            "L-kurtosis, and the closest match is reported as the "
            "recommended distribution for that duration - a quantitative "
            "goodness-of-fit ranking, not a visual read of a chart. A final "
            "recommended-DDF output collapses this to one row per duration "
            "using only that duration's recommended distribution, in the "
            "same column format the Design Rainfall plugin uses - feeds "
            "directly into the SCS Design Storm plugin with no reformatting. "
            "GEV fitting defaults to L-moments (matching the WRC K5/1060 "
            "methodology used in the Design Rainfall plugin) with MLE "
            "available as an alternative - MLE is automatically seeded with "
            "the L-moment fit to avoid a real convergence failure mode found "
            "while building this (see dev/README.md). Years with data "
            "completeness below the threshold are excluded from ALL "
            "durations' AMS and reported transparently, never silently "
            "dropped. Missing rainfall is never treated as zero anywhere in "
            "this pipeline."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_CSV, "Rainfall CSV (Time/Depth)", extension="csv"
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.TIMESTAMP_COL, "Timestamp column name", defaultValue="Time"
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DEPTH_COL, "Depth column name", defaultValue="Depth"
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
                self.DISTRIBUTIONS,
                "Distributions to fit (all four recommended - enables the ratio diagram diagnostic)",
                options=self._DISTRIBUTION_OPTIONS,
                allowMultiple=True,
                defaultValue=list(range(len(self._DISTRIBUTION_OPTIONS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.GEV_METHOD,
                "GEV fitting method (Gumbel/GLO/LP3 always use their closed-form/L-moment fits)",
                options=self._GEV_METHOD_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_COMPLETENESS,
                "Minimum year completeness to include in AMS (0-1)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.90,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_YEARS_WARNING,
                "Minimum years before flagging a duration's fit as low-confidence",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=10,
                minValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.EXCEEDANCE_PROBABILITIES,
                "Exceedance probabilities (comma-separated)",
                defaultValue=",".join(str(p) for p in DEFAULT_EXCEEDANCE_PROBABILITIES),
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_PARAMETERS,
                "Output: distribution parameters CSV (Stage 1.3, all distributions)",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_QUANTILES,
                "Output: design quantiles CSV (Stage 1.4, all distributions)",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_RECOMMENDED_DDF,
                "Output: final recommended DDF table (one row per duration, ratio-diagram-recommended "
                "distribution only - same column format as the Design Rainfall plugin, feeds directly "
                "into the SCS Design Storm plugin)",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_RECOMMENDATIONS,
                "Output: L-moment ratio diagram recommendation CSV (which distribution fits best, per duration)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_AMS,
                "Output: annual maximum series CSV (Stage 1.1, optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_YEAR_COMPLETENESS,
                "Output: year completeness/audit CSV (optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_METADATA,
                "Output: metadata/diagnostics CSV (optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
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
        dist_indices = self.parameterAsEnums(parameters, self.DISTRIBUTIONS, context)
        distributions = (
            tuple(self._DISTRIBUTION_OPTIONS[i] for i in dist_indices)
            or DEFAULT_DISTRIBUTIONS
        )
        gev_method = self._GEV_METHOD_OPTIONS[
            self.parameterAsEnum(parameters, self.GEV_METHOD, context)
        ]
        min_completeness = self.parameterAsDouble(
            parameters, self.MIN_COMPLETENESS, context
        )
        min_years_warning = self.parameterAsInt(
            parameters, self.MIN_YEARS_WARNING, context
        )
        aep_str = self.parameterAsString(
            parameters, self.EXCEEDANCE_PROBABILITIES, context
        )

        try:
            exceedance_probabilities = tuple(
                float(x.strip()) for x in aep_str.split(",") if x.strip()
            )
        except ValueError:
            raise QgsProcessingException(
                f"Could not parse exceedance probabilities: {aep_str!r}"
            ) from None

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise QgsProcessingException(f"Could not read CSV: {e}") from e
        feedback.pushInfo(f"Loaded {len(df)} rows from {csv_path}")
        feedback.pushInfo(f"Columns found: {list(df.columns)}")
        feedback.pushInfo(f"Fitting: {', '.join(distributions)}")

        try:
            result = run_frequency_analysis(
                df,
                timestamp_col=timestamp_col,
                depth_col=depth_col,
                timestamp_format=timestamp_format,
                distributions=distributions,
                gev_method=gev_method,
                min_completeness=min_completeness,
                min_years_warning=min_years_warning,
                exceedance_probabilities=exceedance_probabilities,
            )
        except ValueError as e:
            raise QgsProcessingException(str(e)) from e

        for w in result.warnings:
            feedback.pushWarning(w)

        feedback.pushInfo(
            f"Native interval: {result.metadata.get('native_interval_min')} min"
        )
        feedback.pushInfo(
            f"{len(result.fits)} fit(s) computed across "
            f"{len(result.duration_series)} duration(s) x {len(distributions)} distribution(s)."
        )
        for rec in result.recommendations:
            feedback.pushInfo(
                f"  {rec.duration_label}: recommended = {rec.recommended_distribution} "
                f"(tau3={rec.tau3_sample:.3f}, tau4={rec.tau4_sample:.3f})"
            )

        outputs = {}

        params_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_PARAMETERS, context
        )
        write_parameters(result, params_path)
        outputs[self.OUTPUT_PARAMETERS] = params_path

        quantiles_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_QUANTILES, context
        )
        write_quantiles(result, quantiles_path)
        outputs[self.OUTPUT_QUANTILES] = quantiles_path

        ddf_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_RECOMMENDED_DDF, context
        )
        write_recommended_ddf(result, ddf_path)
        outputs[self.OUTPUT_RECOMMENDED_DDF] = ddf_path

        rec_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_RECOMMENDATIONS, context
        )
        if rec_path:
            write_recommendations(result, rec_path)
            outputs[self.OUTPUT_RECOMMENDATIONS] = rec_path

        ams_path = self.parameterAsFileOutput(parameters, self.OUTPUT_AMS, context)
        if ams_path:
            write_ams(result, ams_path)
            outputs[self.OUTPUT_AMS] = ams_path

        yc_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_YEAR_COMPLETENESS, context
        )
        if yc_path:
            write_year_completeness(result, yc_path)
            outputs[self.OUTPUT_YEAR_COMPLETENESS] = yc_path

        meta_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_METADATA, context
        )
        if meta_path:
            write_metadata(result, meta_path)
            outputs[self.OUTPUT_METADATA] = meta_path

        return outputs
