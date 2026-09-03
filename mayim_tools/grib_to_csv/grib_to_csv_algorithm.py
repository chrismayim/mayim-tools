"""Processing Toolbox algorithm: GRIB file in -> CSV out.

v0.2: rebuilt around xarray+cfgrib (eccodes) instead of GDAL/rasterio
after eccodes proved ~60x faster on real ERA5-scale files - see the
note at the top of core.py for the full diagnosis. The input parameter
changed from a raster layer (GDAL-centric) to a plain file parameter,
since the file is no longer opened as a GDAL raster at all.

Mirrors the design_rainfall plugin's structure: this file is a thin
QGIS wrapper, all real logic lives in core.py (zero QGIS dependency).
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

from .core import grib_to_csv, read_variable_info


class GribToCsvAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    VARIABLE = "VARIABLE"
    CONVERT_MM = "CONVERT_MM"
    DROP_NA = "DROP_NA"
    DECIMALS = "DECIMALS"
    OUTPUT_CSV = "OUTPUT_CSV"

    def createInstance(self):
        return GribToCsvAlgorithm()

    def name(self):
        return "grib_to_csv"

    def displayName(self):
        return "GRIB to CSV Export"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def shortHelpString(self):
        return (
            "Exports a variable from a GRIB file (ERA5 or similar "
            "reanalysis/forecast product) to CSV - one row per grid-cell/"
            "time combination, using the file's own dimensions (typically "
            "latitude, longitude, time, and often step/level). Uses "
            "eccodes (via xarray/cfgrib) to decode the file, not GDAL's "
            "built-in GRIB driver - GDAL is dramatically slower at "
            "ECMWF's typical complex-packed GRIB2 encoding, which matters "
            "a great deal on files with many messages (e.g. a multi-year "
            "hourly ERA5 extract). Leave 'Variable' empty to export the "
            "first variable found in the file. 'Convert metres to mm' is "
            "ERA5-specific: total precipitation (tp) is stored in metres "
            "by convention and this multiplies by 1000, relabelling units "
            "- it only acts on variables whose units are already metre-"
            "based, so it's safe to leave on for non-precipitation "
            "variables too."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT, "GRIB file",
                fileFilter="GRIB files (*.grib *.grib2 *.grb *.grb2);;All files (*.*)",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.VARIABLE, "Variable (leave empty for the first variable found)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CONVERT_MM, "Convert metre-based precipitation to millimetres",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DROP_NA, "Drop rows with no data for the selected variable",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DECIMALS, "Round values to this many decimal places (leave unset for full precision)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0, maxValue=15, optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV, "Output CSV", fileFilter="CSV files (*.csv)",
            )
        )

    def processAlgorithm(self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        input_path = self.parameterAsFile(parameters, self.INPUT, context)
        variable = self.parameterAsString(parameters, self.VARIABLE, context) or None
        convert_mm = self.parameterAsBoolean(parameters, self.CONVERT_MM, context)
        drop_na = self.parameterAsBoolean(parameters, self.DROP_NA, context)
        decimals = (
            self.parameterAsInt(parameters, self.DECIMALS, context)
            if parameters.get(self.DECIMALS) is not None else None
        )
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        try:
            var_info = read_variable_info(input_path)
        except Exception as e:
            feedback.reportError(f"Could not open {input_path}: {e}")
            raise
        feedback.pushInfo(f"{len(var_info)} variable(s) found in {input_path}")
        for v in var_info:
            marker = " <- selected" if (variable == v.name) or (variable is None and v is var_info[0]) else ""
            feedback.pushInfo(f"  {v.name}: {v.long_name} [{v.units}] dims={v.dims}{marker}")

        def _progress(pct):
            if feedback.isCanceled():
                raise InterruptedError("Cancelled by user")
            feedback.setProgress(pct)

        try:
            n_rows = grib_to_csv(
                input_path, output_csv,
                variable=variable, convert_metres_to_mm=convert_mm,
                drop_na=drop_na, decimal_places=decimals,
                progress_callback=_progress,
            )
        except InterruptedError:
            feedback.pushInfo("Cancelled.")
            return {}

        feedback.pushInfo(f"{n_rows} rows written to {output_csv}")
        return {self.OUTPUT_CSV: output_csv}
