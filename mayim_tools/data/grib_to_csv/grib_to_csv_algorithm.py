"""Processing Toolbox algorithm: GRIB file in -> CSV out.

v0.2: rebuilt around xarray+cfgrib (eccodes) instead of GDAL/rasterio
after eccodes proved ~60x faster on real ERA5-scale files - see the
note at the top of core.py for the full diagnosis. The input parameter
changed from a raster layer (GDAL-centric) to a plain file parameter,
since the file is no longer opened as a GDAL raster at all.

Mirrors the design_rainfall plugin's structure: this file is a thin
QGIS wrapper, all real logic lives in core.py (zero QGIS dependency).
"""

import os
from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import single_output_path

from .core import grib_to_csv, read_variable_info

TOOL_VERSION = "0.3.0"


class GribToCsvAlgorithm(QgsProcessingAlgorithm):

    INPUT = "INPUT"
    VARIABLE = "VARIABLE"
    CONVERT_MM = "CONVERT_MM"
    DROP_NA = "DROP_NA"
    DECIMALS = "DECIMALS"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    FOLDER_CSV = "FOLDER_CSV"
    OUTPUT_CSV = "OUTPUT_CSV"

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return GribToCsvAlgorithm()

    def name(self):
        return "grib_to_csv"

    def displayName(self):
        return "GRIB to CSV"

    def group(self):
        return "Data Tools"

    def groupId(self):
        return "data_tools"

    def shortHelpString(self):
        return (
            f"GRIB to CSV (version {TOOL_VERSION})\n\n"
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
            "variables too.\n\n"
            "OUTPUT:\tChoose the CSV folder. The file is named <project "
            "number>_<GRIB file name>_<variable>.csv and replaces a file of "
            "the same name; a file that is open in Excel cannot be replaced "
            "and is reported. The CSV is UTF-8 with a byte-order mark (Excel "
            "shows it cleanly; pandas reads it directly)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT,
                "GRIB file",
                fileFilter="GRIB files (*.grib *.grib2 *.grb *.grb2);;All files (*.*)",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.VARIABLE,
                "Variable (leave empty for the first variable found)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CONVERT_MM,
                "Convert metre-based precipitation to millimetres",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DROP_NA,
                "Drop rows with no data for the selected variable",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DECIMALS,
                "Round values to this many decimal places "
                "(leave unset for full precision)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                maxValue=15,
                optional=True,
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
            QgsProcessingParameterFolderDestination(
                self.FOLDER_CSV, "Output folder for CSV files"
            )
        )
        self.addOutput(QgsProcessingOutputFile(self.OUTPUT_CSV, "Output CSV"))

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        input_path = self.parameterAsFile(parameters, self.INPUT, context)
        variable = self.parameterAsString(parameters, self.VARIABLE, context) or None
        convert_mm = self.parameterAsBoolean(parameters, self.CONVERT_MM, context)
        drop_na = self.parameterAsBoolean(parameters, self.DROP_NA, context)
        decimals = (
            self.parameterAsInt(parameters, self.DECIMALS, context)
            if parameters.get(self.DECIMALS) is not None
            else None
        )
        folder = self.parameterAsFileOutput(parameters, self.FOLDER_CSV, context)
        if not folder:
            raise QgsProcessingException("Choose the output folder for CSV files.")
        os.makedirs(folder, exist_ok=True)
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()

        try:
            var_info = read_variable_info(input_path)
        except Exception as e:
            feedback.reportError(f"Could not open {input_path}: {e}")
            raise
        feedback.pushInfo(f"{len(var_info)} variable(s) found in {input_path}")
        for v in var_info:
            marker = (
                " <- selected"
                if (variable == v.name) or (variable is None and v is var_info[0])
                else ""
            )
            feedback.pushInfo(
                f"  {v.name}: {v.long_name} [{v.units}] dims={v.dims}{marker}"
            )

        selected = variable or (var_info[0].name if var_info else "data")
        output_csv = single_output_path(
            folder, project_number, f"{Path(input_path).stem} {selected}"
        )

        def _progress(pct):
            if feedback.isCanceled():
                raise InterruptedError("Cancelled by user")
            feedback.setProgress(pct)

        try:
            n_rows = grib_to_csv(
                input_path,
                output_csv,
                variable=variable,
                convert_metres_to_mm=convert_mm,
                drop_na=drop_na,
                decimal_places=decimals,
                progress_callback=_progress,
            )
        except InterruptedError:
            feedback.pushInfo("Cancelled.")
            return {}
        except PermissionError as e:
            raise QgsProcessingException(
                f"{Path(output_csv).name} could not be written ({e}). If it is "
                "open in Excel, close it and run again."
            ) from e

        feedback.pushInfo(f"{n_rows} rows written to {output_csv}")
        return {self.OUTPUT_CSV: output_csv, self.FOLDER_CSV: folder}
