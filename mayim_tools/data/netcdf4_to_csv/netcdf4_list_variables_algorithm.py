"""Processing Toolbox algorithm: NetCDF4 file in -> a table (and log
output) of its variables, their dimensions, shape, units, and
description - a discovery step for NetCDF4 to CSV, so a variable name
doesn't need to be guessed or found by triggering an error first.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_netcdf4_to_csv_core.py).
"""

import os
from pathlib import Path

import pandas as pd
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import single_output_path

from .core import describe_variables

TOOL_VERSION = "0.2.0"


class Netcdf4ListVariablesAlgorithm(QgsProcessingAlgorithm):
    INPUT_NC = "INPUT_NC"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    FOLDER_CSV = "FOLDER_CSV"
    OUTPUT_CSV = "OUTPUT_CSV"

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return Netcdf4ListVariablesAlgorithm()

    def name(self):
        return "netcdf4_list_variables"

    def displayName(self):
        return "List NetCDF4 Variables"

    def group(self):
        return "Data Tools"

    def groupId(self):
        return "data_tools"

    def shortHelpString(self):
        return (
            f"List NetCDF4 Variables (version {TOOL_VERSION})\n\n"
            "Lists every data variable in a NetCDF4 file, with its "
            "dimensions, shape, units, and a human-readable description "
            "- a discovery step for the NetCDF4 to CSV tool, so its "
            "Variable parameter doesn't need to be guessed or found by "
            "first triggering an error.\n"
            "\n"
            "Units and description are read from each variable's own "
            "attributes when present (the 'units' attribute, and "
            "'long_name' or 'standard_name' - whichever is present, "
            "long_name preferred as the more human-readable of the two) "
            "- genuinely blank for a non-CF-compliant file that doesn't "
            "carry them, rather than guessed at.\n"
            "\n"
            "PARAMETERS:\n"
            "  Input NetCDF4 file: the file to inspect.\n"
            "  Output: a small CSV table (also logged directly below, "
            "for a quick look without opening the file), written to the CSV "
            "folder as <project number>_<NetCDF file name>_Variables.csv. It "
            "replaces a file of the same name; a file that is open in Excel "
            "cannot be replaced and is reported. UTF-8 with a byte-order mark."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_NC,
                "Input NetCDF4 file",
                extension="nc",
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
        self.addOutput(QgsProcessingOutputFile(self.OUTPUT_CSV, "Variable list"))

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        input_path = self.parameterAsFile(parameters, self.INPUT_NC, context)
        folder = self.parameterAsFileOutput(parameters, self.FOLDER_CSV, context)
        if not folder:
            raise QgsProcessingException("Choose the output folder for CSV files.")
        os.makedirs(folder, exist_ok=True)
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()
        output_path = single_output_path(
            folder, project_number, f"{Path(input_path).stem} Variables"
        )

        feedback.pushInfo(f"Input file: {input_path}")

        try:
            rows = describe_variables(input_path)
        except Exception as e:
            raise QgsProcessingException(f"Could not open input file: {e}") from e

        if not rows:
            feedback.pushWarning("No data variables found in this file.")
        else:
            feedback.pushInfo(f"{len(rows)} variable(s) found:")
            for row in rows:
                units = f" ({row['Units']})" if row["Units"] else ""
                desc = f" - {row['Description']}" if row["Description"] else ""
                feedback.pushInfo(
                    f"  {row['Variable']}{units}: dims [{row['Dimensions']}], "
                    f"shape {row['Shape']}{desc}"
                )

        df = pd.DataFrame(
            rows, columns=["Variable", "Dimensions", "Shape", "Units", "Description"]
        )
        try:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        except PermissionError as e:
            raise QgsProcessingException(
                f"{Path(output_path).name} could not be written ({e}). If it is "
                "open in Excel, close it and run again."
            ) from e
        feedback.pushInfo(f"Written: {output_path}")
        return {self.OUTPUT_CSV: output_path, self.FOLDER_CSV: folder}
