"""Processing Toolbox algorithm: NetCDF4 file in -> a table (and log
output) of its variables, their dimensions, shape, units, and
description - a discovery step for NetCDF4 to CSV, so a variable name
doesn't need to be guessed or found by triggering an error first.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_netcdf4_to_csv_core.py).
"""

from pathlib import Path

import pandas as pd
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
)
from qgis.PyQt.QtGui import QIcon

from .core import describe_variables


class Netcdf4ListVariablesAlgorithm(QgsProcessingAlgorithm):
    INPUT_NC = "INPUT_NC"
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
            "for a quick look without opening the file)."
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
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Variable list",
                fileFilter="CSV files (*.csv)",
            )
        )

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        input_path = self.parameterAsFile(parameters, self.INPUT_NC, context)
        output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

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
        df.to_csv(output_path, index=False)
        return {self.OUTPUT_CSV: output_path}
