"""Processing Toolbox algorithm: DEM or D8 pointer raster in -> D8
flow accumulation raster out - an exact replication of WhiteboxTools'
own D8FlowAccumulation tool. See core.py's module docstring for the
precise, source-confirmed behavioural details this reproduces.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_d8_flow_accumulation_core.py).
"""

from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtGui import QIcon

from .core import compute_d8_flow_accumulation_from_file

OUT_TYPE_OPTIONS = ["Cells", "Catchment area", "Specific contributing area"]
OUT_TYPE_KEYS = ["cells", "ca", "sca"]


class D8FlowAccumulationAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = "INPUT_RASTER"
    IS_POINTER = "IS_POINTER"
    ESRI_STYLE = "ESRI_STYLE"
    OUT_TYPE = "OUT_TYPE"
    LOG_TRANSFORM = "LOG_TRANSFORM"
    OUTPUT = "OUTPUT"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return D8FlowAccumulationAlgorithm()

    def name(self):
        return "d8_flow_accumulation"

    def displayName(self):
        return "D8 Flow Accumulation (WBT)"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        return (
            "D8 Flow Accumulation (version 0.1.0)\n"
            "\n"
            "PURPOSE:\tCalculates a D8 flow-accumulation raster from a "
            "DEM or a pre-computed D8 pointer raster (e.g. from this "
            "suite's own D8 Flow Direction plugin) - an exact "
            "replication of WhiteboxTools' own D8FlowAccumulation "
            "tool, not a generic/independent implementation.\n"
            "\n"
            "METHOD:\t\tEvery valid cell starts with an accumulation "
            "of 1 (itself, not 0). Flow is then propagated downstream "
            "in topological order - each cell's own accumulated value "
            "is added to whatever cell it points to, once every one "
            "of ITS OWN upstream contributors has already been "
            "resolved. A cell with no downslope neighbour (a pit or "
            "outlet) does not propagate further. Three output types, "
            "confirmed exactly from WhiteboxTools' own source: Cells "
            "(raw contributing cell count, WhiteboxTools' own "
            "default), Catchment area (cell count x true cell area, "
            "e.g. m^2), and Specific contributing area (catchment "
            "area divided by a CONSTANT flow width - the average of "
            "the cell's X and Y size, deliberately NOT direction-"
            "dependent, since WhiteboxTools' own source explains a "
            "direction-dependent flow width breaks the property that "
            "accumulation must increase continuously downstream, "
            "which stream-network-extraction tools rely on).\n"
            "\n"
            "BACKGROUND:\tIMPORTANT, same precondition as D8 Flow "
            "Direction: if a raw DEM is supplied, it must already be "
            "hydrologically corrected (pits filled or breached, flats "
            "resolved) - this tool does not do that itself. A genuine "
            "interior pit (no downslope neighbour, and not simply at "
            "the edge of the valid data) is detected and reported as "
            "a warning, matching WhiteboxTools' own diagnostic - it "
            "usually means the input needs further conditioning. "
            "Output is always float, matching WhiteboxTools' own "
            "current behaviour (an older WhiteboxTools version had a "
            "real, reported bug where a 16-bit integer pointer's data "
            "type was inherited by the accumulation output, silently "
            "truncating large catchments - deliberately avoided here, "
            "not by accident).\n"
            "\n"
            "PARAMETERS:\n"
            "  Input raster: a DEM, or a D8 pointer raster if 'Input "
            "is a D8 pointer' is checked.\n"
            "  Input is a D8 pointer: off by default (input treated "
            "as a DEM).\n"
            "  Use ESRI pointer scheme: off by default (WhiteboxTools' "
            "own convention) - only relevant when the input is a "
            "pointer.\n"
            "  Output type: see METHOD above.\n"
            "  Log-transform output: off by default - useful for "
            "visualisation only; WhiteboxTools' own documentation "
            "warns a log-transformed output must not be used to "
            "compute secondary terrain indices (e.g. wetness index)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, "Input DEM or D8 pointer raster"
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.IS_POINTER,
                "Input is a D8 pointer (not a DEM)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ESRI_STYLE,
                "Use ESRI pointer scheme (pointer input only)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OUT_TYPE,
                "Output type",
                options=OUT_TYPE_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOG_TRANSFORM, "Log-transform output", defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "D8 Flow Accumulation")
        )

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        input_layer = self.parameterAsRasterLayer(
            parameters, self.INPUT_RASTER, context
        )
        if input_layer is None:
            raise QgsProcessingException("No input raster provided.")

        is_pointer = self.parameterAsBoolean(parameters, self.IS_POINTER, context)
        esri_style = self.parameterAsBoolean(parameters, self.ESRI_STYLE, context)
        out_type_idx = self.parameterAsEnum(parameters, self.OUT_TYPE, context)
        out_type = OUT_TYPE_KEYS[out_type_idx]
        log_transform = self.parameterAsBoolean(parameters, self.LOG_TRANSFORM, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        input_path = input_layer.source()
        feedback.pushInfo(f"Input: {input_path}")
        feedback.pushInfo(f"Input type: {'D8 pointer' if is_pointer else 'DEM'}")
        feedback.pushInfo(
            f"Pointer scheme: {'ESRI' if esri_style else 'WhiteboxTools default'}"
        )
        feedback.pushInfo(f"Output type: {OUT_TYPE_OPTIONS[out_type_idx]}")
        feedback.setProgress(10)

        try:
            info = compute_d8_flow_accumulation_from_file(
                input_path,
                output_path,
                out_type=out_type,
                log_transform=log_transform,
                pntr_input=is_pointer,
                esri_style=esri_style,
            )
        except Exception as e:
            raise QgsProcessingException(
                f"D8 flow accumulation computation failed: {e}"
            ) from e

        feedback.setProgress(100)

        if info["interior_pit_count"] > 0:
            feedback.pushWarning(
                f"{info['interior_pit_count']} interior pit cell(s) found - cells "
                f"with no downslope neighbour that are not simply at the edge of "
                f"the valid data. This usually means the input needs further "
                f"hydrological conditioning (depression filling/breaching) before "
                f"this output can be trusted, matching WhiteboxTools' own "
                f"diagnostic for this situation."
            )

        feedback.pushInfo(f"D8 flow accumulation raster written to {output_path}")
        return {self.OUTPUT: output_path}
