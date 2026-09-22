"""Processing Toolbox algorithm: DEM raster in -> D8 flow direction
(pointer) raster out - an exact replication of WhiteboxTools' own
D8Pointer tool. See core.py's module docstring for the precise,
source-confirmed behavioural details this reproduces.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_core.py).
"""

from pathlib import Path

from qgis.core import (
    QgsPalettedRasterRenderer,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
)
from qgis.PyQt.QtGui import QColor, QIcon

from .core import compute_d8_pointer_from_file, direction_palette


class D8FlowDirectionAlgorithm(QgsProcessingAlgorithm):

    INPUT_DEM = "INPUT_DEM"
    ESRI_STYLE = "ESRI_STYLE"
    OUTPUT = "OUTPUT"

    def createInstance(self):
        return D8FlowDirectionAlgorithm()

    def name(self):
        return "d8_flow_direction"

    def displayName(self):
        return "D8 Flow Direction (WBT)"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def shortHelpString(self):
        return (
            "D8 Flow Direction (version 0.1.0)\n"
            "\n"
            "PURPOSE:\tCalculates a D8 flow-direction (pointer) raster "
            "from a DEM - an exact replication of WhiteboxTools' own "
            "D8Pointer tool, not a generic/independent D8 "
            "implementation.\n"
            "\n"
            "METHOD:\t\tFor each cell, the 8 neighbours are evaluated "
            "in a fixed order (NE, E, SE, S, SW, W, NW, N); a "
            "neighbour is only a candidate if it is STRICTLY lower "
            "than the centre cell, and diagonal neighbours use the "
            "true diagonal distance (accounting for non-square "
            "pixels), not the same distance as cardinal neighbours. "
            "Ties are broken by this fixed iteration order - the "
            "first direction to reach the steepest slope wins, not an "
            "arbitrary one. Every one of these details (including the "
            "exact numeric direction codes) was confirmed directly "
            "against WhiteboxTools' own source code, not inferred "
            "from general D8 knowledge or documentation alone.\n"
            "\n"
            "BACKGROUND:\tIMPORTANT, stated as plainly as WhiteboxTools "
            "states it: this tool does NOT fill depressions or "
            "resolve flat areas - the input DEM must already be "
            "hydrologically corrected (pits filled or breached, flats "
            "resolved) before running this tool, or unresolved pits "
            "and flats will simply produce a direction of 0 there, "
            "not a connected flow path through them. Direction codes "
            "(default WhiteboxTools scheme): NE=1, E=2, SE=4, S=8, "
            "SW=16, W=32, NW=64, N=128, with 0 meaning 'no lower "
            "neighbour found' (a pit, an unresolved flat, or an edge "
            "cell with nowhere lower to go) - not NoData. NoData input "
            "cells produce NoData output (-32768, matching "
            "WhiteboxTools' own output exactly), and this is distinct "
            "from the 0 case.\n"
            "\n"
            "PARAMETERS:\n"
            "  Input DEM: should already be hydrologically corrected "
            "- see BACKGROUND above.\n"
            "  Use ESRI pointer scheme: off by default (WhiteboxTools' "
            "own numeric convention); enable to match ArcGIS's "
            "Flow Direction convention instead (E=1, SE=2, S=4, SW=8, "
            "W=16, NW=32, N=64, NE=128)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM, "Input DEM (hydrologically corrected)"
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ESRI_STYLE,
                "Use ESRI pointer scheme instead of WhiteboxTools' default",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT, "D8 Flow Direction")
        )

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        if dem_layer is None:
            raise QgsProcessingException("No input DEM provided.")

        esri_style = self.parameterAsBoolean(parameters, self.ESRI_STYLE, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        dem_path = dem_layer.source()
        feedback.pushInfo(f"Input DEM: {dem_path}")
        feedback.pushInfo(
            f"Pointer scheme: {'ESRI' if esri_style else 'WhiteboxTools default'}"
        )

        feedback.setProgress(10)
        try:
            compute_d8_pointer_from_file(dem_path, output_path, esri_style=esri_style)
        except Exception as e:
            raise QgsProcessingException(
                f"D8 flow direction computation failed: {e}"
            ) from e
        feedback.setProgress(100)

        feedback.pushInfo(f"D8 flow direction raster written to {output_path}")

        # Stashed for postProcessAlgorithm() below, which runs after this
        # method returns and applies the default colour/label styling -
        # esri_style must be carried over since the correct value->colour
        # mapping depends on which pointer scheme this specific run used.
        self._output_path = output_path
        self._esri_style = esri_style

        return {self.OUTPUT: output_path}

    def postProcessAlgorithm(self, context, feedback):
        """Applies this tool's default colour scheme and direction
        labels to the output raster automatically, so every run is
        styled without the user having to apply it manually - the
        same mechanism QGIS's own built-in raster analysis algorithms
        use. The value->colour mapping is resolved for whichever
        pointer scheme (default or ESRI) this specific run actually
        used - see core.direction_palette()."""
        output_layer = QgsProcessingUtils.mapLayerFromString(self._output_path, context)
        if output_layer is not None:
            classes = [
                QgsPalettedRasterRenderer.Class(value, QColor(color_hex), label)
                for value, color_hex, label in direction_palette(self._esri_style)
            ]
            renderer = QgsPalettedRasterRenderer(
                output_layer.dataProvider(), 1, classes
            )
            output_layer.setRenderer(renderer)
            output_layer.triggerRepaint()
        return {self.OUTPUT: self._output_path}
