from qgis.core import QgsApplication

from .processing_provider import HuffCurvesProvider


class HuffCurvesPlugin:
    """QGIS plugin entry point. Registers a Processing provider containing
    the Huff curve algorithm. Kept deliberately thin - all logic lives
    in huffrain/ (zero QGIS dependency), same principle as
    design_rainfall and grib_to_csv."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        self.provider = HuffCurvesProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
