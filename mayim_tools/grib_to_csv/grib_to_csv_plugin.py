from qgis.core import QgsApplication

from .processing_provider import GribToCsvProvider


class GribToCsvPlugin:
    """QGIS plugin entry point. Registers a Processing provider containing
    the GRIB-to-CSV algorithm. Kept deliberately thin - all logic lives
    in core.py, which is independently testable outside QGIS. Mirrors
    the design_rainfall plugin's structure for consistency."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        self.provider = GribToCsvProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
