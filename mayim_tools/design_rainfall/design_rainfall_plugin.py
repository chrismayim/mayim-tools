from qgis.core import QgsApplication

from .processing_provider import DesignRainfallProvider


class DesignRainfallPlugin:
    """QGIS plugin entry point. Registers a Processing provider containing
    the design rainfall estimation algorithm(s). Kept deliberately thin -
    all logic lives in core.py / the processing algorithm, both of which
    are independently testable outside QGIS."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        self.provider = DesignRainfallProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
