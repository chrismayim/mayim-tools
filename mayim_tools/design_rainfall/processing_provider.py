from qgis.core import QgsProcessingProvider

from .design_rainfall_algorithm import DesignRainfallPointAlgorithm


class DesignRainfallProvider(QgsProcessingProvider):
    def id(self):
        return "design_rainfall"

    def name(self):
        return "Design Rainfall (South Africa)"

    def loadAlgorithms(self):
        self.addAlgorithm(DesignRainfallPointAlgorithm())
