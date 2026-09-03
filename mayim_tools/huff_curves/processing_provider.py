from qgis.core import QgsProcessingProvider

from .huff_curves_algorithm import HuffCurvesAlgorithm


class HuffCurvesProvider(QgsProcessingProvider):
    def id(self):
        return "huff_curves"

    def name(self):
        return "Huff Curves"

    def loadAlgorithms(self):
        self.addAlgorithm(HuffCurvesAlgorithm())
