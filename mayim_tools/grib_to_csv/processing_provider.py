from qgis.core import QgsProcessingProvider

from .grib_to_csv_algorithm import GribToCsvAlgorithm


class GribToCsvProvider(QgsProcessingProvider):
    def id(self):
        return "grib_to_csv"

    def name(self):
        return "GRIB to CSV Export"

    def loadAlgorithms(self):
        self.addAlgorithm(GribToCsvAlgorithm())
