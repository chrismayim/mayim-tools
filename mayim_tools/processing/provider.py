# -*- coding: utf-8 -*-
"""
Mayim Tools – Processing Provider
Registers Mayim Tools with the QGIS Processing Framework,
making all tools available in the Toolbox, Modeler, and Python console.
"""

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

# Import categories package to trigger all self-registrations
import mayim_tools.categories  # noqa: F401
from mayim_tools.categories.category_registry import CategoryRegistry
from mayim_tools.core.logger import MayimLogger


class MayimToolsProvider(QgsProcessingProvider):
    """
    QGIS Processing Provider for Mayim Tools.
    Exposes all registered category algorithms to the Processing Framework.
    """

    PROVIDER_ID = "mayimtools"
    PROVIDER_NAME = "Mayim Tools"

    def id(self) -> str:
        """Unique provider ID — used in processing.run('mayimtools:toolname')"""
        return self.PROVIDER_ID

    def name(self) -> str:
        """Display name shown in the Processing Toolbox."""
        return self.PROVIDER_NAME

    def longName(self) -> str:
        """Extended name shown in the Processing Toolbox header."""
        return "Mayim Tools — Engineering & Geospatial Plugin"

    def icon(self) -> QIcon:
        """Provider icon shown in the Processing Toolbox."""
        from mayim_tools.resources_rc import get_icon_path
        return QIcon(get_icon_path("mayim_logo.png"))

    def loadAlgorithms(self) -> None:
        """
        Load all algorithms from all registered categories.
        Called by QGIS when the provider is activated.
        This method dynamically pulls algorithms from the CategoryRegistry,
        meaning no hard-coded algorithm list is needed here.
        """
        algorithms = CategoryRegistry.get_all_algorithms()

        if not algorithms:
            MayimLogger.info(
                "Mayim Tools: No algorithms registered yet. "
                "Add tools to category get_algorithms() methods."
            )
            return

        for algorithm in algorithms:
            self.addAlgorithm(algorithm)
            MayimLogger.info(f"Algorithm loaded: {algorithm.displayName()}")

    def supportedOutputRasterLayerExtensions(self) -> list[str]:
        """Declare supported raster output formats."""
        return ["tif", "tiff"]

    def supportedOutputVectorLayerExtensions(self) -> list[str]:
        """Declare supported vector output formats."""
        return ["gpkg", "shp", "geojson"]

