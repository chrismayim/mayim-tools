"""
Mayim Tools - Main Plugin Class
Manages plugin lifecycle: initialisation, GUI setup, and cleanup.
"""

from qgis.core import QgsApplication

from mayim_tools.core.logger import MayimLogger
from mayim_tools.processing.provider import MayimToolsProvider
from mayim_tools.ui.dock_widget import MayimDockWidget


class MayimToolsPlugin:
    """Main plugin class - instantiated by QGIS via classFactory()."""

    PLUGIN_NAME = "Mayim Tools"

    def __init__(self, iface):
        """
        Constructor.

        :param iface: QGIS interface instance.
        """
        self.iface = iface
        self.provider = None
        self.dock_widget = None

        MayimLogger.info(f"{self.PLUGIN_NAME} initialised.")

    def initGui(self) -> None:
        """Set up GUI elements and register the Processing Provider."""
        try:
            self._cleanup()

            # Register Processing Provider
            self.provider = MayimToolsProvider()
            QgsApplication.processingRegistry().addProvider(self.provider)

            # Set up Dock Widget
            self.dock_widget = MayimDockWidget(self.iface)
            self.dock_widget.setup()

            MayimLogger.info(f"{self.PLUGIN_NAME} GUI initialised.")

        except Exception as e:
            MayimLogger.critical(
                f"{self.PLUGIN_NAME} failed to initialise GUI: {e}"
            )
            raise

    def _cleanup(self) -> None:
        """
        Internal cleanup - removes GUI elements safely.
        Called before initGui() and during unload().
        """
        try:
            if self.provider:
                QgsApplication.processingRegistry().removeProvider(
                    self.provider
                )
                self.provider = None

            if self.dock_widget:
                self.dock_widget.teardown()
                self.dock_widget = None

        except Exception as e:
            MayimLogger.warning(
                f"{self.PLUGIN_NAME} cleanup warning: {e}"
            )

    def unload(self) -> None:
        """Remove GUI elements and deregister the Processing Provider."""
        self._cleanup()
        MayimLogger.info(f"{self.PLUGIN_NAME} unloaded.")
