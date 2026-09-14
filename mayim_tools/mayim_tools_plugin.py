"""
Mayim Tools - Main Plugin Class
=================================

Minimal QGIS plugin entry point. This clean rebuild starts with no
registered Processing algorithms. Tool categories are added back
deliberately, one at a time, as they are redesigned.
"""

from __future__ import annotations

from qgis.core import QgsApplication


class MayimToolsPlugin:
    """
    Main Mayim Tools plugin class.

    QGIS instantiates this class through ``classFactory`` and calls
    ``initGui`` when the plugin is loaded, and ``unload`` when it is
    unloaded.
    """

    def __init__(self, iface) -> None:
        """
        Store the QGIS interface for later use.

        Parameters
        ----------
        iface:
            QGIS application interface.
        """
        self.iface = iface
        self.provider = None

    def initGui(self) -> None:  # noqa: N802 - QGIS requires this exact name.
        """
        Called by QGIS once the plugin is loaded.

        Registers the Mayim Tools Processing provider. The provider
        currently registers no algorithms; tool categories are added
        back deliberately during the rebuild.
        """
        from mayim_tools.processing.provider import MayimToolsProvider

        self.provider = MayimToolsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self) -> None:
        """
        Called by QGIS when the plugin is unloaded.

        Unregisters the Mayim Tools Processing provider.
        """
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
