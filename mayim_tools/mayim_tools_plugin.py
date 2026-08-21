# -*- coding: utf-8 -*-
"""
Mayim Tools – Main Plugin Class
Manages plugin lifecycle: initialisation, GUI setup, and cleanup.
"""

from qgis.core import QgsApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from mayim_tools.core.logger import MayimLogger
from mayim_tools.core.settings_manager import SettingsManager
from mayim_tools.processing.provider import MayimToolsProvider
from mayim_tools.ui.dock_widget import MayimDockWidget
from mayim_tools.ui.main_menu import MayimMenu
from mayim_tools.ui.main_toolbar import MayimToolbar


class MayimToolsPlugin:
    """Main plugin class — instantiated by QGIS via classFactory()."""

    PLUGIN_NAME = "Mayim Tools"

    def __init__(self, iface):
        """
        Constructor.

        :param iface: QGIS interface instance.
        :type iface: QgisInterface
        """
        self.iface = iface
        self.provider = None
        self.toolbar = None
        self.menu = None
        self.dock_widget = None

        MayimLogger.info(f"{self.PLUGIN_NAME} initialised.")

    def initGui(self) -> None:
        """Set up all GUI elements and register the Processing Provider."""

        # ── Register Processing Provider ──
        self.provider = MayimToolsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

        # ── Set up Toolbar ──
        self.toolbar = MayimToolbar(self.iface)
        self.toolbar.setup()

        # ── Set up Menu ──
        self.menu = MayimMenu(self.iface)
        self.menu.setup()

        # ── Set up Dock Widget ──
        self.dock_widget = MayimDockWidget(self.iface)
        self.dock_widget.setup()

        MayimLogger.info(f"{self.PLUGIN_NAME} GUI initialised.")

    def unload(self) -> None:
        """Remove all GUI elements and deregister the Processing Provider."""

        # ── Remove Processing Provider ──
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)

        # ── Remove Toolbar ──
        if self.toolbar:
            self.toolbar.teardown()

        # ── Remove Menu ──
        if self.menu:
            self.menu.teardown()

        # ── Remove Dock Widget ──
        if self.dock_widget:
            self.dock_widget.teardown()

        MayimLogger.info(f"{self.PLUGIN_NAME} unloaded.")
