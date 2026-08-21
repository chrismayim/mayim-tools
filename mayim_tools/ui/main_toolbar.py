# -*- coding: utf-8 -*-
"""
Mayim Tools – Main Toolbar
Creates and manages the Mayim Tools toolbar in the QGIS main window.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QToolBar

from mayim_tools.core.logger import MayimLogger


class MayimToolbar:
    """
    Manages the Mayim Tools toolbar added to the QGIS main window.
    Provides quick-access buttons for each registered category.
    """

    TOOLBAR_NAME = "Mayim Tools Toolbar"

    def __init__(self, iface):
        """
        Constructor.

        :param iface: QGIS interface instance
        """
        self.iface = iface
        self.toolbar: QToolBar = None
        self.actions: list[QAction] = []

    def setup(self) -> None:
        """
        Create the toolbar and add it to the QGIS main window.
        Called from MayimToolsPlugin.initGui().
        """
        self.toolbar = self.iface.mainWindow().addToolBar(self.TOOLBAR_NAME)
        self.toolbar.setObjectName("MayimToolsToolbar")
        self.toolbar.setToolTip("Mayim Tools")

        # ── Add About action ──
        about_action = QAction(
            QIcon(":/icons/mayim_logo.png"),
            "Mayim Tools",
            self.iface.mainWindow(),
        )
        about_action.setToolTip("About Mayim Tools")
        about_action.triggered.connect(self._show_about)
        self.toolbar.addAction(about_action)
        self.actions.append(about_action)

        # ── Add separator ──
        self.toolbar.addSeparator()

        # ── Add category shortcut buttons ──
        self._add_category_actions()

        MayimLogger.info("Mayim Tools toolbar created.")

    def _add_category_actions(self) -> None:
        """
        Dynamically add a toolbar button for each registered category.
        """
        from mayim_tools.categories.category_registry import CategoryRegistry

        for category in CategoryRegistry.get_all():
            action = QAction(
                category.icon,
                category.name,
                self.iface.mainWindow(),
            )
            action.setToolTip(category.description)
            # Connect to open the Processing Toolbox filtered to this category
            action.triggered.connect(
                lambda checked, cat=category: self._open_category(cat)
            )
            self.toolbar.addAction(action)
            self.actions.append(action)

    def _open_category(self, category) -> None:
        """
        Open the Processing Toolbox and filter to the selected category.

        :param category: BaseCategory instance
        """
        self.iface.openMessageLog()
        MayimLogger.info(f"Opening category: {category.name}")
        # Future: filter Processing Toolbox to this category directly

    def _show_about(self) -> None:
        """Open the About dialog."""
        from mayim_tools.ui.about_dialog import AboutDialog
        dialog = AboutDialog(self.iface.mainWindow())
        dialog.exec()

    def teardown(self) -> None:
        """
        Remove the toolbar from the QGIS main window.
        Called from MayimToolsPlugin.unload().
        """
        for action in self.actions:
            self.toolbar.removeAction(action)
        if self.toolbar:
            self.iface.mainWindow().removeToolBar(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        MayimLogger.info("Mayim Tools toolbar removed.")
