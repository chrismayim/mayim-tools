# -*- coding: utf-8 -*-
"""
Mayim Tools – Dock Widget
A dockable side panel providing a category browser and quick-access tools.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mayim_tools.core.logger import MayimLogger


class MayimDockWidget:
    """
    Manages the Mayim Tools dockable side panel.
    Displays a category and tool browser tree.
    """

    DOCK_TITLE = "Mayim Tools"

    def __init__(self, iface):
        """
        Constructor.

        :param iface: QGIS interface instance
        """
        self.iface = iface
        self.dock = None

    def setup(self) -> None:
        """
        Create and add the dock widget to the QGIS main window.
        Called from MayimToolsPlugin.initGui().
        """
        try:
            # ── Get the main window safely ──
            main_window = self.iface.mainWindow()
            if main_window is None:
                MayimLogger.warning(
                    "Mayim Tools: Main window not available — "
                    "dock widget will not be created."
                )
                return

            # ── Create the dock widget ──
            self.dock = QDockWidget(self.DOCK_TITLE, main_window)

            if self.dock is None:
                MayimLogger.warning(
                    "Mayim Tools: Failed to create dock widget."
                )
                return

            self.dock.setObjectName("MayimToolsDockWidget")
            self.dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea |
                Qt.DockWidgetArea.RightDockWidgetArea
            )

            # ── Build the inner widget and layout ──
            container = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(6)

            # ── Header label ──
            header = QLabel("Mayim Tools")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet(
                "font-weight: bold; font-size: 14px; padding: 6px;"
            )
            layout.addWidget(header)

            # ── Category & Tool browser tree ──
            self.tree = QTreeWidget()
            self.tree.setHeaderLabel("Categories & Tools")
            self.tree.setColumnCount(1)
            self.tree.setAnimated(True)
            self.tree.setRootIsDecorated(True)
            self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
            self._populate_tree()
            layout.addWidget(self.tree)

            # ── Footer label ──
            footer = QLabel("Double-click a tool to open it.")
            footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            footer.setStyleSheet(
                "font-size: 10px; color: grey; padding: 4px;"
            )
            layout.addWidget(footer)

            container.setLayout(layout)
            self.dock.setWidget(container)

            # ── Add dock to the QGIS main window ──
            main_window.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea,
                self.dock,
            )

            MayimLogger.info("Mayim Tools dock widget created.")

        except Exception as e:
            MayimLogger.critical(f"Mayim Tools: Dock widget setup failed: {e}")

    def _populate_tree(self) -> None:
        """
        Populate the tree widget with all registered categories and tools.
        """
        try:
            from mayim_tools.categories.category_registry import CategoryRegistry

            self.tree.clear()
            categories = CategoryRegistry.get_all()

            if not categories:
                empty_item = QTreeWidgetItem(["No categories registered yet."])
                self.tree.addTopLevelItem(empty_item)
                return

            for category in categories:
                # ── Top-level category node ──
                category_item = QTreeWidgetItem([category.name])
                category_item.setIcon(0, category.icon)
                category_item.setToolTip(0, category.description)
                category_item.setData(
                    0, Qt.ItemDataRole.UserRole, category
                )

                # ── Child nodes for each algorithm ──
                algorithms = category.get_algorithms()
                if algorithms:
                    for algorithm in algorithms:
                        tool_item = QTreeWidgetItem([algorithm.displayName()])
                        tool_item.setToolTip(0, algorithm.shortHelpString())
                        tool_item.setData(
                            0, Qt.ItemDataRole.UserRole, algorithm
                        )
                        category_item.addChild(tool_item)
                else:
                    placeholder = QTreeWidgetItem(["No tools yet..."])
                    placeholder.setDisabled(True)
                    category_item.addChild(placeholder)

                self.tree.addTopLevelItem(category_item)

            # ── Expand all categories by default ──
            self.tree.expandAll()

        except Exception as e:
            MayimLogger.critical(f"Mayim Tools: Tree population failed: {e}")

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """
        Handle double-click on a tree item.
        """
        try:
            import processing
            from qgis.core import QgsProcessingAlgorithm

            data = item.data(0, Qt.ItemDataRole.UserRole)

            if isinstance(data, QgsProcessingAlgorithm):
                processing.execAlgorithmDialog(data.id())
                MayimLogger.info(f"Opened tool: {data.displayName()}")
            else:
                item.setExpanded(not item.isExpanded())

        except Exception as e:
            MayimLogger.critical(f"Mayim Tools: Failed to open tool: {e}")

    def refresh(self) -> None:
        """Refresh the tree widget."""
        if self.tree:
            self._populate_tree()
            MayimLogger.info("Mayim Tools dock widget refreshed.")

    def teardown(self) -> None:
        """
        Remove the dock widget from the QGIS main window.
        Called from MayimToolsPlugin.unload().
        """
        try:
            if self.dock:
                self.iface.mainWindow().removeDockWidget(self.dock)
                self.dock.deleteLater()
                self.dock = None
            MayimLogger.info("Mayim Tools dock widget removed.")
        except Exception as e:
            MayimLogger.critical(f"Mayim Tools: Dock teardown failed: {e}")
