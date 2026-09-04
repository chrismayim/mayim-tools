"""
Mayim Tools – Main Menu
Adds Mayim Tools entries to the QGIS Plugins menu.
Structure: Plugins > Mayim Tools > [Categories] > [Tools]
"""

import processing
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu

from mayim_tools.core.logger import MayimLogger
from mayim_tools.resources_rc import get_icon_path


class MayimMenu:
    """
    Manages the Mayim Tools entry in the QGIS Plugins menu.
    """

    MENU_TITLE = "Mayim Tools"

    def __init__(self, iface):
        """
        Constructor.

        :param iface: QGIS interface instance
        """
        self.iface = iface
        self.menu: QMenu = None
        self.actions: list[QAction] = []

    def setup(self) -> None:
        """
        Build and insert the Mayim Tools menu into the QGIS Plugins menu.
        Called from MayimToolsPlugin.initGui().
        """
        # ── Create the top-level Mayim Tools menu ──
        self.menu = QMenu(self.MENU_TITLE, self.iface.mainWindow())
        self.menu.setIcon(QIcon(get_icon_path("mayim_logo.png")))

        # ── Add category submenus ──
        self._add_category_menus()

        # ── Add separator and About ──
        self.menu.addSeparator()
        about_action = QAction(
            QIcon(get_icon_path("mayim_logo.png")),
            "About Mayim Tools...",
            self.iface.mainWindow(),
        )
        about_action.triggered.connect(self._show_about)
        self.menu.addAction(about_action)
        self.actions.append(about_action)

        # ── Insert into QGIS Plugins menu ──
        self.iface.pluginMenu().addMenu(self.menu)

        MayimLogger.info("Mayim Tools menu created.")

    def _add_category_menus(self) -> None:
        """
        Dynamically add a submenu for each registered category,
        with a menu item for each tool in that category.
        """

        from mayim_tools.categories.category_registry import CategoryRegistry

        for category in CategoryRegistry.get_all():
            category_menu = QMenu(category.name, self.menu)
            category_menu.setIcon(category.icon)

            # Get algorithms for this category
            algorithms = category.get_algorithms()

            if algorithms:
                for algorithm in algorithms:
                    # Create a menu action for each tool
                    tool_action = QAction(
                        algorithm.displayName(),
                        category_menu,
                    )
                    tool_action.setToolTip(algorithm.shortHelpString())

                    # Capture algorithm name in closure correctly
                    def make_handler(alg_name: str):
                        def handler():
                            try:
                                alg_id = f"mayimtools:{alg_name}"
                                processing.execAlgorithmDialog(alg_id)
                            except Exception as e:
                                MayimLogger.critical(
                                    f"Failed to open tool " f"{alg_name}: {e}"
                                )

                        return handler

                    tool_action.triggered.connect(make_handler(algorithm.name()))
                    category_menu.addAction(tool_action)
                    self.actions.append(tool_action)
            else:
                # Placeholder when no tools exist yet
                placeholder = QAction(
                    "No tools available yet",
                    category_menu,
                )
                placeholder.setEnabled(False)
                category_menu.addAction(placeholder)

            self.menu.addMenu(category_menu)

    def _show_about(self) -> None:
        """Open the About dialog."""
        from mayim_tools.ui.about_dialog import AboutDialog

        dialog = AboutDialog(self.iface.mainWindow())
        dialog.exec()

    def teardown(self) -> None:
        """
        Remove the Mayim Tools menu from the QGIS Plugins menu.
        Called from MayimToolsPlugin.unload().
        """
        self.iface.pluginMenu().removeAction(self.menu.menuAction())
        MayimLogger.info("Mayim Tools menu removed.")
