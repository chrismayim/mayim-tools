"""
Mayim Tools - Main Toolbar
Creates and manages the Mayim Tools toolbar in the QGIS main window.
"""

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from mayim_tools.core.logger import MayimLogger
from mayim_tools.resources_rc import get_icon_path


class MayimToolbar:
    """
    Manages the Mayim Tools toolbar added to the QGIS main window.
    Provides quick-access buttons for the plugin and each category.
    """

    TOOLBAR_NAME = "Mayim Tools Toolbar"

    def __init__(self, iface):
        """
        Constructor.

        :param iface: QGIS interface instance
        """
        self.iface = iface
        self.toolbar = None
        self.actions = []

    def setup(self) -> None:
        """
        Create the toolbar and add it to the QGIS main window.
        Called from MayimToolsPlugin.initGui().
        """
        self.toolbar = self.iface.mainWindow().addToolBar(self.TOOLBAR_NAME)
        self.toolbar.setObjectName("MayimToolsToolbar")
        self.toolbar.setToolTip("Mayim Tools")

        # ── Mayim Tools branding button ────────────────────────────────── #
        about_action = QAction(
            QIcon(get_icon_path("mayim_logo.png")),
            "Mayim Tools",
            self.iface.mainWindow(),
        )
        about_action.setToolTip("About Mayim Tools")
        about_action.triggered.connect(self._show_about)
        self.toolbar.addAction(about_action)
        self.actions.append(about_action)

        # ── Separator ─────────────────────────────────────────────────── #
        self.toolbar.addSeparator()

        # ── Category buttons ───────────────────────────────────────────── #
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
            action.setToolTip(f"{category.name}\n{category.description}")

            def make_handler(cat):
                def handler():
                    self._open_category(cat)

                return handler

            action.triggered.connect(make_handler(category))
            self.toolbar.addAction(action)
            self.actions.append(action)

    def _open_category(self, category) -> None:
        """
        Open the first tool in the category via the Processing dialog.
        Shows an informational message if no tools are available yet.
        """
        try:
            import processing

            algorithms = category.get_algorithms()
            if algorithms:
                alg_id = f"mayimtools:{algorithms[0].name()}"
                processing.execAlgorithmDialog(alg_id)
                MayimLogger.info(f"Opened tool: {algorithms[0].displayName()}")
            else:
                from qgis.PyQt.QtWidgets import QMessageBox

                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Mayim Tools",
                    f"{category.name} has no tools yet.\n\n"
                    f"Tools are being developed and will appear "
                    f"here in a future release.",
                )
        except Exception as e:
            MayimLogger.critical(f"Failed to open category {category.name}: {e}")

    def _show_about(self) -> None:
        """Open the About dialog."""
        try:
            from mayim_tools.ui.about_dialog import AboutDialog

            dialog = AboutDialog(self.iface.mainWindow())
            dialog.exec()
        except Exception as e:
            MayimLogger.critical(f"Failed to open About dialog: {e}")

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
        self.actions = []
        MayimLogger.info("Mayim Tools toolbar removed.")
