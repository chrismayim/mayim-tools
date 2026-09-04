"""
Mayim Tools – About Dialog
Displays plugin version, author, and license information.
"""

from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon, QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from mayim_tools.resources_rc import get_icon_path


class AboutDialog(QDialog):
    """
    About dialog for Mayim Tools.
    Shows version, description, author, license, and repository link.
    """

    PLUGIN_NAME = "Mayim Tools"
    PLUGIN_VERSION = "0.2.0"
    PLUGIN_AUTHOR = "Chris Etsebeth / Mayim Consulting Engineers"
    PLUGIN_EMAIL = "chris@mayimconsulting.com"
    PLUGIN_LICENSE = "GNU General Public License v2.0 or later (GPL-2.0+)"
    PLUGIN_REPO = "https://github.com/chrismayim/mayim-tools"
    PLUGIN_DESC = (
        "Mayim Tools is a modular QGIS 4+ plugin providing a suite of "
        "engineering and geospatial processing tools. Organised into "
        "clearly defined categories, each tool is accessible via the "
        "Processing Toolbox, Graphical Modeler, and Python console."
    )

    def __init__(self, parent=None):
        """
        Constructor.

        :param parent: Parent QWidget (typically the QGIS main window)
        """
        super().__init__(parent)
        self.setWindowTitle(f"About {self.PLUGIN_NAME}")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the dialog layout and widgets."""

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ── Logo and plugin name header ──
        header_layout = QHBoxLayout()

        logo_label = QLabel()
        logo_label.setPixmap(
            QPixmap(get_icon_path("mayim_logo.png")).scaled(
                64,
                64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        header_layout.addWidget(logo_label)

        title_layout = QVBoxLayout()
        name_label = QLabel(f"<h2>{self.PLUGIN_NAME}</h2>")
        version_label = QLabel(f"<i>Version {self.PLUGIN_VERSION}</i>")
        title_layout.addWidget(name_label)
        title_layout.addWidget(version_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # ── Divider ──
        divider = QLabel("<hr>")
        main_layout.addWidget(divider)

        # ── Description ──
        desc_label = QLabel(self.PLUGIN_DESC)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        main_layout.addWidget(desc_label)

        # ── Details ──
        details = QLabel(
            f"<table>"
            f"<tr><td><b>Author:&nbsp;</b></td><td>{self.PLUGIN_AUTHOR}</td></tr>"
            f"<tr><td><b>Email:&nbsp;</b></td><td>{self.PLUGIN_EMAIL}</td></tr>"
            f"<tr><td><b>License:&nbsp;</b></td><td>{self.PLUGIN_LICENSE}</td></tr>"
            f"</table>"
        )
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        main_layout.addWidget(details)

        # ── Repository button ──
        repo_button = QPushButton(
            QIcon(get_icon_path("mayim_logo.png")), "  View on GitHub"
        )
        repo_button.setToolTip(self.PLUGIN_REPO)
        repo_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(self.PLUGIN_REPO))
        )
        main_layout.addWidget(repo_button)

        # ── Close button ──
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
