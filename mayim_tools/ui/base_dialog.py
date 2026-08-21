# -*- coding: utf-8 -*-
"""
Mayim Tools – Base Dialog
Abstract base class for all custom tool dialogs in Mayim Tools.
Provides consistent styling, layout helpers, and shared behaviour.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

from mayim_tools.core.logger import MayimLogger


class MayimBaseDialog(QDialog):
    """
    Base class for all Mayim Tools custom dialogs.
    Inherit from this class when building tool-specific dialogs.

    Usage:
        class MyCatchmentDialog(MayimBaseDialog):
            def __init__(self, parent=None):
                super().__init__(
                    parent=parent,
                    title="Catchment Delineation",
                    description="Delineate catchment areas from a DEM."
                )
                self._build_tool_ui()
    """

    def __init__(
        self,
        parent=None,
        title: str = "Mayim Tool",
        description: str = "",
    ):
        """
        Constructor.

        :param parent: Parent QWidget
        :param title: Dialog window title
        :param description: Short description shown at the top of the dialog
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.setModal(True)

        # ── Main layout ──
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(16, 16, 16, 16)

        # ── Description label ──
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #555555; font-size: 11px;")
            self.main_layout.addWidget(desc_label)

            divider = QLabel("<hr>")
            self.main_layout.addWidget(divider)

        # ── Tool content area (populated by subclass) ──
        self.content_layout = QVBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # ── Standard OK / Cancel buttons ──
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box)

        self.setLayout(self.main_layout)

    def _on_accept(self) -> None:
        """
        Called when the user clicks OK.
        Override in subclass to add validation before accepting.
        """
        if self.validate():
            self.accept()

    def validate(self) -> bool:
        """
        Validate dialog inputs before accepting.
        Override in subclass to add tool-specific validation logic.

        :returns: True if inputs are valid, False otherwise
        """
        return True

    def show_error(self, message: str) -> None:
        """
        Display an error message at the bottom of the dialog.

        :param message: Error message string
        """
        from qgis.PyQt.QtWidgets import QMessageBox
        MayimLogger.warning(message)
        QMessageBox.critical(self, "Mayim Tools — Input Error", message)

    def show_info(self, message: str) -> None:
        """
        Display an informational message.

        :param message: Info message string
        """
        from qgis.PyQt.QtWidgets import QMessageBox
        MayimLogger.info(message)
        QMessageBox.information(self, "Mayim Tools", message)
