"""
Draw-on-map input for the section line(s) of Flood Extent Terrain Profile.

The algorithm's section parameter is a QgsProcessingParameterGeometry
(line). In the normal Processing dialog this module replaces its plain WKT
text box with a panel like the point picker of Design Rainfall: press
"Draw section", the dialog steps aside, click the vertices on the map
(snapping is honoured), right-click or press Enter to finish, and the
dialog comes back. Each drawn line is one section (A, B, C ... in drawing
order). Backspace removes the last vertex, Esc cancels the line.

The value is "CRS=<authid>;MultiLineString (...)" in the map canvas CRS -
the same text can be typed or pasted, and it is what batch mode and the
model designer use (plain text box there).
"""

from __future__ import annotations

from qgis.core import Qgis, QgsCoordinateTransform, QgsGeometry, QgsProject
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:  # the Processing plugin's Python widget wrappers
    from processing.gui.wrappers import DIALOG_STANDARD, WidgetWrapper
except ImportError:  # pragma: no cover - Processing plugin not loaded
    WidgetWrapper = None
    DIALOG_STANDARD = None

SECTION_COLOUR = QColor(217, 83, 30)


def _canvas():
    from qgis.utils import iface

    return None if iface is None else iface.mapCanvas()


def _geometry_type_line():
    return Qgis.GeometryType.Line


class SectionMapTool(QgsMapTool):
    """Click vertices; right-click or Enter finishes; Esc cancels."""

    line_finished = pyqtSignal(object)  # list[QgsPointXY] in canvas CRS
    canceled = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.points = []
        self.band = QgsRubberBand(canvas, _geometry_type_line())
        self.band.setColor(SECTION_COLOUR)
        self.band.setWidth(2)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _map_point(self, event):
        try:
            match = self.canvas().snappingUtils().snapToMap(event.pos())
            if match.isValid():
                return match.point()
        except Exception:  # snapping is a convenience only
            pass
        return event.mapPoint()

    def _show(self, hover=None):
        pts = list(self.points) + ([hover] if hover is not None else [])
        self.band.reset(_geometry_type_line())
        for p in pts:
            self.band.addPoint(p, False)
        self.band.updatePosition()
        self.band.update()

    def canvasMoveEvent(self, event):
        if self.points:
            self._show(self._map_point(event))

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.points.append(self._map_point(event))
            self._show()
        elif event.button() == Qt.MouseButton.RightButton:
            self._finish()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._clear()
            self.canceled.emit()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish()
        elif key == Qt.Key.Key_Backspace and self.points:
            self.points.pop()
            self._show()

    def _finish(self):
        pts = list(self.points)
        self._clear()
        if len(pts) >= 2:
            self.line_finished.emit(pts)
        else:
            self.canceled.emit()

    def _clear(self):
        self.points = []
        self.band.reset(_geometry_type_line())

    def deactivate(self):
        self._clear()
        super().deactivate()


class SectionLinePanel(QWidget):
    """Text box with the drawn sections, plus Draw and Clear buttons."""

    def __init__(self, dialog=None, parent=None):
        super().__init__(parent)
        self.dialog = dialog
        self.parts = []  # list of lists of QgsPointXY, in self.crs
        self.crs = None
        self.tool = None
        self.prev_tool = None
        self.canvas = _canvas()

        self.text = QLineEdit()
        self.text.setPlaceholderText("Draw the section on the map (or paste WKT)")
        self.draw_btn = QToolButton()
        self.draw_btn.setText("Draw section…")
        self.draw_btn.setToolTip(
            "Click the vertices on the map; right-click or Enter to finish, "
            "Backspace removes the last vertex, Esc cancels. Draw again to add "
            "another section (A, B, C ...)."
        )
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear")
        self.info = QLabel("No section drawn")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.text, 1)
        row.addWidget(self.draw_btn)
        row.addWidget(self.clear_btn)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(row)
        box.addWidget(self.info)
        self.draw_btn.clicked.connect(self.start_drawing)
        self.clear_btn.clicked.connect(self.clear)
        if self.canvas is None:
            self.draw_btn.setEnabled(False)

        # drawn sections stay visible on the map while the dialog is open
        self.preview = None
        if self.canvas is not None:
            self.preview = QgsRubberBand(self.canvas, _geometry_type_line())
            self.preview.setColor(SECTION_COLOUR)
            self.preview.setWidth(2)
            canvas, band = self.canvas, self.preview

            def remove_preview(*_):
                try:
                    band.reset(_geometry_type_line())
                    canvas.scene().removeItem(band)
                except Exception:  # canvas already gone
                    pass

            self.destroyed.connect(remove_preview)
            # the Processing dialog overrides QDialog.finished with a method,
            # so only connect attributes that really are signals
            for name in ("rejected", "accepted", "destroyed"):
                signal = getattr(dialog, name, None) if dialog is not None else None
                if signal is not None and hasattr(signal, "connect"):
                    try:
                        signal.connect(remove_preview)
                    except (TypeError, RuntimeError):
                        pass

    # -- drawing -------------------------------------------------------

    def start_drawing(self):
        if self.canvas is None:
            return
        self.prev_tool = self.canvas.mapTool()
        self.tool = SectionMapTool(self.canvas)
        self.tool.line_finished.connect(self._line_drawn)
        self.tool.canceled.connect(self._restore)
        self.canvas.setMapTool(self.tool)
        if self.dialog is not None:
            self.dialog.showMinimized()

    def _line_drawn(self, points):
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        if self.crs is None or not self.parts:
            self.crs = canvas_crs
        elif canvas_crs != self.crs:
            xf = QgsCoordinateTransform(canvas_crs, self.crs, QgsProject.instance())
            points = [xf.transform(p) for p in points]
        self.parts.append(points)
        self._update_text()
        self._restore()

    def _restore(self):
        if self.canvas is not None and self.prev_tool is not None:
            self.canvas.setMapTool(self.prev_tool)
        elif self.canvas is not None and self.tool is not None:
            self.canvas.unsetMapTool(self.tool)
        self.tool = None
        if self.dialog is not None:
            self.dialog.showNormal()
            self.dialog.raise_()
            self.dialog.activateWindow()

    def clear(self):
        self.parts = []
        self.text.clear()
        self._update_preview()

    # -- value -----------------------------------------------------------

    def _update_text(self):
        geom = QgsGeometry.fromMultiPolylineXY(self.parts)
        self.text.setText(f"CRS={self.crs.authid()};{geom.asWkt(3)}")
        self._update_preview()

    def _update_preview(self):
        n = len(self.parts)
        self.info.setText(
            "No section drawn"
            if n == 0
            else f"{n} section(s) drawn - "
            + ", ".join(chr(ord("A") + i) if i < 26 else str(i + 1) for i in range(n))
        )
        if self.preview is None:
            return
        self.preview.reset(_geometry_type_line())
        if n:
            geom = QgsGeometry.fromMultiPolylineXY(self.parts)
            self.preview.setToGeometry(geom, self.crs)

    def value(self):
        return self.text.text().strip()

    def setValue(self, value):
        self.text.setText("" if value is None else str(value))


if WidgetWrapper is not None:

    class SectionLineWidgetWrapper(WidgetWrapper):
        """Uses the drawing panel in the normal dialog; a plain text box in
        batch mode and the model designer."""

        def createWidget(self):
            self._panel = None
            if self.dialogType == DIALOG_STANDARD:
                self._panel = SectionLinePanel(self.dialog)
                return self._panel
            return QLineEdit()

        def setValue(self, value):
            if value is None:
                return
            if getattr(self, "_panel", None) is not None:
                self._panel.setValue(value)
            else:
                self.widget.setText(str(value))

        def value(self):
            if getattr(self, "_panel", None) is not None:
                return self._panel.value()
            return self.widget.text().strip()

else:  # pragma: no cover
    SectionLineWidgetWrapper = None
