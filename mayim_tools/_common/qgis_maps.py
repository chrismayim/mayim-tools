"""
QGIS side of the report layout maps (report and figure standard, section 3):
render styled layers to an image, legend keys drawn by QGIS from the layer
styles (vector and paletted raster), the OpenStreetMap background with a
blank-tile check, and a projected map CRS for layers in geographic
coordinates. The frame around the map is drawn by report_maps.py.

Moved here from hydrology/catchment_delineation (qgis_outputs.py) for reuse.
"""

from __future__ import annotations

import io
import math

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFillSymbol,
    QgsMapRendererCustomPainterJob,
    QgsMapSettings,
    QgsPalettedRasterRenderer,
    QgsRasterLayer,
    QgsRectangle,
    QgsRenderContext,
    QgsSingleSymbolRenderer,
)
from qgis.PyQt.QtCore import QBuffer, QIODevice, QSize
from qgis.PyQt.QtGui import QColor, QImage, QPainter

from .report_figures import CHART_SIZE

OSM_XYZ_SOURCE = (
    "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmin=0&zmax=19"
)
STUDY_AREA_SYMBOL = {
    "style": "no",
    "outline_color": "0,0,0,255",
    "outline_width": "0.5",
    "outline_width_unit": "MM",
    "outline_style": "dash",
}
# legend key size in the report figure (w, h): the legend's handlelength 2.0
# and handleheight 1.0, in units of the figure font size
LEGEND_KEY_PT = (2.0 * CHART_SIZE, 1.0 * CHART_SIZE)
LEGEND_DPI = 300


def png_bytes(image: QImage) -> bytes:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    data = bytes(buffer.data())
    buffer.close()
    return data


def new_image(width: int, height: int, dpi: float, fill) -> QImage:
    image = QImage(QSize(width, height), QImage.Format.Format_ARGB32_Premultiplied)
    dots = int(round(dpi / 0.0254))
    image.setDotsPerMeterX(dots)
    image.setDotsPerMeterY(dots)
    image.fill(fill)
    return image


def symbol_image(symbol, dpi: float = LEGEND_DPI) -> bytes:
    """Legend key of a symbol at the report's legend key size."""
    w = int(round(LEGEND_KEY_PT[0] / 72.0 * dpi))
    h = int(round(LEGEND_KEY_PT[1] / 72.0 * dpi))
    image = new_image(w, h, dpi, QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    context = QgsRenderContext.fromQPainter(painter)
    context.setScaleFactor(dpi / 25.4)
    context.setFlag(Qgis.RenderContextFlag.Antialiasing, True)
    symbol.drawPreviewIcon(painter, QSize(w, h), context)
    painter.end()
    return png_bytes(image)


def colour_key_image(color: QColor, dpi: float = LEGEND_DPI) -> bytes:
    """Legend key for a raster class: a filled box with a thin black edge."""
    symbol = QgsFillSymbol.createSimple(
        {
            "color": f"{color.red()},{color.green()},{color.blue()},{color.alpha()}",
            "outline_color": "0,0,0,255",
            "outline_width": "0.1",
            "outline_width_unit": "MM",
        }
    )
    return symbol_image(symbol, dpi)


def legend_entries(layer, title: str, present_values=None) -> list:
    """[(label, key image)] from a styled layer: one entry (``title``) for a
    single-symbol style, one per category / palette class present for a
    categorised vector or paletted raster style."""
    renderer = layer.renderer()
    if renderer is None:
        return []
    present = None if present_values is None else {str(v) for v in present_values}
    if isinstance(renderer, QgsPalettedRasterRenderer):
        # keys at the renderer's opacity, so they look as on the map
        opacity = renderer.opacity() if hasattr(renderer, "opacity") else 1.0
        entries = []
        for cls in renderer.classes():
            value = cls.value
            key = str(int(value)) if float(value).is_integer() else str(value)
            if present is not None and key not in present:
                continue
            color = QColor(cls.color)
            color.setAlphaF(color.alphaF() * opacity)
            entries.append((cls.label or key, colour_key_image(color)))
        return entries
    if isinstance(renderer, QgsSingleSymbolRenderer):
        return [(title, symbol_image(renderer.symbol()))]
    entries = []
    if isinstance(renderer, QgsCategorizedSymbolRenderer):
        for cat in renderer.categories():
            if not cat.renderState():
                continue
            value = cat.value()
            if present is not None and str(value) not in present:
                continue
            label = cat.label() or str(value)
            entries.append((label, symbol_image(cat.symbol())))
        return entries
    if hasattr(renderer, "legendSymbolItems"):
        for item in renderer.legendSymbolItems():
            if item.symbol() is not None:
                entries.append((item.label() or title, symbol_image(item.symbol())))
    return entries


def render_map(layers, extent, crs, transform_context, width_px, height_px, dpi):
    """Renders ``layers`` (top first) over ``extent``; returns (PNG bytes,
    rendered extent, render errors)."""
    settings = QgsMapSettings()
    settings.setLayers(layers)
    settings.setDestinationCrs(crs)
    settings.setTransformContext(transform_context)
    settings.setOutputSize(QSize(width_px, height_px))
    settings.setOutputDpi(dpi)
    settings.setExtent(QgsRectangle(*extent))
    settings.setBackgroundColor(QColor(255, 255, 255))
    settings.setFlag(Qgis.MapSettingsFlag.Antialiasing, True)
    image = new_image(width_px, height_px, dpi, QColor(255, 255, 255))
    painter = QPainter(image)
    job = QgsMapRendererCustomPainterJob(settings, painter)
    job.renderSynchronously()
    painter.end()
    errors = [f"{e.layerID}: {e.message}" for e in job.errors()]
    visible = settings.visibleExtent()
    rendered = (
        visible.xMinimum(),
        visible.yMinimum(),
        visible.xMaximum(),
        visible.yMaximum(),
    )
    return png_bytes(image), rendered, errors


def osm_layer():
    layer = QgsRasterLayer(OSM_XYZ_SOURCE, "OpenStreetMap", "wms")
    return layer if layer.isValid() else None


def osm_has_tiles(osm, extent, crs, transform_context, width_px, height_px, dpi):
    """Renders the OSM layer alone (tiles are cached for the real render)
    and checks that it is not blank, e.g. when offline."""
    import matplotlib.image as mpimg

    try:
        data, _, _ = render_map(
            [osm], extent, crs, transform_context, width_px, height_px, dpi
        )
        rgb = mpimg.imread(io.BytesIO(data), format="png")[..., :3]
        return float(rgb.max() - rgb.min()) > 0.02
    except Exception:
        return False


def utm_crs_for(lon: float, lat: float) -> QgsCoordinateReferenceSystem:
    """WGS 84 / UTM zone of a point (EPSG 326xx north, 327xx south)."""
    zone = min(max(int(math.floor((lon + 180.0) / 6.0)) + 1, 1), 60)
    return QgsCoordinateReferenceSystem(f"EPSG:{(32600 if lat >= 0 else 32700) + zone}")


def map_crs_and_extent(crs, extent, transform_context):
    """(map CRS, extent in it). Layers in a projected CRS are mapped in that
    CRS; geographic layers in the UTM zone of the extent's centre, so the
    axes read Easting / Northing in metres and the scale bar is true."""
    if not crs.isGeographic():
        return crs, tuple(extent)
    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    to_ll = QgsCoordinateTransform(crs, wgs84, transform_context)
    centre = to_ll.transform(QgsRectangle(*extent).center())  # QgsPointXY in lon / lat
    dest = utm_crs_for(centre.x(), centre.y())
    tr = QgsCoordinateTransform(crs, dest, transform_context)
    box = tr.transformBoundingBox(QgsRectangle(*extent))
    return dest, (box.xMinimum(), box.yMinimum(), box.xMaximum(), box.yMaximum())
