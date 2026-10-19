"""
QGIS side of the Catchment Delineation outputs: styled in-memory layers,
shapefile export, loading saved files into the project with the house
layer styles, and the report's layout map rendered by QGIS.

The layer styles (QML) in ./styles are copies of the Mayim layer styles
(D:\\...\\QGIS Layer styles); recopy them here when a style changes.
"""

from __future__ import annotations

import os
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsMapRendererCustomPainterJob,
    QgsMapSettings,
    QgsMemoryProviderUtils,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingLayerPostProcessorInterface,
    QgsRasterLayer,
    QgsRectangle,
    QgsRenderContext,
    QgsSingleSymbolRenderer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorFileWriter,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QBuffer, QIODevice, QMetaType, QSize
from qgis.PyQt.QtGui import QColor, QImage, QPainter

from mayim_tools._common.output_files import shapefile_field_names
from mayim_tools._common.qgis_maps import contour_layers
from mayim_tools._common.report_figures import CHART_SIZE

from .report import OSM_ATTRIBUTION, PlanMap, plan_extent

STYLES_DIR = Path(__file__).resolve().parent / "styles"
STYLE_CATCHMENTS = "DRN_watershed.qml"
STYLE_CENTROIDS = "DRN_centroids.qml"
STYLE_FLOWPATHS = "DRN_longest_flowpath.qml"
STYLE_OUTLETS = "DRN_outlets.qml"
STYLE_STREAMS = "DRN_strahler_stream_order.qml"
STYLE_NODES = "DRN_stream_nodes.qml"

OSM_XYZ_SOURCE = (
    "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmin=0&zmax=19"
)
# single contour weight, still used by DEM Inundation Boundary (Catchment
# Delineation uses the minor / major contours of _common.qgis_maps)
CONTOUR_SYMBOL = {
    "line_color": "0,0,0,255",
    "line_width": "0.2",
    "line_width_unit": "MM",
}
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


# ---------------------------------------------------------------------------
# In-memory layers
# ---------------------------------------------------------------------------


class MemoryLayers:
    """Collects the run's vector outputs as memory layers. ``parameterAsSink``
    mimics QgsProcessingAlgorithm.parameterAsSink so the Stream Network
    writers (which take the algorithm) can fill these layers directly."""

    def __init__(self):
        self.layers = {}

    def parameterAsSink(
        self, parameters, key, context, fields, wkb_type, crs
    ):  # noqa: N802
        layer = QgsMemoryProviderUtils.createMemoryLayer(key, fields, wkb_type, crs)
        self.layers[key] = layer
        return layer.dataProvider(), key

    def get(self, key):
        return self.layers.get(key)


def style_path(name: str) -> str:
    return str(STYLES_DIR / name)


def apply_style(layer, qml_name: str | None) -> None:
    if layer is None or not qml_name:
        return
    path = style_path(qml_name)
    if os.path.exists(path):
        layer.loadNamedStyle(path)


def lines_layer(name, lines, crs, value_field=None, wkb=None):
    """Memory line layer from [(value, [(x, y), ...]), ...]."""
    fields = QgsFields()
    if value_field:
        fields.append(QgsField(value_field, QMetaType.Type.Double))
    layer = QgsMemoryProviderUtils.createMemoryLayer(
        name, fields, wkb or Qgis.WkbType.LineString, crs
    )
    provider = layer.dataProvider()
    feats = []
    for value, coords in lines:
        if len(coords) < 2:
            continue
        feat = QgsFeature(fields)
        feat.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in coords])
        )
        if value_field:
            feat[value_field] = float(value)
        feats.append(feat)
    provider.addFeatures(feats)
    layer.updateExtents()
    return layer


def stream_segments_layer(streams, crs):
    """Report-map stream layer from the catchment core's stream segments
    (x0, y0, x1, y1, Strahler order) - carries the 'strahler' field the
    Strahler style classifies on."""
    fields = QgsFields()
    fields.append(QgsField("strahler", QMetaType.Type.Int))
    layer = QgsMemoryProviderUtils.createMemoryLayer(
        "Streams", fields, Qgis.WkbType.LineString, crs
    )
    feats = []
    for x0, y0, x1, y1, order in streams:
        feat = QgsFeature(fields)
        feat.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(x0, y0), QgsPointXY(x1, y1)])
        )
        feat["strahler"] = int(order)
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


def copy_source_layer(source, name, crs, transform_context):
    """Memory copy of a Processing feature source, reprojected to ``crs``."""
    from qgis.core import QgsCoordinateTransform

    layer = QgsMemoryProviderUtils.createMemoryLayer(
        name, source.fields(), source.wkbType(), crs
    )
    transform = None
    if source.sourceCrs() != crs:
        transform = QgsCoordinateTransform(source.sourceCrs(), crs, transform_context)
    feats = []
    for feat in source.getFeatures():
        geom = QgsGeometry(feat.geometry())
        if geom.isNull():
            continue
        if transform is not None:
            geom.transform(transform)
        new = QgsFeature(feat)
        new.setGeometry(geom)
        feats.append(new)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def save_shapefile(layer, path: str, transform_context) -> str:
    """Writes a memory layer to a shapefile, replacing an existing one.
    Field names longer than 10 characters are shortened explicitly."""
    if os.path.exists(path) and not QgsVectorFileWriter.deleteShapeFile(path):
        raise RuntimeError(
            f"{Path(path).name} exists and could not be replaced - it is "
            "probably open in QGIS or another program. Remove it from the "
            "project and run again."
        )
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    names = [f.name() for f in layer.fields()]
    options.attributes = list(range(len(names)))
    options.attributesExportNames = shapefile_field_names(names)
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, path, transform_context, options
    )
    error, message = result[0], result[1]
    if error != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(f"{Path(path).name} could not be written: {message}")
    return path


class StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Applies a saved QML style once Processing has loaded the layer."""

    def __init__(self, qml_name: str):
        super().__init__()
        self.qml_name = qml_name

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            apply_style(layer, self.qml_name)
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style {self.qml_name} not applied: {e}")


def load_on_completion(context, path: str, name: str, qml_name, keep_alive) -> None:
    """Adds a written file to the project when the run finishes, styled with
    its QML (the post-processor is kept alive in ``keep_alive``)."""
    details = QgsProcessingContext.LayerDetails(name, context.project(), name)
    if qml_name:
        processor = StylePostProcessor(qml_name)
        details.setPostProcessor(processor)
        keep_alive.append(processor)
    context.addLayerToLoadOnCompletion(path, details)


# ---------------------------------------------------------------------------
# Report layout map
# ---------------------------------------------------------------------------


def _png_bytes(image: QImage) -> bytes:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    data = bytes(buffer.data())
    buffer.close()
    return data


def _new_image(width: int, height: int, dpi: float, fill) -> QImage:
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
    image = _new_image(w, h, dpi, QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    context = QgsRenderContext.fromQPainter(painter)
    context.setScaleFactor(dpi / 25.4)
    context.setFlag(Qgis.RenderContextFlag.Antialiasing, True)
    symbol.drawPreviewIcon(painter, QSize(w, h), context)
    painter.end()
    return _png_bytes(image)


def legend_entries(layer, title: str, present_values=None) -> list:
    """[(label, key image)] from a styled layer: one entry (``title``) for a
    single-symbol style, one per category present for a categorised one."""
    renderer = layer.renderer()
    if renderer is None:
        return []
    if isinstance(renderer, QgsSingleSymbolRenderer):
        return [(title, symbol_image(renderer.symbol()))]
    entries = []
    if isinstance(renderer, QgsCategorizedSymbolRenderer):
        present = None if present_values is None else {str(v) for v in present_values}
        for cat in renderer.categories():
            if not cat.renderState():
                continue
            value = cat.value()
            if present is not None and str(value) not in present:
                continue
            label = cat.label() or str(value)
            entries.append((label, symbol_image(cat.symbol())))
        return entries
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
    image = _new_image(width_px, height_px, dpi, QColor(255, 255, 255))
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
    return _png_bytes(image), rendered, errors


def _osm_has_tiles(osm, extent, crs, transform_context, width_px, height_px, dpi):
    """Renders the OSM layer alone (tiles are cached for the real render)
    and checks that it is not blank, e.g. when offline."""
    import io

    import matplotlib.image as mpimg

    try:
        data, _, _ = render_map(
            [osm], extent, crs, transform_context, width_px, height_px, dpi
        )
        rgb = mpimg.imread(io.BytesIO(data), format="png")[..., :3]
        return float(rgb.max() - rgb.min()) > 0.02
    except Exception:
        return False


def osm_layer():
    layer = QgsRasterLayer(OSM_XYZ_SOURCE, "OpenStreetMap", "wms")
    return layer if layer.isValid() else None


def label_layer(layer, field: str, size_pt: float = 8.0) -> None:
    """Labels ``layer`` with ``field`` in Arial, with a white buffer."""
    from qgis.PyQt.QtGui import QFont

    fmt = QgsTextFormat()
    fmt.setFont(QFont("Arial"))
    fmt.setSize(size_pt)
    fmt.setColor(QColor(0, 0, 0))
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(0.8)
    buf.setColor(QColor(255, 255, 255))
    fmt.setBuffer(buf)
    settings = QgsPalLayerSettings()
    settings.fieldName = field
    settings.setFormat(fmt)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def make_plan_map(
    crs,
    transform_context,
    memory: MemoryLayers,
    keys: dict,
    study_area=None,
    contours_for=None,
    contour_interval: float = 0.0,
    use_osm: bool = True,
    messages: list | None = None,
    label_catchments: bool = False,
):
    """Returns the report's PlanMapFn. ``keys`` maps the roles 'catchments',
    'flowpaths', 'outlets', 'centroids', 'nodes' to MemoryLayers keys (a
    missing role is left off the map). ``contours_for(extent)`` returns
    (minor, major) contour lines [(level, coords)] for an extent, with
    ``contour_interval`` the minor interval (major = 5 x minor).
    ``label_catchments`` names every catchment on the map (incremental
    mode); the overview map of all sub-catchments always does."""
    messages = messages if messages is not None else []
    osm = osm_layer() if use_osm else None
    if use_osm and osm is None:
        messages.append("OpenStreetMap background not added: tile layer unavailable.")
    styled = {
        "catchments": STYLE_CATCHMENTS,
        "flowpaths": STYLE_FLOWPATHS,
        "outlets": STYLE_OUTLETS,
        "centroids": STYLE_CENTROIDS,
        "nodes": STYLE_NODES,
        "streams": STYLE_STREAMS,
    }
    titles = {
        "catchments": "Catchment",
        "flowpaths": "Longest flow path",
        "outlets": "Outlet",
        "centroids": "Centroid",
    }
    base = {}
    for role, key in keys.items():
        layer = memory.get(key)
        if layer is None or layer.featureCount() == 0:
            continue
        clone = layer.clone()
        apply_style(clone, styled[role])
        if role != "catchments":
            clone.setLabelsEnabled(False)  # names go on the catchments
        base[role] = clone
    if study_area is not None:
        study_area.setRenderer(
            QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(STUDY_AREA_SYMBOL))
        )

    def render(c, plot_width_cm, dpi):
        extent = plan_extent(c)
        if study_area is not None and study_area.featureCount():
            sa = study_area.extent()
            pad = 0.05 * max(sa.width(), sa.height(), 1.0)
            extent = (
                min(extent[0], sa.xMinimum() - pad),
                min(extent[1], sa.yMinimum() - pad),
                max(extent[2], sa.xMaximum() + pad),
                max(extent[3], sa.yMaximum() + pad),
            )
        width_px = int(round(plot_width_cm / 2.54 * dpi))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)

        if "catchments" in base:
            if c.detail.get("label_catchments") or label_catchments:
                label_layer(base["catchments"], "outlet_id")
            else:
                base["catchments"].setLabelsEnabled(False)

        # draw order: first in the list is on top
        top_down, legend = [], []
        overview = bool(c.detail.get("overview"))
        streams = c.detail.get("streams")
        stream_layer = None
        if overview and "streams" in base:
            stream_layer = base["streams"]  # the named, ordered network
        elif streams is not None and len(streams):
            stream_layer = stream_segments_layer(streams, crs)
            apply_style(stream_layer, STYLE_STREAMS)
        point_roles = ("nodes",) if overview else (
            "centroids", "outlets", "nodes", "flowpaths"
        )  # fmt: skip
        for role in point_roles:
            if role in base:
                top_down.append(base[role])
        if stream_layer is not None:
            top_down.append(stream_layer)
        if "catchments" in base:
            top_down.append(base["catchments"])
        if study_area is not None and study_area.featureCount():
            top_down.append(study_area)
        contour_sets = []  # [(layer, legend label)], major first
        if contours_for is not None and contour_interval > 0:
            minor, major = contours_for(extent)
            contour_sets = contour_layers(minor, major, crs, contour_interval)
            top_down += [layer for layer, _label in contour_sets]
        osm_ok = False
        if osm is not None:
            osm_ok = _osm_has_tiles(
                osm, extent, crs, transform_context, width_px, height_px, dpi
            )
            if osm_ok:
                top_down.append(osm)
            else:
                messages.append(
                    "OpenStreetMap background not added: no map tiles were "
                    "received (check the internet connection or proxy)."
                )

        image, rendered, errors = render_map(
            top_down, extent, crs, transform_context, width_px, height_px, dpi
        )
        for err in errors:
            messages.append(f"Layout map, catchment {c.outlet_id}: {err}")

        # legend, in reading order
        if "catchments" in base:
            legend += legend_entries(base["catchments"], titles["catchments"])
        if study_area is not None and study_area.featureCount():
            legend += legend_entries(study_area, "Study area")
        for layer, label in reversed(contour_sets):  # minor, then major
            legend += legend_entries(layer, label)
        if stream_layer is not None:
            if stream_layer is base.get("streams"):
                orders = sorted(
                    {int(f["strahler"]) for f in stream_layer.getFeatures()}
                )
            else:
                orders = sorted({int(o) for o in streams[:, 4]})
            for label, img in legend_entries(stream_layer, "Streams", orders):
                legend.append((f"Stream {label[0].lower()}{label[1:]}", img))
        for role in () if overview else ("flowpaths", "outlets", "centroids"):
            if role in base:
                legend += legend_entries(base[role], titles[role])
        if "nodes" in base:
            types = {f["node_type"] for f in base["nodes"].getFeatures()}
            for label, img in legend_entries(base["nodes"], "Stream node", types):
                legend.append(
                    (f"Stream node: {label}" if label else "Stream node", img)
                )
        return PlanMap(
            image=image,
            extent=rendered,
            legend=legend,
            attribution=OSM_ATTRIBUTION if osm_ok else None,
        )

    return render
