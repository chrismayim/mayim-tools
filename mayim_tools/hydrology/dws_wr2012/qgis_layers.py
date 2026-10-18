"""
QGIS side of Extract: DWS and WR2012 data: REST fetching through the QGIS
network stack (proxy settings honoured), memory layers from the fetched
features, shapefile writing with reprojection, layer styles (a QML in
./styles when present, otherwise the built-in default symbol) and the
report's layout map.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from qgis.core import (
    QgsBlockingNetworkRequest,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsMemoryProviderUtils,
    QgsPalLayerSettings,
    QgsProcessingContext,
    QgsProcessingLayerPostProcessorInterface,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsTextFormat,
    QgsVectorFileWriter,
    QgsVectorLayerSimpleLabeling,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType, QUrl, QUrlQuery
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtNetwork import QNetworkRequest

from mayim_tools._common.output_files import shapefile_field_names

from .core import RestError

STYLES_DIR = Path(__file__).resolve().parent / "styles"
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

_QTYPE = {
    "int": QMetaType.Type.Int,
    "double": QMetaType.Type.Double,
    "string": QMetaType.Type.QString,
    "date": QMetaType.Type.QString,
}
_WKB = {
    "polygon": QgsWkbTypes.Type.MultiPolygon,
    "line": QgsWkbTypes.Type.MultiLineString,
    "point": QgsWkbTypes.Type.Point,
}

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def make_fetch(feedback=None, timeout_ms: int = 120000):
    """fetch(url, params) -> dict for core.fetch_layer, through the QGIS
    network access manager."""

    def fetch(url: str, params: dict) -> dict:
        qurl = QUrl(url)
        query = QUrlQuery()
        for k, v in params.items():
            query.addQueryItem(k, str(v))
        qurl.setQuery(query)
        request = QNetworkRequest(qurl)
        request.setRawHeader(b"User-Agent", b"Mayim Tools (QGIS)")
        blocking = QgsBlockingNetworkRequest()
        try:
            blocking.setTimeout(timeout_ms)
        except AttributeError:  # older API
            pass
        err = blocking.get(request, forceRefresh=True, feedback=feedback)
        if err != QgsBlockingNetworkRequest.ErrorCode.NoError:
            raise RestError(blocking.errorMessage() or "network error")
        content = bytes(blocking.reply().content())
        try:
            return json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise RestError(f"reply is not JSON ({e})") from e

    return fetch


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def memory_layer(name: str, geometry: str, fields: list[tuple[str, str]]):
    """Empty WGS 84 memory layer with the given (name, kind) fields."""
    qfields = QgsFields()
    for fname, kind in fields:
        qfields.append(QgsField(fname, _QTYPE.get(kind, QMetaType.Type.QString)))
    return QgsMemoryProviderUtils.createMemoryLayer(
        name, qfields, _WKB[geometry], WGS84
    )


def _coerce(value, kind):
    if value is None or value == "":
        return None
    if kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if kind == "double":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value)


def _single_part_point(geom: QgsGeometry) -> QgsGeometry:
    if geom.isMultipart():
        pts = geom.asMultiPoint()
        if pts:
            return QgsGeometry.fromPointXY(pts[0])
    return geom


def add_features(layer, fields, items, geometry: str) -> int:
    """items: [(QgsGeometry, attrs dict)] in WGS 84. Geometries are made
    the layer's type (multi for lines and polygons, single points)."""
    feats = []
    names = [n for n, _ in fields]
    kinds = dict(fields)
    for geom, attrs in items:
        if geom is None or geom.isEmpty():
            continue
        if geometry == "point":
            geom = _single_part_point(geom)
        else:
            geom = QgsGeometry(geom)
            geom.convertToMultiType()
        f = QgsFeature(layer.fields())
        f.setGeometry(geom)
        f.setAttributes([_coerce(attrs.get(n), kinds[n]) for n in names])
        feats.append(f)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return len(feats)


def save_shapefile(layer, path: str, dest_crs, transform_context) -> str:
    """Writes a layer to a shapefile in ``dest_crs`` (UTF-8, field names of
    10 characters or fewer), replacing an existing file."""
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
    if dest_crs is not None and dest_crs.isValid() and dest_crs != layer.crs():
        options.ct = QgsCoordinateTransform(layer.crs(), dest_crs, transform_context)
    names = [f.name() for f in layer.fields()]
    options.attributes = list(range(len(names)))
    options.attributesExportNames = shapefile_field_names(names)
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, path, transform_context, options
    )
    if result[0] != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(f"{Path(path).name} could not be written: {result[1]}")
    return path


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

# output key -> (QML file name in ./styles, default symbol, label field)
STATION_COLOURS = {
    "River": "#0000FF",
    "Reservoir": "#00FFFF",
    "Rainfall": "#00CC00",
    "Evaporation": "#FFC000",
    "Groundwater": "#FF00FF",
    "Tidal": "#FF0000",
}
STATION_LABELS = {
    "River": "River flow gauge",
    "Reservoir": "Reservoir gauge",
    "Rainfall": "Rainfall station",
    "Evaporation": "Evaporation station",
    "Groundwater": "Groundwater borehole",
    "Tidal": "Tidal station",
}
STATION_CLASS_EXPRESSION = "regexp_substr(\"CATEGORY\", '^[A-Za-z]+')"
_OUTLINE = {"style": "no", "outline_width_unit": "MM"}
DEFAULT_STYLES = {
    "OUT_WMA": (
        "DWS_WMA.qml",
        ("fill", {**_OUTLINE, "outline_color": "#FF00FF", "outline_width": "0.8"}),
        None,
    ),
    "OUT_PRIMARY": (
        "DWS_primary.qml",
        (
            "fill",
            {
                **_OUTLINE,
                "outline_color": "#000000",
                "outline_width": "0.7",
                "outline_style": "dash",
            },
        ),
        None,
    ),
    "OUT_SECONDARY": (
        "DWS_secondary.qml",
        (
            "fill",
            {
                **_OUTLINE,
                "outline_color": "#000000",
                "outline_width": "0.5",
                "outline_style": "dash",
            },
        ),
        None,
    ),
    "OUT_TERTIARY": (
        "DWS_tertiary.qml",
        (
            "fill",
            {
                **_OUTLINE,
                "outline_color": "#000000",
                "outline_width": "0.4",
                "outline_style": "dot",
            },
        ),
        None,
    ),
    "OUT_QUATERNARY": (
        "DWS_quaternary.qml",
        ("fill", {**_OUTLINE, "outline_color": "#0000FF", "outline_width": "0.4"}),
        "QUATERNARY",
    ),
    "OUT_QUINARY": (
        "DWS_quinary.qml",
        ("fill", {**_OUTLINE, "outline_color": "#00FFFF", "outline_width": "0.2"}),
        None,
    ),
    "OUT_RIVERS": (
        "DWS_rivers_50k.qml",
        (
            "line",
            {"line_color": "#0000FF", "line_width": "0.3", "line_width_unit": "MM"},
        ),
        None,
    ),
    "OUT_DAMS": (
        "DWS_dams_50k.qml",
        (
            "fill",
            {
                "color": "#00FFFF",
                "outline_color": "#0000FF",
                "outline_width": "0.2",
                "outline_width_unit": "MM",
            },
        ),
        None,
    ),
    "OUT_SW_RESERVE": (
        "DWS_sw_reserve.qml",
        ("fill", {**_OUTLINE, "outline_color": "#FFC000", "outline_width": "0.6"}),
        None,
    ),
    "OUT_GW_RESERVE": (
        "DWS_gw_reserve.qml",
        ("fill", {**_OUTLINE, "outline_color": "#00CC00", "outline_width": "0.6"}),
        None,
    ),
    "OUT_EWR_SITES": (
        "DWS_ewr_sites.qml",
        (
            "marker",
            {
                "name": "triangle",
                "color": "#FF0000",
                "outline_color": "#000000",
                "size": "3",
                "size_unit": "MM",
            },
        ),
        None,
    ),
    "OUT_WR_QUAT": (
        "WR2012_quaternaries.qml",
        ("fill", {**_OUTLINE, "outline_color": "#00CC00", "outline_width": "0.5"}),
        "QUAT",
    ),
    "OUT_STATIONS": ("DWS_stations.qml", None, "STATION"),
    "STUDY_AREA": (
        None,
        (
            "fill",
            {
                **_OUTLINE,
                "outline_color": "#000000",
                "outline_width": "0.6",
                "outline_style": "dash",
            },
        ),
        None,
    ),
}


def style_file(key: str) -> str | None:
    qml = DEFAULT_STYLES.get(key, (None, None, None))[0]
    if qml and (STYLES_DIR / qml).is_file():
        return str(STYLES_DIR / qml)
    return None


def _symbol(kind, props):
    if kind == "fill":
        return QgsFillSymbol.createSimple(props)
    if kind == "line":
        return QgsLineSymbol.createSimple(props)
    return QgsMarkerSymbol.createSimple(props)


def _station_renderer():
    cats = []
    for value, colour in STATION_COLOURS.items():
        sym = QgsMarkerSymbol.createSimple(
            {
                "name": "circle",
                "color": colour,
                "outline_color": "#000000",
                "outline_width": "0.2",
                "size": "2.4",
                "size_unit": "MM",
            }
        )
        cats.append(QgsRendererCategory(value, sym, STATION_LABELS[value]))
    # CATEGORY can hold two types ("Rainfall / Evaporation"): colour by the first
    return QgsCategorizedSymbolRenderer(STATION_CLASS_EXPRESSION, cats)


def _labels(layer, field_name: str, size_pt: float = 8.0):
    if layer.fields().indexOf(field_name) < 0:
        return
    settings = QgsPalLayerSettings()
    settings.fieldName = field_name
    fmt = QgsTextFormat()
    font = QFont("Arial Narrow")
    fmt.setFont(font)
    fmt.setSize(size_pt)
    fmt.setColor(QColor("#000000"))
    buffer = fmt.buffer()
    buffer.setEnabled(True)
    buffer.setColor(QColor("#FFFFFF"))
    buffer.setSize(0.8)
    fmt.setBuffer(buffer)
    settings.setFormat(fmt)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def apply_style(layer, key: str) -> None:
    """The QML in ./styles when it exists, else the built-in symbol."""
    if layer is None:
        return
    path = style_file(key)
    if path:
        layer.loadNamedStyle(path)
        return
    _qml, symbol, label = DEFAULT_STYLES.get(key, (None, None, None))
    if key == "OUT_STATIONS":
        layer.setRenderer(_station_renderer())
    elif symbol is not None:
        layer.setRenderer(QgsSingleSymbolRenderer(_symbol(*symbol)))
    if label:
        _labels(layer, label)
    layer.triggerRepaint()


class StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            apply_style(layer, self.key)
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style not applied to {layer.name()}: {e}")


def load_on_completion(context, path: str, name: str, key: str, keep_alive) -> None:
    details = QgsProcessingContext.LayerDetails(name, context.project(), name)
    processor = StylePostProcessor(key)
    details.setPostProcessor(processor)
    keep_alive.append(processor)
    context.addLayerToLoadOnCompletion(path, details)


def styled_clone(layer, key: str):
    clone = layer.clone()
    apply_style(clone, key)
    return clone


def present_station_types(layer) -> set:
    idx = layer.fields().indexOf("CATEGORY")
    out = set()
    if idx < 0:
        return out
    for f in layer.getFeatures():
        out.add(str(f[idx]).split(" / ")[0])
    return out
