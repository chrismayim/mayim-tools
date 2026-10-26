"""QGIS side of the Design Rainfall vector outputs: memory point layers
for the snapped grid points, points of interest and nearest rainfall
stations are built here, written to shapefiles with the shared writer
(Catchment Delineation's ``qgis_outputs.save_shapefile``) and loaded into
the project with this tool's own saved QML styles.

The layer styles (QML) in ./styles are copies of the Mayim layer styles
(D:\\...\\QGIS Layer styles); recopy them here when a style changes.
"""

from __future__ import annotations

from pathlib import Path

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMemoryProviderUtils,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingLayerPostProcessorInterface,
)
from qgis.PyQt.QtCore import QMetaType

from mayim_tools.hydrology.catchment_delineation.qgis_outputs import save_shapefile

__all__ = [
    "save_shapefile",
    "load_on_completion",
    "grid_points_layer",
    "poi_layer",
    "stations_layer",
]

STYLES_DIR = Path(__file__).resolve().parent / "styles"


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Loads one of this tool's saved QML styles (symbology + labeling)
    onto an output layer once Processing has added it to the project."""

    def __init__(self, style_filename: str):
        super().__init__()
        self.style_path = str(STYLES_DIR / style_filename)

    def postProcessLayer(self, layer, context, feedback):
        layer.loadNamedStyle(self.style_path)
        layer.triggerRepaint()


def load_on_completion(
    context: QgsProcessingContext,
    path: str,
    name: str,
    style_filename: str | None,
    keep_alive: list,
) -> None:
    """Adds a written shapefile to the project when the run finishes,
    styled with one of this tool's saved QML files (``style_filename``,
    e.g. ``snapped_grid_point.qml``). The post-processor is kept alive in
    ``keep_alive`` for the duration of the run, since QGIS holds only a
    weak reference to it."""
    details = QgsProcessingContext.LayerDetails(name, context.project(), name)
    if style_filename:
        processor = _StylePostProcessor(style_filename)
        details.setPostProcessor(processor)
        keep_alive.append(processor)
    context.addLayerToLoadOnCompletion(path, details)


def _qmetatype_for_dtype(dtype):
    """Maps a pandas/numpy dtype to the closest QMetaType for QgsField."""
    kind = getattr(dtype, "kind", "O")
    if kind in ("i", "u"):
        return QMetaType.Type.LongLong
    if kind == "f":
        return QMetaType.Type.Double
    return QMetaType.Type.QString


def grid_points_layer(estimates: dict, crs):
    """One feature per site: site label, MAP, altitude, cluster, and one
    field per duration x return-period combo (median/lower/upper). Field
    names over 10 characters are shortened by ``save_shapefile`` when the
    layer is written."""
    if not estimates:
        return None
    fields = QgsFields()
    fields.append(QgsField("site", QMetaType.Type.QString))
    fields.append(QgsField("map_mm", QMetaType.Type.Double))
    fields.append(QgsField("altitude_m", QMetaType.Type.Double))
    fields.append(QgsField("cluster", QMetaType.Type.Int))

    sample_results = next(iter(estimates.values())).results
    combo_keys = []
    for r in sample_results:
        key = f"{r.duration_label.replace(' ', '')}_{r.return_period}y"
        combo_keys.append(key)
        fields.append(QgsField(key, QMetaType.Type.Double))
        fields.append(QgsField(f"{key}_L", QMetaType.Type.Double))
        fields.append(QgsField(f"{key}_U", QMetaType.Type.Double))

    layer = QgsMemoryProviderUtils.createMemoryLayer(
        "Snapped grid point", fields, Qgis.WkbType.Point, crs
    )
    feats = []
    for label, est in estimates.items():
        feat = QgsFeature(fields)
        feat.setGeometry(
            QgsGeometry.fromPointXY(QgsPointXY(est.longitude, est.latitude))
        )
        lookup = {
            f"{r.duration_label.replace(' ', '')}_{r.return_period}y": r
            for r in est.results
        }
        attrs = [label, est.map_mm, est.altitude_m, est.cluster]
        for key in combo_keys:
            r = lookup.get(key)
            attrs += [
                r.depth if r else None,
                r.lower if r else None,
                r.upper if r else None,
            ]
        feat.setAttributes(attrs)
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


def poi_layer(sites: list, crs):
    """One feature per site: label and the requested latitude/longitude -
    useful to see/symbolise the input location(s) independently of the
    snapped grid points."""
    if not sites:
        return None
    fields = QgsFields()
    fields.append(QgsField("site", QMetaType.Type.QString))
    fields.append(QgsField("latitude", QMetaType.Type.Double))
    fields.append(QgsField("longitude", QMetaType.Type.Double))
    layer = QgsMemoryProviderUtils.createMemoryLayer(
        "POI", fields, Qgis.WkbType.Point, crs
    )
    feats = []
    for label, lat, lon in sites:
        feat = QgsFeature(fields)
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
        feat.setAttributes([label, lat, lon])
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


def stations_layer(per_site_stations: dict, crs):
    """Nearest N daily rainfall stations per site, for reference/
    comparison only - NOT used in the grid-based calculation itself
    (mirrors the original tool's 'closest stations' listing).
    ``per_site_stations``: dict[label, GeoDataFrame]."""
    if not per_site_stations or all(df.empty for df in per_site_stations.values()):
        return None
    sample_df = next(df for df in per_site_stations.values() if not df.empty)
    data_cols = [c for c in sample_df.columns if c != "geometry"]
    fields = QgsFields()
    fields.append(QgsField("site", QMetaType.Type.QString))
    for col in data_cols:
        fields.append(
            QgsField(str(col)[:80], _qmetatype_for_dtype(sample_df[col].dtype))
        )
    layer = QgsMemoryProviderUtils.createMemoryLayer(
        "Nearest rainfall stations", fields, Qgis.WkbType.Point, crs
    )
    feats = []
    for label, nearby in per_site_stations.items():
        for _, row in nearby.iterrows():
            feat = QgsFeature(fields)
            feat.setGeometry(
                QgsGeometry.fromPointXY(QgsPointXY(row.geometry.x, row.geometry.y))
            )
            feat.setAttributes([label] + [row[col] for col in data_cols])
            feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer
