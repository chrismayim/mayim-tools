"""
QGIS side of the Design Rainfall report's layout map (report and figure
standard, section 3): per site, the point of interest, the snapped grid
point and the nearest daily rainfall stations are drawn by QGIS with the
tool's QML styles (styles/poi.qml, snapped_grid_point.qml,
nearest_stations.qml) over the OpenStreetMap background. The frame around
the map (axes, legend, scale bar, north arrow) is composed in report.py by
mayim_tools._common.report_maps.
"""

from __future__ import annotations

from pathlib import Path

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType

from mayim_tools._common.qgis_maps import (
    legend_entries,
    map_crs_and_extent,
    osm_has_tiles,
    osm_layer,
    render_map,
)
from mayim_tools._common.report_maps import OSM_ATTRIBUTION, PlanMap

from .report import map_extent

STYLES_DIR = Path(__file__).resolve().parent / "styles"
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

POI_QML = STYLES_DIR / "poi.qml"
GRID_QML = STYLES_DIR / "snapped_grid_point.qml"
STATIONS_QML = STYLES_DIR / "nearest_stations.qml"

LEGEND_POI = "Point of interest"
LEGEND_GRID = "Snapped grid point"
LEGEND_STATIONS = "Nearest daily rainfall stations (reference only)"

# station attributes carried into the map layer (the station QML labels
# use STN_NAME, MAP and REC_LEN)
_STATION_FIELDS = (
    ("FSTN_NO", QMetaType.Type.QString),
    ("STN_NAME", QMetaType.Type.QString),
    ("MAP", QMetaType.Type.Double),
    ("REC_LEN", QMetaType.Type.Int),
    ("DIST_KM", QMetaType.Type.Double),
)


def _point_layer(name: str, fields, rows, qml: Path, messages: list):
    """Memory point layer in WGS 84 from [(lon, lat, [values])], styled with
    ``qml``."""
    layer = QgsVectorLayer("Point?crs=EPSG:4326", name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([QgsField(n, t) for n, t in fields])
    layer.updateFields()
    feats = []
    for lon, lat, values in rows:
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
        feat.setAttributes(list(values))
        feats.append(feat)
    provider.addFeatures(feats)
    layer.updateExtents()
    if qml.exists():
        _msg, ok = layer.loadNamedStyle(str(qml))
        if not ok:
            messages.append(f"Layout map: style {qml.name} could not be loaded.")
    else:
        messages.append(f"Layout map: style {qml.name} not found.")
    return layer


def _make_label_obstacle(layer) -> None:
    """Registers the layer's points as obstacles for other labels without
    drawing labels of its own, so the station labels do not cover the
    point of interest or the grid point."""
    settings = QgsPalLayerSettings()
    settings.fieldName = "site"
    settings.drawLabels = False
    settings.obstacleSettings().setIsObstacle(True)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def _station_values(row):
    values = []
    for name, qtype in _STATION_FIELDS:
        value = row.get(name)
        if value is None or value != value:  # missing / NaN
            values.append(None)
        elif qtype == QMetaType.Type.Int:
            values.append(int(value))
        elif qtype == QMetaType.Type.Double:
            values.append(float(value))
        else:
            values.append(str(value).strip())
    return values


def make_plan_map(
    sites: dict,
    transform_context,
    use_osm: bool = True,
    messages: list | None = None,
):
    """Returns the report's PlanMapFn (site label, plot width cm, dpi) ->
    PlanMap. ``sites`` maps a site label to a dict with ``poi`` (lon, lat),
    ``grid`` (lon, lat) and ``stations`` (GeoDataFrame in WGS 84 or None)."""
    messages = messages if messages is not None else []
    osm = osm_layer() if use_osm else None
    if use_osm and osm is None:
        messages.append("OpenStreetMap background not added: tile layer unavailable.")
    osm_warned = []

    def render(label, plot_width_cm, dpi):
        site = sites.get(label)
        if site is None:
            return None
        poi_lon, poi_lat = site["poi"]
        grid_lon, grid_lat = site["grid"]
        stations = site.get("stations")

        poi = _point_layer(
            "POI",
            [("site", QMetaType.Type.QString)],
            [(poi_lon, poi_lat, [label])],
            POI_QML,
            messages,
        )
        grid = _point_layer(
            "Snapped grid point",
            [("site", QMetaType.Type.QString)],
            [(grid_lon, grid_lat, [label])],
            GRID_QML,
            messages,
        )
        for layer in (poi, grid):
            try:
                _make_label_obstacle(layer)
            except Exception as e:  # cosmetic only
                messages.append(f"Layout map: label obstacle not set ({e}).")
        station_layer = None
        lonlat = [(poi_lon, poi_lat), (grid_lon, grid_lat)]
        if stations is not None and not stations.empty:
            rows = []
            for _, s in stations.iterrows():
                rows.append((s.geometry.x, s.geometry.y, _station_values(s)))
                lonlat.append((s.geometry.x, s.geometry.y))
            station_layer = _point_layer(
                "Nearest rainfall stations",
                _STATION_FIELDS,
                rows,
                STATIONS_QML,
                messages,
            )

        # map CRS (UTM zone of the site) and the projected point positions
        xs = [p[0] for p in lonlat]
        ys = [p[1] for p in lonlat]
        crs, box = map_crs_and_extent(
            WGS84, (min(xs), min(ys), max(xs), max(ys)), transform_context
        )
        to_map = QgsCoordinateTransform(WGS84, crs, transform_context)
        pts = []
        for lon, lat in lonlat:
            p = to_map.transform(QgsPointXY(lon, lat))
            pts.append((p.x(), p.y()))
        extent = map_extent(pts) if pts else box

        width_px = int(round(plot_width_cm / 2.54 * dpi))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)

        top_down = [poi, grid]  # first in the list is drawn on top
        if station_layer is not None:
            top_down.append(station_layer)
        osm_ok = False
        if osm is not None:
            osm_ok = osm_has_tiles(
                osm, extent, crs, transform_context, width_px, height_px, dpi
            )
            if osm_ok:
                top_down.append(osm)
            elif not osm_warned:
                osm_warned.append(True)
                messages.append(
                    "OpenStreetMap background not added: no map tiles were "
                    "received (check the internet connection or proxy)."
                )
        image, rendered, errors = render_map(
            top_down, extent, crs, transform_context, width_px, height_px, dpi
        )
        for err in errors:
            messages.append(f"Layout map ({label}): {err}")

        legend = legend_entries(poi, LEGEND_POI) + legend_entries(grid, LEGEND_GRID)
        if station_layer is not None:
            legend += legend_entries(station_layer, LEGEND_STATIONS)
        return PlanMap(
            image=image,
            extent=rendered,
            legend=legend,
            attribution=OSM_ATTRIBUTION if osm_ok else None,
        )

    return render
