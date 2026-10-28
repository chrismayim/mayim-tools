"""
QGIS side of the DEM Inundation Boundary outputs: in-memory layers of the
boundary, extent and seeds, their styles, loading saved files into the
project, and the report's layout map rendered by QGIS - in the Mayim
report and figure standard.

Reuses the 'Catchments: 1. Delineation' building blocks (shapefile export, map
render, legend keys, OpenStreetMap background, contour layer) rather than
re-coding them.
"""

from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsMemoryProviderUtils,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingLayerPostProcessorInterface,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
)
from qgis.PyQt.QtCore import QMetaType

from mayim_tools.hydrology.catchment_delineation.qgis_outputs import (
    CONTOUR_SYMBOL,
    STUDY_AREA_SYMBOL,
    _osm_has_tiles,
    legend_entries,
    lines_layer,
    osm_layer,
    render_map,
)
from mayim_tools.hydrology.catchment_delineation.report import (
    OSM_ATTRIBUTION,
    PlanMap,
)
from mayim_tools.hydrology.terrain_storage.terrain_storage_algorithm import (
    _rescale_pseudocolor,
)

from .report import plan_extent

# Content colours from the house palette: blue for water, red for seeds.
BOUNDARY_SYMBOL = {
    "line_color": "0,0,255,255",
    "line_width": "0.5",
    "line_width_unit": "MM",
}
EXTENT_SYMBOL = {
    "color": "0,0,255,60",
    "outline_color": "0,0,255,255",
    "outline_width": "0.26",
    "outline_width_unit": "MM",
}
SEED_SYMBOL = {
    "name": "circle",
    "color": "255,0,0,255",
    "outline_color": "0,0,0,255",
    "outline_width": "0.2",
    "size": "2.5",
    "size_unit": "MM",
}

_D = QMetaType.Type.Double
_I = QMetaType.Type.Int
_S = QMetaType.Type.QString


def _renderer(kind: str):
    if kind == "boundary":
        return QgsSingleSymbolRenderer(QgsLineSymbol.createSimple(BOUNDARY_SYMBOL))
    if kind == "extent":
        return QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(EXTENT_SYMBOL))
    if kind == "seeds":
        return QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple(SEED_SYMBOL))
    if kind == "study":
        return QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(STUDY_AREA_SYMBOL))
    raise ValueError(kind)


def style_layer(layer, kind: str) -> None:
    layer.setRenderer(_renderer(kind))


def fields_from(specs) -> QgsFields:
    fields = QgsFields()
    for name, qtype, _ in specs:
        fields.append(QgsField(name, qtype))
    return fields


def memory_layer(name, specs, wkb_type, crs, rows):
    """Memory layer from [(QgsGeometry, {field: value})]."""
    fields = fields_from(specs)
    layer = QgsMemoryProviderUtils.createMemoryLayer(name, fields, wkb_type, crs)
    feats = []
    for geom, values in rows:
        feat = QgsFeature(fields)
        feat.setGeometry(geom)
        for key, value in values.items():
            feat[key] = value
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


def points_layer(name, points, crs):
    from qgis.core import QgsGeometry

    rows = [(QgsGeometry.fromPointXY(QgsPointXY(x, y)), {}) for x, y in points]
    return memory_layer(name, [], Qgis.WkbType.Point, crs, rows)


def polygon_layer(name, geometry, crs):
    """Memory layer holding one polygon geometry (e.g. the bounding area)."""
    from qgis.core import QgsGeometry

    geom = QgsGeometry(geometry)
    geom.convertToMultiType()
    return memory_layer(name, [], Qgis.WkbType.MultiPolygon, crs, [(geom, {})])


class VectorStyler(QgsProcessingLayerPostProcessorInterface):
    """Applies one of the code styles above once Processing has loaded the
    layer."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            style_layer(layer, self.kind)
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style not applied to {layer.name()}: {e}")


def load_on_completion(context, path, name, processor, keep_alive) -> None:
    """Adds a written file to the project when the run finishes, with an
    optional post-processor (kept alive in ``keep_alive`` - Processing holds
    only a weak reference)."""
    details = QgsProcessingContext.LayerDetails(name, context.project(), name)
    if processor is not None:
        details.setPostProcessor(processor)
        keep_alive.append(processor)
    context.addLayerToLoadOnCompletion(path, details)


def _raster_legend(layer, title: str) -> list:
    """[(label, RGBA key image)] from a raster renderer's legend items."""
    import numpy as np

    renderer = layer.renderer()
    if renderer is None:
        return []
    entries = []
    for label, colour in renderer.legendSymbologyItems():
        key = np.zeros((9, 18, 4), dtype=float)
        key[..., 0] = colour.redF()
        key[..., 1] = colour.greenF()
        key[..., 2] = colour.blueF()
        key[..., 3] = colour.alphaF()
        entries.append((f"{title} {label}".strip(), key))
    return entries


def make_plan_map(
    crs,
    transform_context,
    *,
    boundary=None,
    seeds=None,
    study_area=None,
    depth_path: str | None = None,
    depth_qml: str | None = None,
    depth_range=None,
    bounding=None,
    contours_for=None,
    contour_interval: float = 0.0,
    use_osm: bool = True,
    messages: list | None = None,
):
    """Returns the report's PlanMapFn. Layers (memory layers, styled here):
    ``boundary`` lines, ``seeds`` points, ``study_area`` polygon; the depth
    raster from ``depth_path`` with the stage-storage depth style.
    ``contours_for(extent)`` returns [(level, coords)] contour lines."""
    messages = messages if messages is not None else []
    osm = osm_layer() if use_osm else None
    if use_osm and osm is None:
        messages.append("OpenStreetMap background not added: tile layer unavailable.")
    for layer, kind in (
        (boundary, "boundary"),
        (seeds, "seeds"),
        (study_area, "study"),
    ):
        if layer is not None:
            style_layer(layer, kind)
    depth = None
    if depth_path:
        depth = QgsRasterLayer(depth_path, "Depth")
        if not depth.isValid():
            depth = None
            messages.append("Layout map: the depth raster could not be read.")
        elif depth_qml:
            depth.loadNamedStyle(depth_qml)
            if depth_range is not None:
                _rescale_pseudocolor(depth, *depth_range)

    def render(result, plot_width_cm, dpi):
        extent = plan_extent(result, bounding)
        width_px = int(round(plot_width_cm / 2.54 * dpi))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)

        # draw order: first in the list is on top
        top_down = []
        if seeds is not None and seeds.featureCount():
            top_down.append(seeds)
        if boundary is not None and boundary.featureCount():
            top_down.append(boundary)
        if study_area is not None and study_area.featureCount():
            top_down.append(study_area)
        contour_layer = None
        if contours_for is not None and contour_interval > 0:
            lines = contours_for(extent)
            if lines:
                contour_layer = lines_layer("Contours", lines, crs, "elev_m")
                contour_layer.setRenderer(
                    QgsSingleSymbolRenderer(QgsLineSymbol.createSimple(CONTOUR_SYMBOL))
                )
                top_down.append(contour_layer)
        if depth is not None:
            top_down.append(depth)
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
            messages.append(f"Layout map: {err}")

        legend = []
        if boundary is not None and boundary.featureCount():
            legend += legend_entries(
                boundary, f"Inundation boundary ({result.level:.2f} m)"
            )
        if depth is not None:
            legend += _raster_legend(depth, "Depth (m)")
        if study_area is not None and study_area.featureCount():
            legend += legend_entries(study_area, "Bounding polygon")
        if seeds is not None and seeds.featureCount():
            legend += legend_entries(seeds, "Seed point")
        if contour_layer is not None:
            legend += legend_entries(
                contour_layer, f"Contours ({contour_interval:g} m interval)"
            )
        return PlanMap(
            image=image,
            extent=rendered,
            legend=legend,
            attribution=OSM_ATTRIBUTION if osm_ok else None,
        )

    return render
