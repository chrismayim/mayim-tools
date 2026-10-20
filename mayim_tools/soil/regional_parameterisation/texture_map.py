"""QGIS layout maps of the USDA texture class per depth layer for the
Regional Soil Parameterisation report: the texture raster in the Mayim layer
style (styles/SOIL_USDA_classification.qml) at 70 % transparency over the
OpenStreetMap background, with the optional site and catchment boundary
layers drawn above it in their project symbology (the zones as a dashed
study-area outline when no catchment layer is given). The extent fits the
catchment layer with a margin, or else the texture raster. The report frames
the map (report_maps.figure_rendered_map)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsFillSymbol,
    QgsGeometry,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)

from mayim_tools._common.qgis_maps import (
    STUDY_AREA_SYMBOL,
    legend_entries,
    map_crs_and_extent,
    osm_has_tiles,
    osm_layer,
    render_map,
)
from mayim_tools._common.report_maps import OSM_ATTRIBUTION, PlanMap

from .texture import TEXTURE_STYLE

STYLE_PATH = Path(__file__).resolve().parent / "styles" / TEXTURE_STYLE
TEXTURE_OPACITY = 0.3  # 70 % transparency, so the background shows through
CATCHMENT_MARGIN = 0.08  # map margin around the catchment, fraction of its size
RASTER_MARGIN = 0.02  # small margin around the texture raster


def snapshot_layer(layer, name: str | None = None):
    """A memory copy of a project vector layer with its symbology (renderer,
    opacity), safe to render on the Processing thread. None if empty."""
    if layer is None or not layer.isValid() or layer.featureCount() == 0:
        return None
    copy = layer.materialize(QgsFeatureRequest())
    copy.setName(name or layer.name())
    if layer.renderer() is not None:
        copy.setRenderer(layer.renderer().clone())
    copy.setOpacity(layer.opacity())
    return copy


def _extent_in(layer, crs, transform_context):
    box = layer.extent()
    if layer.crs() != crs:
        tr = QgsCoordinateTransform(layer.crs(), crs, transform_context)
        box = tr.transformBoundingBox(box)
    return box.xMinimum(), box.yMinimum(), box.xMaximum(), box.yMaximum()


def _pad(extent, fraction):
    pad = fraction * max(extent[2] - extent[0], extent[3] - extent[1], 1.0)
    return (extent[0] - pad, extent[1] - pad, extent[2] + pad, extent[3] + pad)


def _zones_layer(zones, crs):
    layer = QgsVectorLayer(
        f"Polygon?crs={crs.authid() or 'EPSG:4326'}", "Zones", "memory"
    )
    if not crs.authid():
        layer.setCrs(crs)
    feats = []
    for _name, wkt in zones:
        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromWkt(wkt))
        feats.append(f)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    layer.setRenderer(
        QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(STUDY_AREA_SYMBOL))
    )
    return layer


def make_texture_map(
    transform_context,
    use_osm: bool = True,
    messages=None,
    site_layer=None,
    catchment_layer=None,
):
    """Returns map_fn(result, layer label, plot width cm, dpi) -> PlanMap.
    ``site_layer`` and ``catchment_layer`` are snapshot_layer() copies (or
    None), drawn above the texture in their own symbology."""
    messages = messages if messages is not None else []
    osm = osm_layer() if use_osm else None
    if use_osm and osm is None:
        messages.append("OpenStreetMap background not added: tile layer unavailable.")
    state = {"osm_ok": None}

    def render(result, lab, plot_width_cm, dpi):
        path = result.texture_paths.get(lab)
        if not path:
            return None
        grid = result.grid
        crs = QgsCoordinateReferenceSystem.fromWkt(grid.crs_wkt)
        raster = QgsRasterLayer(path, f"USDA texture {lab}", "gdal")
        if not raster.isValid():
            raise RuntimeError(f"cannot open {path}")
        if STYLE_PATH.is_file():
            raster.loadNamedStyle(str(STYLE_PATH))
        renderer = raster.renderer()
        if renderer is not None:
            renderer.setOpacity(TEXTURE_OPACITY)
        zones = None
        if result.settings.zones and catchment_layer is None:
            zones = _zones_layer(result.settings.zones, crs)
        if catchment_layer is not None:
            # fit the catchment with a margin
            map_crs, _ = map_crs_and_extent(
                crs, (grid.xmin, grid.ymin, grid.xmax, grid.ymax), transform_context
            )
            extent = _pad(
                _extent_in(catchment_layer, map_crs, transform_context),
                CATCHMENT_MARGIN,
            )
        else:
            # fit the texture raster (its cells with a class)
            valid = result.texture_class[lab] > 0
            rows, cols = np.nonzero(valid)
            if rows.size:
                ext = (
                    grid.xmin + cols.min() * grid.res,
                    grid.ymax - (rows.max() + 1) * grid.res,
                    grid.xmin + (cols.max() + 1) * grid.res,
                    grid.ymax - rows.min() * grid.res,
                )
            else:
                ext = (grid.xmin, grid.ymin, grid.xmax, grid.ymax)
            map_crs, extent = map_crs_and_extent(crs, ext, transform_context)
            extent = _pad(extent, RASTER_MARGIN)
        width_px = int(round(plot_width_cm / 2.54 * dpi))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)

        top_down = [
            lyr for lyr in (site_layer, catchment_layer, zones) if lyr is not None
        ]
        top_down.append(raster)
        osm_ok = False
        if osm is not None:
            osm_ok = osm_has_tiles(
                osm, extent, map_crs, transform_context, width_px, height_px, dpi
            )
            if osm_ok:
                top_down.append(osm)
            elif state["osm_ok"] is not False:
                messages.append(
                    "OpenStreetMap background not added: no map tiles were "
                    "received (check the internet connection or proxy)."
                )
            state["osm_ok"] = osm_ok
        image, rendered, errors = render_map(
            top_down, extent, map_crs, transform_context, width_px, height_px, dpi
        )
        for err in errors:
            messages.append(f"Texture map {lab}: {err}")

        tc = result.texture_class[lab]
        if result.settings.zones:
            tc = tc[result.zone_raster > 0]
        present = sorted({int(v) for v in np.unique(tc) if v > 0})
        legend = legend_entries(raster, "USDA texture", present)
        for lyr, title in (
            (site_layer, "Site boundary"),
            (catchment_layer, "Catchment boundary"),
        ):
            if lyr is not None:
                legend += legend_entries(lyr, title)
        if zones is not None:
            legend += legend_entries(zones, "Study area (zones)")
        return PlanMap(
            image=image,
            extent=rendered,
            legend=legend,
            attribution=OSM_ATTRIBUTION if osm_ok else None,
        )

    render.messages = messages  # the report lists them with the run warnings
    render.overlays = [
        name
        for lyr, name in (
            (site_layer, "the site boundary"),
            (catchment_layer, "the catchment boundary"),
        )
        if lyr is not None
    ]
    return render
