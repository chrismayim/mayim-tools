"""QGIS layout maps for the Rational run-off coefficients report: one raster
(role, slope, permeability, vegetation, C1 or C 1:50) in its Mayim style
(styles/RC_*.qml, 70 % opaque) over the OpenStreetMap background, with the
catchment boundaries (black outline, labelled with the catchment ID) above
it. The report frames the map (report_maps.figure_rendered_map)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qgis.core import (
    QgsFillSymbol,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
)

from mayim_tools._common.qgis_maps import (
    legend_entries,
    osm_has_tiles,
    osm_layer,
    render_map,
)
from mayim_tools._common.report_maps import OSM_ATTRIBUTION, PlanMap
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import label_layer

STYLES = Path(__file__).resolve().parent / "styles"
RASTER_STYLES = {
    "role": "RC_land_cover_role.qml",
    "slope_class": "RC_slope_class.qml",
    "perm_class": "RC_permeability_class.qml",
    "veg_class": "RC_vegetation_class.qml",
    "c1": "RC_c_value.qml",
    "c2": "RC_c_value.qml",
    "c50": "RC_c_value.qml",
}
TITLES = {
    "role": "Role",
    "slope_class": "Slope class",
    "perm_class": "Permeability",
    "veg_class": "Vegetation",
    "c1": "C1",
    "c2": "C2",
    "c50": "C 1:50",
}
BOUNDARY = {
    "color": "0,0,0,0",
    "outline_color": "0,0,0,255",
    "outline_width": "0.4",
    "outline_width_unit": "MM",
}
MARGIN = 0.05


def style_raster(layer, key: str) -> None:
    qml = STYLES / RASTER_STYLES.get(key, "")
    if qml.is_file():
        layer.loadNamedStyle(str(qml))


def make_map_fn(
    result, catchments_layer, crs, transform_context, use_osm=True, messages=None
):
    """map_fn(key, plot width cm, dpi) -> PlanMap for the report.
    ``catchments_layer`` is a memory polygon layer with a 'catch_id' field."""
    messages = messages if messages is not None else []
    osm = osm_layer() if use_osm else None
    if use_osm and osm is None:
        messages.append("OpenStreetMap background not added: tile layer unavailable.")
    bounds = None
    if catchments_layer is not None:
        bounds = catchments_layer.clone()
        bounds.setRenderer(
            QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(BOUNDARY))
        )
        label_layer(bounds, "catch_id")
    state = {"osm": None}
    paths = result.settings.rasters

    def render(key, plot_width_cm, dpi):
        path = paths.get(key)
        if not path or not Path(path).is_file():
            return None
        raster = QgsRasterLayer(path, TITLES[key], "gdal")
        if not raster.isValid():
            raise RuntimeError(f"cannot open {path}")
        style_raster(raster, key)
        if bounds is not None and bounds.featureCount():
            e = bounds.extent()
            ext = (e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum())
        else:
            g = result.grid
            ext = (g.xmin, g.ymin, g.xmax, g.ymax)
        pad = MARGIN * max(ext[2] - ext[0], ext[3] - ext[1], 1.0)
        extent = (ext[0] - pad, ext[1] - pad, ext[2] + pad, ext[3] + pad)
        width_px = int(round(plot_width_cm / 2.54 * dpi))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)
        layers = ([bounds] if bounds is not None else []) + [raster]
        osm_ok = False
        if osm is not None:
            if state["osm"] is None:
                state["osm"] = osm_has_tiles(
                    osm, extent, crs, transform_context, width_px, height_px, dpi
                )
                if not state["osm"]:
                    messages.append(
                        "OpenStreetMap background not added: no map tiles were "
                        "received (check the internet connection or proxy)."
                    )
            osm_ok = state["osm"]
            if osm_ok:
                layers.append(osm)
        image, rendered, errors = render_map(
            layers, extent, crs, transform_context, width_px, height_px, dpi
        )
        for err in errors:
            messages.append(f"Map {TITLES[key]}: {err}")
        present = None
        pv = _present(result, key)
        if pv is not None:
            present = pv
        legend = legend_entries(raster, TITLES[key], present)
        if bounds is not None:
            legend += legend_entries(bounds, "Catchment boundary")
        return PlanMap(
            image=image,
            extent=rendered,
            legend=legend,
            attribution=OSM_ATTRIBUTION if osm_ok else None,
        )

    render.messages = messages
    return render


def _present(result, key):
    p = result.preview
    arr = {
        "role": p.get("role"),
        "slope_class": None if p.get("scls") is None else p["scls"] + 1,
        "perm_class": (
            None
            if p.get("perm") is None
            else np.where(p["role"] == 1, p["perm"] + 1, 0)
        ),
        "veg_class": None if p.get("veg") is None else p["veg"] + 1,
    }.get(key)
    if arr is None:
        return None
    return sorted({int(v) for v in np.unique(arr) if 0 < v < 255})
