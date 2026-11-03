"""
QGIS side of the Import LandXML Surface tool: memory layers for the
LandXML elements, styles applied on load (vector QMLs in styles/, the DEM
and TIN mesh coloured with the HEC elevation colours stretched to the
surface's own range), and the report's layout map.
"""

from __future__ import annotations

from pathlib import Path

from qgis.core import (
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsHillshadeRenderer,
    QgsLineString,
    QgsMeshLayer,
    QgsPoint,
    QgsPointXY,
    QgsPolygon,
    QgsProcessingContext,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingUtils,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor

from mayim_tools._common.qgis_maps import (
    legend_entries,
    map_crs_and_extent,
    osm_has_tiles,
    osm_layer,
    render_map,
)
from mayim_tools._common.report_maps import OSM_ATTRIBUTION, PlanMap

STYLES_DIR = Path(__file__).resolve().parent / "styles"

# HEC elevation colours (Chris's 'DEM elevation (HEC)' style), spread over
# the surface's own elevation range.
HEC_COLOURS = (
    "#66cdaa",
    "#ffff00",
    "#008000",
    "#ffa500",
    "#8b0000",
    "#a52a2a",
    "#808080",
    "#fffafa",
)


def style_path(name: str) -> str:
    return str(STYLES_DIR / name)


# ----------------------------------------------------------------- layers


def _layer(geom: str, crs, fields: list[tuple[str, QMetaType.Type]], name: str):
    layer = QgsVectorLayer(f"{geom}?crs={crs.authid()}", name, "memory")
    qf = QgsFields()
    for fname, ftype in fields:
        qf.append(QgsField(fname, ftype))
    layer.dataProvider().addAttributes(qf)
    layer.updateFields()
    return layer


def _add(layer, rows):
    """rows: [(QgsGeometry, [values])]."""
    feats = []
    for geom, values in rows:
        f = QgsFeature(layer.fields())
        f.setGeometry(geom)
        f.setAttributes(list(values))
        feats.append(f)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


S, D = QMetaType.Type.QString, QMetaType.Type.Double


def _line(coords):
    if coords.shape[1] >= 3:
        return QgsGeometry(
            QgsLineString(
                [QgsPoint(float(x), float(y), float(z)) for x, y, z in coords]
            )
        )
    return QgsGeometry.fromPolylineXY(
        [QgsPointXY(float(x), float(y)) for x, y in coords]
    )


def _num(v):
    return None if v is None else float(v)


def survey_points_layer(points, crs):
    layer = _layer(
        "PointZ",
        crs,
        [
            ("name", S),
            ("code", S),
            ("desc", S),
            ("groups", S),
            ("x", D),
            ("y", D),
            ("z", D),
        ],
        "Survey points",
    )
    return _add(
        layer,
        [
            (
                QgsGeometry(QgsPoint(p.x, p.y, p.z if p.z is not None else 0.0)),
                [p.name, p.code, p.desc, ", ".join(p.groups)[:254], p.x, p.y, p.z],
            )
            for p in points
        ],
    )


def tin_points_layer(surface, crs):
    layer = _layer("PointZ", crs, [("id", S), ("z", D)], f"{surface.name} TIN points")
    return _add(
        layer,
        [
            (QgsGeometry(QgsPoint(float(x), float(y), float(z))), [pid, float(z)])
            for pid, (x, y, z) in zip(surface.point_ids, surface.xyz, strict=True)
        ],
    )


def breaklines_layer(surface, crs):
    layer = _layer(
        "LineStringZ", crs, [("desc", S), ("kind", S)], f"{surface.name} breaklines"
    )
    rows = [(_line(c), [d, "breakline"]) for d, c in surface.breaklines]
    rows += [
        (_line(c), [n or t, f"boundary {t}".strip()]) for n, t, c in surface.boundaries
    ]
    return _add(layer, rows)


def tin_boundary_layer(surface, rings, crs):
    layer = _layer("Polygon", crs, [("surface", S)], f"{surface.name} TIN boundary")
    rows = []
    for ring in rings:
        poly = QgsPolygon(QgsLineString([QgsPoint(x, y) for x, y in ring]))
        rows.append((QgsGeometry(poly), [surface.name]))
    return _add(layer, rows)


def alignments_layer(items, crs):
    layer = _layer(
        "LineString",
        crs,
        [("name", S), ("length", D), ("sta_start", D), ("desc", S)],
        "Alignments",
    )
    return _add(
        layer,
        [
            (
                _line(a.coords),
                [
                    a.name,
                    _num(a.attributes.get("length")),
                    _num(a.attributes.get("sta_start")),
                    a.attributes.get("desc", ""),
                ],
            )
            for a in items
        ],
    )


def feature_lines_layer(items, crs):
    layer = _layer(
        "LineStringZ", crs, [("name", S), ("group", S), ("desc", S)], "Feature lines"
    )
    rows = []
    for f in items:
        coords = f.coords
        if coords.shape[1] == 2:
            import numpy as np

            coords = np.column_stack([coords, np.zeros(len(coords))])
        rows.append(
            (
                _line(coords),
                [f.name, f.attributes.get("group", ""), f.attributes.get("desc", "")],
            )
        )
    return _add(layer, rows)


def parcels_layer(items, crs):
    layer = _layer(
        "Polygon",
        crs,
        [("name", S), ("site", S), ("type", S), ("area_m2", D), ("desc", S)],
        "Parcels",
    )
    rows = []
    for p in items:
        geom = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in p.coords]])
        area = p.attributes.get("area")
        rows.append(
            (
                geom,
                [
                    p.name,
                    p.attributes.get("site", ""),
                    p.attributes.get("type", ""),
                    _num(area) if area is not None else geom.area(),
                    p.attributes.get("desc", ""),
                ],
            )
        )
    return _add(layer, rows)


def structures_layer(items, crs):
    layer = _layer(
        "Point",
        crs,
        [("name", S), ("network", S), ("rim_m", D), ("sump_m", D), ("desc", S)],
        "Pipe structures",
    )
    return _add(
        layer,
        [
            (
                QgsGeometry.fromPointXY(QgsPointXY(s.x, s.y)),
                [
                    s.name,
                    s.attributes.get("network", ""),
                    _num(s.attributes.get("rim_m")),
                    _num(s.attributes.get("sump_m")),
                    s.attributes.get("desc", ""),
                ],
            )
            for s in items
        ],
    )


def pipes_layer(items, crs):
    layer = _layer(
        "LineString",
        crs,
        [
            ("name", S),
            ("network", S),
            ("from", S),
            ("to", S),
            ("shape", S),
            ("size_m", D),
            ("length_m", D),
            ("slope", D),
            ("inv_start", D),
            ("inv_end", D),
            ("desc", S),
        ],
        "Pipes",
    )
    rows = []
    for p in items:
        a = p.attributes
        rows.append(
            (
                _line(p.coords),
                [
                    p.name,
                    a.get("network", ""),
                    a.get("from", ""),
                    a.get("to", ""),
                    a.get("shape", ""),
                    _num(a.get("size")),
                    _num(a.get("length")),
                    _num(a.get("slope")),
                    _num(a.get("inv_start")),
                    _num(a.get("inv_end")),
                    a.get("desc", ""),
                ],
            )
        )
    return _add(layer, rows)


# ----------------------------------------------------------------- styling


def elevation_shader(zmin: float, zmax: float) -> QgsRasterShader:
    if zmax <= zmin:
        zmax = zmin + 1.0
    n = len(HEC_COLOURS)
    ramp = QgsColorRampShader(zmin, zmax)
    ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
    items = []
    for k, colour in enumerate(HEC_COLOURS):
        value = zmin + (zmax - zmin) * k / (n - 1)
        items.append(
            QgsColorRampShader.ColorRampItem(value, QColor(colour), f"{value:.1f} m")
        )
    ramp.setColorRampItemList(items)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    return shader


def style_dem(layer, zmin: float, zmax: float) -> None:
    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(), 1, elevation_shader(zmin, zmax)
    )
    renderer.setClassificationMin(zmin)
    renderer.setClassificationMax(zmax)
    layer.setRenderer(renderer)


def style_mesh(layer, zmin: float, zmax: float) -> None:
    settings = layer.rendererSettings()
    native = settings.nativeMeshSettings()
    native.setEnabled(True)
    native.setColor(QColor(60, 60, 60))
    native.setLineWidth(0.1)
    settings.setNativeMeshSettings(native)
    groups = layer.datasetGroupsIndexes()
    if groups:
        group = groups[0]
        scalar = settings.scalarSettings(group)
        ramp = QgsColorRampShader(zmin, zmax)
        ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
        n = len(HEC_COLOURS)
        ramp.setColorRampItemList(
            [
                QgsColorRampShader.ColorRampItem(
                    zmin + (zmax - zmin) * k / (n - 1), QColor(c)
                )
                for k, c in enumerate(HEC_COLOURS)
            ]
        )
        scalar.setColorRampShader(ramp)
        scalar.setClassificationMinimumMaximum(zmin, zmax)
        settings.setScalarSettings(group, scalar)
        settings.setActiveScalarDatasetGroup(group)
    layer.setRendererSettings(settings)


class QmlStyler(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, qml: str):
        super().__init__()
        self.qml = qml

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            layer.loadNamedStyle(style_path(self.qml))
            layer.triggerRepaint()
        except Exception as e:  # styling never fails the run
            feedback.pushWarning(f"Style {self.qml} not applied: {e}")


class ElevationStyler(QgsProcessingLayerPostProcessorInterface):
    """DEM raster or TIN mesh: HEC elevation colours over [zmin, zmax]; the
    mesh also gets its CRS (a 2DM file carries none)."""

    def __init__(self, zmin: float, zmax: float, crs=None):
        super().__init__()
        self.zmin, self.zmax, self.crs = zmin, zmax, crs

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            if isinstance(layer, QgsMeshLayer):
                if self.crs is not None and self.crs.isValid():
                    layer.setCrs(self.crs)
                style_mesh(layer, self.zmin, self.zmax)
            elif isinstance(layer, QgsRasterLayer):
                style_dem(layer, self.zmin, self.zmax)
            layer.triggerRepaint()
        except Exception as e:
            feedback.pushWarning(f"Elevation colours not applied: {e}")


def load_on_completion(context, path, name, processor, keep_alive, hint=None):
    details = QgsProcessingContext.LayerDetails(name, context.project(), name)
    if hint is not None:
        details.layerTypeHint = hint
    if processor is not None:
        details.setPostProcessor(processor)
        keep_alive.append(processor)
    context.addLayerToLoadOnCompletion(path, details)


MESH_HINT = QgsProcessingUtils.LayerHint.Mesh


def crs_from(data, override) -> tuple[QgsCoordinateReferenceSystem, str]:
    """(CRS, how it was found)."""
    if override is not None and override.isValid():
        return override, "set in the tool"
    if data.epsg:
        crs = QgsCoordinateReferenceSystem(f"EPSG:{data.epsg}")
        if crs.isValid():
            return crs, f"from the file (EPSG:{data.epsg})"
    if data.crs_wkt:
        crs = QgsCoordinateReferenceSystem.fromWkt(data.crs_wkt)
        if crs.isValid():
            return crs, "from the file (WKT)"
    return QgsCoordinateReferenceSystem(), "not known"


# ----------------------------------------------------------------- report map


def make_plan_map(
    crs, transform_context, dem_paths, vector_layers, use_osm=True, messages=None
):
    """Report PlanMapFn (surface name or None, plot width cm, dpi) ->
    PlanMap. ``dem_paths`` {surface name: (tif path, zmin, zmax)};
    ``vector_layers`` [(layer, legend title, qml, surface name or None)],
    drawn top first; a layer tied to a surface is shown on that surface's
    map only, the others on every map."""
    messages = messages if messages is not None else []
    osm = osm_layer() if use_osm else None
    styled = []
    for layer, title, qml, surface in vector_layers:
        if layer is None or layer.featureCount() == 0:
            continue
        clone = layer.clone()
        if qml:
            clone.loadNamedStyle(style_path(qml))
        clone.setLabelsEnabled(False)
        styled.append((clone, title, surface))

    def render(surface_name, plot_width_cm, dpi):
        layers = [(lyr, t) for lyr, t, s in styled if s in (None, surface_name)]
        dem = None
        if surface_name in dem_paths:
            path, zmin, zmax = dem_paths[surface_name]
            dem = QgsRasterLayer(path, "DEM")
            if dem.isValid():
                style_dem(dem, zmin, zmax)
                dem.renderer().setOpacity(0.85)
                shade = QgsRasterLayer(path, "Hillshade")
                shade.setRenderer(
                    QgsHillshadeRenderer(shade.dataProvider(), 1, 315, 45)
                )
                shade.renderer().setOpacity(0.35)
            else:
                dem, shade = None, None
        boxes = [lyr.extent() for lyr, _t in layers]
        if dem is not None:
            boxes.append(dem.extent())
        if not boxes:
            return None
        box = [
            min(b.xMinimum() for b in boxes),
            min(b.yMinimum() for b in boxes),
            max(b.xMaximum() for b in boxes),
            max(b.yMaximum() for b in boxes),
        ]
        pad = 0.05 * max(box[2] - box[0], box[3] - box[1], 1.0)
        box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        map_crs, extent = map_crs_and_extent(crs, box, transform_context)
        width_px = int(round(plot_width_cm / 2.54 * dpi))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)
        top_down = [lyr for lyr, _t in layers]
        if dem is not None:
            top_down += [shade, dem]
        osm_ok = False
        if osm is not None:
            osm_ok = osm_has_tiles(
                osm, extent, map_crs, transform_context, width_px, height_px, dpi
            )
            if osm_ok:
                top_down.append(osm)
            else:
                messages.append(
                    "OpenStreetMap background not added: no map tiles were "
                    "received (check the internet connection or proxy)."
                )
        image, rendered, errors = render_map(
            top_down, extent, map_crs, transform_context, width_px, height_px, dpi
        )
        messages.extend(f"Layout map: {e}" for e in errors)
        legend = []
        if dem is not None:
            legend += legend_entries(dem, "Elevation")
        for lyr, title in layers:
            legend += legend_entries(lyr, title)
        return PlanMap(
            image=image,
            extent=rendered,
            legend=legend,
            attribution=OSM_ATTRIBUTION if osm_ok else None,
        )

    return render
