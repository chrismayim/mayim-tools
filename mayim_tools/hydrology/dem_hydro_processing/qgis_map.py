"""
QGIS side of the DEM Hydrological Processing report's layout map (report and
figure standard, section 3): the elevation change raster (corrected minus
original) and the remaining sinks, drawn by QGIS with the tool's QML styles
(styles/DEM_elevation_change.qml, styles/DEM_remaining_sinks.qml), with the
optional study area boundary and DEM contours, over the OpenStreetMap
background. The frame around the map (axes, legend, scale bar, north arrow)
is composed by mayim_tools._common.report_maps.

Also holds the small QGIS helpers the algorithm uses to write the sinks
shapefile and load the written layers with their styles.
"""

from __future__ import annotations

import os
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsMemoryProviderUtils,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingLayerPostProcessorInterface,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorFileWriter,
)
from qgis.PyQt.QtCore import QMetaType

from mayim_tools._common.output_files import shapefile_field_names
from mayim_tools._common.qgis_maps import (
    STUDY_AREA_SYMBOL,
    contour_layers,
    legend_entries,
    map_crs_and_extent,
    osm_has_tiles,
    osm_layer,
    render_map,
)
from mayim_tools._common.report_maps import OSM_ATTRIBUTION, PlanMap

STYLES_DIR = Path(__file__).resolve().parent / "styles"
STYLE_CHANGE = STYLES_DIR / "DEM_elevation_change.qml"
STYLE_SINKS = STYLES_DIR / "DEM_remaining_sinks.qml"
MAP_PAD_FRACTION = 0.02

SINK_FIELDS = (  # (name, type, length; 0 = default)
    ("sink_id", QMetaType.Type.Int, 0),
    ("kind", QMetaType.Type.QString, 20),
    ("reason", QMetaType.Type.QString, 200),
    ("cells", QMetaType.Type.Int, 0),
    ("elev_m", QMetaType.Type.Double, 0),
)


def apply_qml(layer, qml: Path, messages: list | None = None) -> None:
    if layer is None:
        return
    if not qml.exists():
        if messages is not None:
            messages.append(f"Style {qml.name} not found.")
        return
    _msg, ok = layer.loadNamedStyle(str(qml))
    if not ok and messages is not None:
        messages.append(f"Style {qml.name} could not be loaded.")


class QmlPostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Applies a QML style once Processing has loaded the layer."""

    def __init__(self, qml: Path):
        super().__init__()
        self.qml = qml

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            apply_qml(layer, self.qml)
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style {self.qml.name} not applied: {e}")


def load_on_completion(context, path: str, qml: Path | None, keep_alive: list):
    """Adds a written file to the project when the run finishes, styled with
    ``qml`` (the post-processor is kept alive in ``keep_alive``)."""
    name = Path(path).stem
    details = QgsProcessingContext.LayerDetails(name, context.project(), name)
    if qml is not None:
        processor = QmlPostProcessor(qml)
        details.setPostProcessor(processor)
        keep_alive.append(processor)
    context.addLayerToLoadOnCompletion(path, details)


def save_shapefile(layer, path: str, transform_context) -> str:
    """Writes a memory layer to a shapefile, replacing an existing one; field
    names longer than 10 characters are shortened explicitly."""
    if os.path.exists(path) and not QgsVectorFileWriter.deleteShapeFile(path):
        raise RuntimeError(
            f"{Path(path).name} exists and could not be replaced - it is "
            "probably open in QGIS. Remove it from the project and run again."
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
    if result[0] != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(f"{Path(path).name} could not be written: {result[1]}")
    return path


def copy_source_layer(source, name, crs, transform_context):
    """Memory copy of a Processing feature source, reprojected to ``crs``."""
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


def sinks_layer(sinks, geotransform, crs):
    """Memory point layer of the remaining sink clusters (lowest cell of
    each), field names already within the shapefile limit."""
    fields = QgsFields()
    for name, qtype, length in SINK_FIELDS:
        field = QgsField(name, qtype)
        if length:
            field.setLength(length)
        fields.append(field)
    layer = QgsMemoryProviderUtils.createMemoryLayer(
        "Remaining sinks", fields, Qgis.WkbType.Point, crs
    )
    gt = geotransform
    feats = []
    for k in sinks:
        x = gt[0] + (k.col + 0.5) * gt[1]
        y = gt[3] + (k.row + 0.5) * gt[5]
        feat = QgsFeature(fields)
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        feat.setAttributes(
            [
                int(k.sink_id),
                str(k.kind),
                str(k.reason),
                int(k.cells),
                float(k.elevation),
            ]
        )
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


def dem_extent(geotransform, shape):
    """(xmin, ymin, xmax, ymax) of a north-up raster."""
    gt = geotransform
    rows, cols = shape
    xs = (gt[0], gt[0] + cols * gt[1])
    ys = (gt[3], gt[3] + rows * gt[5])
    return (min(xs), min(ys), max(xs), max(ys))


def make_plan_map(
    change_path: str,
    sinks,
    crs,
    transform_context,
    geotransform,
    shape,
    study_area=None,
    contours=None,
    contour_interval: float = 0.0,
    present_classes=None,
    use_osm: bool = True,
    messages: list | None = None,
):
    """Returns the report's PlanMapFn (item label, plot width cm, dpi) ->
    PlanMap. ``change_path`` is a GeoTIFF of the elevation change (corrected
    minus original); ``sinks`` the remaining sink memory layer (or None);
    ``study_area`` a memory polygon layer in ``crs`` (or None);
    ``contours(extent)`` returns (minor, major) DEM contour lines
    [(level, [(x, y), ...])] for an extent in ``crs``, ``contour_interval``
    being the minor interval (major = 5 x minor; 0 = no contours); ``present_classes`` the labels of the change
    classes that occur (legend)."""
    messages = messages if messages is not None else []
    osm = osm_layer() if use_osm else None
    if use_osm and osm is None:
        messages.append("OpenStreetMap background not added: tile layer unavailable.")

    def render(_label, plot_width_cm, dpi):
        box = dem_extent(geotransform, shape)
        if study_area is not None and study_area.featureCount():
            sa = study_area.extent()
            box = (
                min(box[0], sa.xMinimum()),
                min(box[1], sa.yMinimum()),
                max(box[2], sa.xMaximum()),
                max(box[3], sa.yMaximum()),
            )
        pad = MAP_PAD_FRACTION * max(box[2] - box[0], box[3] - box[1])
        box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        map_crs, extent = map_crs_and_extent(crs, box, transform_context)

        width_px = int(round(plot_width_cm / 2.54 * dpi))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)

        change = QgsRasterLayer(change_path, "Elevation change")
        if not change.isValid():
            messages.append(
                "Layout map: the elevation change raster could not be read."
            )
            change = None
        apply_qml(change, STYLE_CHANGE, messages)

        sink_map = None
        if sinks is not None and sinks.featureCount():
            sink_map = sinks.clone()
            apply_qml(sink_map, STYLE_SINKS, messages)
        if study_area is not None and study_area.featureCount():
            study_area.setRenderer(
                QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(STUDY_AREA_SYMBOL))
            )
        contour_sets = []  # [(layer, legend label)], major first
        if contours is not None and contour_interval > 0:
            minor, major = contours(dem_extent(geotransform, shape))
            contour_sets = contour_layers(minor, major, crs, contour_interval)

        # draw order: first in the list is on top
        top_down = []
        if sink_map is not None:
            top_down.append(sink_map)
        if study_area is not None and study_area.featureCount():
            top_down.append(study_area)
        if change is not None:
            top_down.append(change)
        top_down += [layer for layer, _label in contour_sets]
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
        for err in errors:
            messages.append(f"Layout map: {err}")

        legend = []
        if change is not None:
            legend += legend_entries(change, "Elevation change", present_classes)
        if sink_map is not None:
            kinds = {f["kind"] for f in sink_map.getFeatures()}
            legend += legend_entries(sink_map, "Remaining sink", kinds)
        if study_area is not None and study_area.featureCount():
            legend += legend_entries(study_area, "Study area")
        for layer, label in reversed(contour_sets):  # minor, then major
            legend += legend_entries(layer, label)
        return PlanMap(
            image=image,
            extent=rendered,
            legend=legend,
            attribution=OSM_ATTRIBUTION if osm_ok else None,
        )

    return render
