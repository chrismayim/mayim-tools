"""QGIS-side helpers shared by the soil tool wrappers (Processing only).

Area and point collection, CRS handling, output-resolution conversion and
the band-1 display style. Imported only by *_algorithm.py modules, so the
core modules stay free of QGIS.
"""

from dataclasses import dataclass, field

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterDefinition,
    QgsUnitTypes,
)

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


def advanced(param):
    """Mark a parameter as advanced (QGIS 4 scoped enum, QGIS 3 fallback)."""
    try:
        flag = Qgis.ProcessingParameterFlag.Advanced
    except AttributeError:  # pragma: no cover - older QGIS
        flag = QgsProcessingParameterDefinition.FlagAdvanced
    param.setFlags(param.flags() | flag)
    return param


def crs_wkt(crs: QgsCoordinateReferenceSystem) -> str:
    """WKT in the variant GDAL prefers (QGIS 4 scoped enum, QGIS 3 fallback)."""
    try:
        return crs.toWkt(Qgis.CrsWktVariant.PreferredGdal)
    except AttributeError:  # pragma: no cover - older QGIS
        return crs.toWkt(QgsCoordinateReferenceSystem.WKT_PREFERRED_GDAL)


def resolution_in_crs_units(res_m: float, crs) -> float:
    """Metres -> the CRS's map units (degrees for a geographic CRS)."""
    factor = QgsUnitTypes.fromUnitToUnitFactor(Qgis.DistanceUnit.Meters, crs.mapUnits())
    return res_m * factor


# ----------------------------------------------------------------------
# Styling (in memory - no style files are written)
# ----------------------------------------------------------------------

LOAD_OPTIONS = ["All outputs", "Main layers only", "None"]
LOAD_ALL, LOAD_MAIN, LOAD_NONE = 0, 1, 2


@dataclass
class OutputLayer:
    """A raster to load and/or list in the run's .qlr layer file.
    ``classes`` (value, label) gives a categorised (paletted) style on band 1;
    otherwise band 1 is shown as stretched single-band grey."""

    path: str
    name: str
    main: bool = False
    classes: list = field(default_factory=list)


def apply_style(layer, out: OutputLayer) -> None:
    """Single-band grey (band 1, min-max stretch) or a paletted style.
    Multi-band rasters would otherwise show the first three depths as RGB."""
    from qgis.core import (
        QgsContrastEnhancement,
        QgsPalettedRasterRenderer,
        QgsSingleBandGrayRenderer,
    )
    from qgis.PyQt.QtGui import QColor

    if out.classes:
        from mayim_tools.soil._common.export import class_colour

        classes = [
            QgsPalettedRasterRenderer.Class(value, QColor(class_colour(value)), label)
            for value, label in out.classes
        ]
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
    else:
        layer.setRenderer(QgsSingleBandGrayRenderer(layer.dataProvider(), 1))
        layer.setContrastEnhancement(
            QgsContrastEnhancement.ContrastEnhancementAlgorithm.StretchToMinimumMaximum
        )
    layer.triggerRepaint()


class StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, out: OutputLayer):
        super().__init__()
        self.out = out

    def postProcessLayer(self, layer, context, feedback):
        try:
            apply_style(layer, self.out)
        except Exception as exc:  # noqa: BLE001 - styling must never fail a run
            feedback.pushWarning(f"Could not style {self.out.name}: {exc}")


def load_outputs(context, outputs, choice: int, keep_alive: list) -> int:
    """Queue outputs to load on completion, styled in memory. ``choice`` is
    LOAD_ALL / LOAD_MAIN / LOAD_NONE. Post-processors are kept alive in
    ``keep_alive`` (QGIS needs a live Python reference until they run).
    Returns the number of layers queued."""
    if choice == LOAD_NONE:
        return 0
    n = 0
    for out in outputs:
        if choice == LOAD_MAIN and not out.main:
            continue
        details = QgsProcessingContext.LayerDetails(
            out.name, context.project(), out.name
        )
        processor = StylePostProcessor(out)
        keep_alive.append(processor)
        details.setPostProcessor(processor)
        context.addLayerToLoadOnCompletion(out.path, details)
        n += 1
    return n


def write_layer_file(outputs, qlr_path: str, feedback) -> bool:
    """One QGIS layer-definition file (.qlr) for the whole run: dragging it
    into any project reloads every output with its style. Paths are stored
    relative to the .qlr, so the folder can be moved. Never fails a run."""
    try:
        from qgis.core import (
            QgsLayerDefinition,
            QgsPathResolver,
            QgsRasterLayer,
            QgsReadWriteContext,
        )

        layers = []
        for out in outputs:
            layer = QgsRasterLayer(out.path, out.name, "gdal")
            if not layer.isValid():
                continue
            apply_style(layer, out)
            layers.append(layer)
        if not layers:
            return False
        rw = QgsReadWriteContext()
        rw.setPathResolver(QgsPathResolver(qlr_path))
        doc = QgsLayerDefinition.exportLayerDefinitionLayers(layers, rw)
        with open(qlr_path, "w", encoding="utf-8") as f:
            f.write(doc.toString())
        return True
    except Exception as exc:  # noqa: BLE001 - optional convenience file
        feedback.pushWarning(f"Could not write the layer file {qlr_path}: {exc}")
        return False


def area_bounds(alg, parameters, context, out_crs, feedback, layer_param, extent_param):
    """(bounds in output CRS, bounds in lon/lat, description) from an
    extent or a polygon layer (the layer wins if both are given)."""
    source = alg.parameterAsSource(parameters, layer_param, context)
    if source is not None and source.featureCount() > 0:
        extent = source.sourceExtent()
        src_crs = source.sourceCrs()
        desc = "polygon layer extent"
        if not alg.parameterAsExtent(parameters, extent_param, context).isNull():
            feedback.pushWarning(
                "Both an extent and a polygon layer were given - using the "
                "polygon layer."
            )
    else:
        extent = alg.parameterAsExtent(parameters, extent_param, context)
        if extent.isNull() or extent.isEmpty():
            raise QgsProcessingException(
                "Area mode needs an area of interest: draw or select an extent, "
                "or give a polygon layer."
            )
        src_crs = alg.parameterAsExtentCrs(parameters, extent_param, context)
        desc = "extent"
    tc = context.transformContext()
    out_rect = QgsCoordinateTransform(src_crs, out_crs, tc).transformBoundingBox(extent)
    ll_rect = QgsCoordinateTransform(src_crs, WGS84, tc).transformBoundingBox(extent)
    bounds_out = (
        out_rect.xMinimum(),
        out_rect.yMinimum(),
        out_rect.xMaximum(),
        out_rect.yMaximum(),
    )
    bounds_ll = (
        ll_rect.xMinimum(),
        ll_rect.yMinimum(),
        ll_rect.xMaximum(),
        ll_rect.yMaximum(),
    )
    return bounds_out, bounds_ll, desc


def point_sites(
    alg, parameters, context, feedback, point_param, layer_param, name_param
):
    """[(label, lon, lat)] from a single point or a point layer (the layer
    wins if both are given)."""
    source = alg.parameterAsSource(parameters, layer_param, context)
    point = alg.parameterAsPoint(parameters, point_param, context, crs=WGS84)
    has_layer = source is not None and source.featureCount() > 0
    has_point = not point.isEmpty()
    if has_layer and has_point:
        feedback.pushWarning(
            "Both a point and a point layer were given - using the point layer."
        )
    if not has_layer and not has_point:
        raise QgsProcessingException(
            "Points mode needs a point of interest (click the map or type "
            "coordinates) OR a point layer - neither was supplied."
        )
    if not has_layer:
        return [("Point of interest", point.x(), point.y())]
    name_field = alg.parameterAsString(parameters, name_param, context)
    transform = QgsCoordinateTransform(
        source.sourceCrs(), WGS84, context.transformContext()
    )
    sites = []
    for feature in source.getFeatures():
        geom = feature.geometry()
        if geom is None or geom.isEmpty():
            continue
        pt = transform.transform(geom.asPoint())
        label = str(feature[name_field]) if name_field else f"Point {feature.id()}"
        sites.append((label, pt.x(), pt.y()))
    return sites
