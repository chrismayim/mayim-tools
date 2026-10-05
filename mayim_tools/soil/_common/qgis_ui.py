"""QGIS-side helpers shared by the soil tool wrappers (Processing only).

Area and point collection, CRS handling, output-resolution conversion and
the band-1 display style. Imported only by *_algorithm.py modules, so the
core modules stay free of QGIS.
"""

from pathlib import Path

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


class BandOnePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Multi-band rasters would otherwise load as an RGB composite of the
    first three depths, which is meaningless. Show band 1 as a stretched
    single-band grey layer instead."""

    def postProcessLayer(self, layer, context, feedback):
        try:
            from qgis.core import QgsContrastEnhancement, QgsSingleBandGrayRenderer

            layer.setRenderer(QgsSingleBandGrayRenderer(layer.dataProvider(), 1))
            layer.setContrastEnhancement(
                QgsContrastEnhancement.ContrastEnhancementAlgorithm.StretchToMinimumMaximum
            )
            layer.triggerRepaint()
        except Exception as exc:  # noqa: BLE001 - styling must never fail a run
            feedback.pushWarning(f"Could not set the display style: {exc}")


def load_rasters(context, paths, keep_alive: list, name_fn=None, restyle=True) -> None:
    """Queue rasters to load on completion. With ``restyle`` the band-1 grey
    style is applied by a post-processor (kept alive in ``keep_alive`` - QGIS
    needs a live Python reference until it runs); without it the layer keeps
    the style QGIS loads from the .qml beside the file."""
    for path in paths:
        name = Path(path).stem
        if name_fn is not None:
            name = name_fn(name)
        details = QgsProcessingContext.LayerDetails(name, context.project(), name)
        if restyle:
            processor = BandOnePostProcessor()
            keep_alive.append(processor)
            details.setPostProcessor(processor)
        context.addLayerToLoadOnCompletion(path, details)


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
