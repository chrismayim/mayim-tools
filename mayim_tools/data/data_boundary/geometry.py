"""
Data Boundary (mask polygon) - QGIS geometry steps.

Collecting the data points from the input features, the buffer method,
and the finishing steps applied to every method's polygon: fill holes,
outward offset, drop small parts, smooth, one feature or one per part.
"""

from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsGeometry,
    QgsLineString,
    QgsMultiPolygon,
    QgsPolygon,
    QgsWkbTypes,
)

BUFFER_SEGMENTS = 8
SMOOTH_ITERATIONS = 2
SMOOTH_OFFSET = 0.25


def feature_points(geom: QgsGeometry, spacing: float | None) -> np.ndarray:
    """(n, 2) vertices of a geometry. Lines and polygon outlines are first
    densified so no two vertices along them are more than ``spacing``
    apart (None: the vertices as they are)."""
    if geom is None or geom.isEmpty():
        return np.empty((0, 2))
    g = QgsGeometry(geom)
    if (
        spacing
        and spacing > 0
        and g.type()
        in (
            QgsWkbTypes.GeometryType.LineGeometry,
            QgsWkbTypes.GeometryType.PolygonGeometry,
        )
    ):
        g = g.densifyByDistance(spacing)
    pts = [(v.x(), v.y()) for v in g.vertices()]
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def polygons_to_geometry(polys) -> QgsGeometry:
    """A multipolygon geometry from core's [(shell, [holes])]."""
    multi = QgsMultiPolygon()
    for shell, holes in polys:
        poly = QgsPolygon()
        poly.setExteriorRing(QgsLineString(list(shell[:, 0]), list(shell[:, 1])))
        for h in holes:
            poly.addInteriorRing(QgsLineString(list(h[:, 0]), list(h[:, 1])))
        multi.addGeometry(poly)
    return QgsGeometry(multi)


def buffer_union(geometries, distance: float) -> QgsGeometry:
    """Every geometry buffered by ``distance`` and dissolved into one."""
    if distance <= 0:
        raise ValueError("The buffer distance must be greater than 0.")
    parts = [
        g.buffer(distance, BUFFER_SEGMENTS)
        for g in geometries
        if g is not None and not g.isEmpty()
    ]
    if not parts:
        raise ValueError("The input layer has no geometries.")
    return QgsGeometry.unaryUnion(parts)


def parts(geom: QgsGeometry) -> list[QgsGeometry]:
    """The polygon parts of a (multi)polygon geometry."""
    if geom is None or geom.isEmpty():
        return []
    if geom.isMultipart():
        return [QgsGeometry.fromPolygonXY(p) for p in geom.asMultiPolygon()]
    return [QgsGeometry(geom)]


def finish(
    geom: QgsGeometry,
    fill_holes: bool = True,
    offset: float = 0.0,
    min_area: float = 0.0,
    smooth: bool = False,
) -> list[QgsGeometry]:
    """The finished boundary parts (largest first): holes filled, offset
    outward by ``offset`` (overlapping parts then join), parts smaller
    than ``min_area`` dropped (the largest is always kept), smoothed."""
    g = geom.makeValid() if not geom.isGeosValid() else QgsGeometry(geom)
    if fill_holes:
        # a part inside another part's hole now overlaps it: dissolve
        g = QgsGeometry.unaryUnion(parts(g.removeInteriorRings()))
    if offset > 0:
        g = g.buffer(offset, BUFFER_SEGMENTS)
        if fill_holes:
            g = g.removeInteriorRings()
    out = sorted((p for p in parts(g) if p.area() > 0), key=lambda p: -p.area())
    if min_area > 0 and out:
        out = [out[0]] + [p for p in out[1:] if p.area() >= min_area]
    if smooth:
        out = [p.smooth(SMOOTH_ITERATIONS, SMOOTH_OFFSET) for p in out]
    return out


def merged(parts_list) -> QgsGeometry:
    """One multipolygon geometry of all parts."""
    g = QgsGeometry.collectGeometry(list(parts_list))
    if not g.isMultipart():
        g.convertToMultiType()
    return g
