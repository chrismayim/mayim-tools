"""
Data Boundary (mask polygon) - QGIS geometry steps.

Collecting the data points from the input features, the buffer method,
and the finishing steps applied to every method's polygon: fill holes,
outward offset, drop small parts, smooth, one feature or one per part.
"""

from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsCurvePolygon,
    QgsGeometry,
    QgsLineString,
    QgsMultiPolygon,
    QgsPoint,
    QgsPolygon,
)

from .core import densify_line

BUFFER_SEGMENTS = 8
SMOOTH_ITERATIONS = 2
SMOOTH_OFFSET = 0.25


def _curve_xy(curve) -> np.ndarray:
    """(n, 2) x, y of a curve, read from its WKB as one array (a
    point-by-point loop is slow for millions of vertices)."""
    if not isinstance(curve, QgsLineString):
        curve = curve.curveToLine()  # arcs and compound curves
    wkb = bytes(curve.asWkb())
    dims = 2 + int(curve.is3D()) + int(curve.isMeasure())
    order = "<" if wkb[0] == 1 else ">"
    # header: byte order (1), geometry type (4), number of points (4)
    vals = np.frombuffer(wkb, dtype=f"{order}f8", offset=9)
    return vals.reshape(-1, dims)[:, :2].astype(float)


def feature_points(geom: QgsGeometry, spacing: float | None) -> np.ndarray:
    """(n, 2) vertices of a geometry. Lines and polygon outlines get extra
    points so no two vertices along them are more than ``spacing`` apart
    (None: the vertices as they are). Reads the coordinates as arrays, as
    a contour layer can have millions of vertices."""
    if geom is None or geom.isEmpty():
        return np.empty((0, 2))
    out = []
    for part in geom.constParts():
        if isinstance(part, QgsPoint):
            out.append(np.array([[part.x(), part.y()]]))
            continue
        if isinstance(part, QgsCurvePolygon):
            rings = [part.exteriorRing()] + [
                part.interiorRing(i) for i in range(part.numInteriorRings())
            ]
        else:
            rings = [part]
        for ring in rings:
            if ring is None:
                continue
            xy = _curve_xy(ring)
            out.append(densify_line(xy, spacing) if spacing else xy)
    if not out:
        return np.empty((0, 2))
    return np.vstack(out)


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
