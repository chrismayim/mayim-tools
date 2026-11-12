"""Tests for Data Boundary (mask polygon): core hulls and QGIS steps."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from mayim_tools.data.data_boundary.core import (  # noqa: E402
    concave_polygons,
    convex_polygon,
    point_in_ring,
    polygon_area,
    ring_area,
    unique_points,
)


def grid(x0, x1, y0, y1, step=10.0):
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def l_shape():
    """An L of points: 0-100 x 0-40 plus 0-40 x 50-100 (10 m grid)."""
    a = grid(0, 100, 0, 40)
    b = grid(0, 40, 50, 100)
    return np.vstack([a, b])


def test_unique_points_drops_duplicates_and_nan():
    xy = np.array([[0, 0], [0, 0], [1, 1], [np.nan, 2]])
    assert len(unique_points(xy)) == 2


def test_convex_hull_of_l_fills_the_corner():
    shell, holes = convex_polygon(l_shape())[0]
    assert holes == []
    assert ring_area(shell) > 0  # counter-clockwise
    # convex hull of the L: 100x40 + 40x60 + the triangle of the corner
    assert polygon_area((shell, holes)) == pytest.approx(
        100 * 100 - 0.5 * 60 * 60, rel=1e-6
    )


def test_concave_follows_the_l():
    polys = concave_polygons(l_shape(), max_edge=15.0)
    assert len(polys) == 1
    area = polygon_area(polys[0])
    # 100 x 40 + 40 x 60, plus one 10 m grid triangle across the inner
    # corner (its long edge, 14.1 m, is within 15 m)
    assert area == pytest.approx(100 * 40 + 40 * 60 + 50, rel=1e-6)
    shell = polys[0][0]
    assert not point_in_ring((80, 80), shell)  # the empty corner is out
    assert point_in_ring((20, 80), shell)


def test_concave_large_edge_equals_convex():
    xy = l_shape()
    big = concave_polygons(xy, max_edge=1e6)
    hull = convex_polygon(xy)
    assert polygon_area(big[0]) == pytest.approx(polygon_area(hull[0]), rel=1e-9)


def test_concave_keeps_a_hole():
    ring = np.vstack([grid(0, 100, 0, 20), grid(0, 100, 80, 100)])
    ring = np.vstack([ring, grid(0, 20, 30, 70), grid(80, 100, 30, 70)])
    polys = concave_polygons(ring, max_edge=15.0)
    assert len(polys) == 1
    shell, holes = polys[0]
    assert len(holes) == 1
    assert ring_area(holes[0]) < 0  # clockwise
    assert point_in_ring((50, 50), holes[0])
    # 60 x 60 m hole with its four corners cut by 10 m grid triangles
    assert polygon_area(polys[0]) == pytest.approx(100 * 100 - 3600 + 4 * 50)


def test_concave_separate_clusters():
    xy = np.vstack([grid(0, 40, 0, 40), grid(200, 220, 0, 20)])
    polys = concave_polygons(xy, max_edge=15.0)
    assert len(polys) == 2
    assert polygon_area(polys[0]) == pytest.approx(1600)  # largest first
    assert polygon_area(polys[1]) == pytest.approx(400)


def test_contour_lines_densified_stay_one_area():
    """Parallel contours 20 m apart, vertices every 5 m: a maximum edge of
    40 m (2 x spacing) keeps the whole band."""
    xs = np.arange(0, 200.01, 5.0)
    lines = [np.column_stack([xs, np.full_like(xs, y)]) for y in (0, 20, 40, 60)]
    polys = concave_polygons(np.vstack(lines), max_edge=40.0)
    assert len(polys) == 1
    assert polygon_area(polys[0]) == pytest.approx(200 * 60, rel=1e-6)


def test_errors():
    with pytest.raises(ValueError, match="At least 3"):
        concave_polygons([[0, 0], [1, 1]], 10)
    with pytest.raises(ValueError, match="straight line"):
        convex_polygon([[0, 0], [1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError, match="No area is left"):
        concave_polygons(grid(0, 100, 0, 100, step=50), 10)
    with pytest.raises(ValueError, match="greater than 0"):
        concave_polygons(l_shape(), 0)


# --------------------------------------------------------------- QGIS steps
def _qgis():
    pytest.importorskip("qgis.core")
    from mayim_tools.data.data_boundary import geometry as gm

    return gm


def test_finish_fills_holes_offsets_and_drops_small():
    gm = _qgis()
    from qgis.core import QgsGeometry

    ring = np.vstack([grid(0, 100, 0, 20), grid(0, 100, 80, 100)])
    ring = np.vstack([ring, grid(0, 20, 30, 70), grid(80, 100, 30, 70)])
    xy = np.vstack([ring, grid(300, 310, 0, 10)])  # small separate area
    geom = gm.polygons_to_geometry(concave_polygons(xy, 15.0))
    kept = gm.finish(geom, fill_holes=False)
    assert len(kept) == 2
    assert kept[0].area() == pytest.approx(10000 - 3600 + 200, rel=1e-6)
    filled = gm.finish(geom, fill_holes=True, min_area=500)
    assert len(filled) == 1
    assert filled[0].area() == pytest.approx(10000, rel=1e-6)
    grown = gm.finish(geom, fill_holes=True, offset=5, min_area=500)
    assert grown[0].area() > 10000 + 4 * 100 * 5 * 0.99
    one = gm.merged(gm.finish(geom))
    assert one.isMultipart() and len(one.asMultiPolygon()) == 2
    assert isinstance(one, QgsGeometry)


def test_feature_points_densifies_lines():
    gm = _qgis()
    from qgis.core import QgsGeometry, QgsPointXY

    line = QgsGeometry.fromPolylineXY([QgsPointXY(0, 0), QgsPointXY(100, 0)])
    assert len(gm.feature_points(line, None)) == 2
    pts = gm.feature_points(line, 10)
    assert len(pts) >= 11 and np.diff(pts[:, 0]).max() <= 10 + 1e-9


def test_buffer_union():
    gm = _qgis()
    from qgis.core import QgsGeometry, QgsPointXY

    pts = [QgsGeometry.fromPointXY(QgsPointXY(x, 0)) for x in (0, 10)]
    g = gm.buffer_union(pts, 10)
    assert not g.isMultipart() or len(g.asMultiPolygon()) == 1
    with pytest.raises(ValueError):
        gm.buffer_union(pts, 0)


def test_thin_points_one_per_cell():
    from mayim_tools.data.data_boundary.core import thin_points

    xy = grid(0, 99, 0, 99, step=1.0)  # 10000 points
    thin = thin_points(xy, 10.0)
    assert len(thin) == 100
    assert thin[:, 0].min() == 0 and thin[:, 1].min() == 0


def test_thin_for_concave_caps_the_point_count():
    from mayim_tools.data.data_boundary.core import thin_for_concave

    xy = grid(0, 999, 0, 999, step=1.0)  # 1e6 points
    pts, cell = thin_for_concave(xy, 50.0, max_points=5000)
    assert len(pts) <= 5000 and cell > 5.0
    pts, cell = thin_for_concave(xy, 50.0)
    assert cell == 5.0 and len(pts) == 200 * 200


def test_large_contour_set_stays_one_area():
    """Thinned contours give the same band as the full set."""
    from mayim_tools.data.data_boundary.core import thin_for_concave

    xs = np.arange(0, 2000.01, 1.0)
    xy = np.vstack(
        [np.column_stack([xs, np.full_like(xs, y)]) for y in range(0, 401, 20)]
    )
    pts, cell = thin_for_concave(xy, 50.0)
    polys = concave_polygons(pts, 50.0, keep_holes=False)
    assert len(polys) == 1
    assert polygon_area(polys[0]) == pytest.approx(2000 * 400, rel=0.01)


def test_densify_line():
    from mayim_tools.data.data_boundary.core import densify_line

    line = np.array([[0, 0], [100, 0], [100, 25]])
    d = densify_line(line, 10)
    assert len(d) == 10 + 3 + 1
    assert np.hypot(*np.diff(d, axis=0).T).max() <= 10 + 1e-9
    assert (d[0] == [0, 0]).all() and (d[-1] == [100, 25]).all()
    assert len(densify_line(line, None)) == 3


def test_feature_points_parts_and_rings():
    gm = _qgis()
    from qgis.core import QgsGeometry

    poly = QgsGeometry.fromWkt(
        "MultiPolygon(((0 0, 10 0, 10 10, 0 10, 0 0),(2 2, 4 2, 4 4, 2 2)),"
        "((20 0, 30 0, 30 10, 20 0)))"
    )
    assert len(gm.feature_points(poly, None)) == 5 + 4 + 4
    pts = gm.feature_points(QgsGeometry.fromWkt("MultiPoint((1 2),(3 4))"), 5)
    assert pts.tolist() == [[1, 2], [3, 4]]


def test_feature_points_3d_lines():
    gm = _qgis()
    from qgis.core import QgsGeometry

    z = QgsGeometry.fromWkt("LineStringZ(0 0 5, 10 0 5, 10 10 6)")
    assert gm.feature_points(z, None).tolist() == [[0, 0], [10, 0], [10, 10]]
    zm = QgsGeometry.fromWkt("MultiLineStringZM((0 0 5 1, 10 0 5 2))")
    assert gm.feature_points(zm, None).tolist() == [[0, 0], [10, 0]]
