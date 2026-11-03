"""Import LandXML Surface: reader and TIN outputs (no QGIS)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from mayim_tools.data.landxml_import import tin
from mayim_tools.data.landxml_import.core import coord_geom_points, read_landxml

DATA = Path(__file__).resolve().parent / "data" / "landxml_elements.xml"
NS = "http://www.landxml.org/schema/LandXML-1.2"


def _cg(xml: str):
    return ET.fromstring(f'<CoordGeom xmlns="{NS}">{xml}</CoordGeom>')


def test_reads_every_element_and_northing_easting_order():
    d = read_landxml(str(DATA))
    assert d.epsg == 2053 and d.linear_unit == "meter" and not d.imperial
    assert d.counts() == {
        "Surfaces": 1,
        "Survey points": 3,
        "Alignments": 1,
        "Feature lines": 1,
        "Parcels": 1,
        "Pipe network structures": 2,
        "Pipe network pipes": 1,
    }
    p1 = next(p for p in d.points if p.name == "1")
    assert (p1.x, p1.y, p1.z) == (-92300.0, -2865700.0, 1505.0)  # x = easting
    assert p1.groups == ["All", "Services"] and p1.code == "FH"
    assert next(p for p in d.points if p.name == "3").z is None
    fl = d.feature_lines[0]
    assert fl.coords.shape == (3, 3) and fl.coords[0, 2] == 1506.0
    pc = d.parcels[0]
    assert np.allclose(pc.coords[0], pc.coords[-1])  # closed ring
    assert pc.attributes["area"] == 400.0 and pc.attributes["site"] == "Site 1"
    pipe = d.pipes[0].attributes
    assert pipe["size"] == 0.45 and pipe["inv_start"] == 1503.2
    assert pipe["inv_end"] == 1502.8 and pipe["from"] == "MH1"
    s = d.surfaces[0]
    assert len(s.triangles) == 2 and s.hidden_faces == 1  # hidden face dropped
    assert s.computed_area_2d() == pytest.approx(100.0)
    assert s.boundaries[0][1] == "outer"


def test_arc_is_on_the_circle_and_turns_the_right_way():
    # quarter circle, radius 10, centre (0, 0): from (10, 0) to (0, 10)
    ccw = coord_geom_points(
        _cg(
            '<Curve rot="ccw"><Start>0 10</Start><Center>0 0</Center>'
            "<End>10 0</End></Curve>"
        )
    )
    pts = np.array(ccw)
    assert np.allclose(np.hypot(pts[:, 0], pts[:, 1]), 10.0)
    assert pts[0].tolist() == [10.0, 0.0] and pts[-1].tolist() == [0.0, 10.0]
    assert np.all(pts[:, 0] >= -1e-9) and np.all(pts[:, 1] >= -1e-9)  # short way
    cw = np.array(
        coord_geom_points(
            _cg(
                '<Curve rot="cw"><Start>0 10</Start><Center>0 0</Center>'
                "<End>10 0</End></Curve>"
            )
        )
    )
    assert cw[:, 1].min() < -9.0  # the long way round (270 degrees)


def test_spiral_meets_its_ends_and_has_about_its_length():
    # clothoid L = 30, R 0 -> 100: end point from the series expansion
    L, R = 30.0, 100.0
    A2 = R * L
    x = L - L**5 / (40 * A2**2)
    y = L**3 / (6 * A2) - L**7 / (336 * A2**3)
    # left turn (ccw) along +x from the origin; LandXML order is N E
    cg = _cg(
        f'<Spiral length="{L}" radiusStart="INF" radiusEnd="{R}" rot="ccw">'
        f"<Start>0 0</Start><End>{y} {x}</End></Spiral>"
    )
    pts = np.array(coord_geom_points(cg))
    assert pts[0].tolist() == [0.0, 0.0] and np.allclose(pts[-1], [x, y])
    length = np.hypot(*np.diff(pts, axis=0).T).sum()
    assert length == pytest.approx(L, rel=1e-3)
    # curvature grows: the first half bends less than the second
    mid = len(pts) // 2
    assert pts[mid, 1] < y / 2


def test_real_civil3d_surface():
    path = Path(
        "/mnt/user-data/attachments/15cdf0a0-d95c-505b-b1c7-56e9701d6d09/"
        "d5be4611-2625038_survey_LandXML.xml"
    )
    if not path.exists():
        pytest.skip("example file not available")
    s = read_landxml(str(path)).surfaces[0]
    assert (len(s.xyz), len(s.triangles), len(s.breaklines)) == (455, 863, 57)
    # areas agree with Civil 3D's own (Definition area2DSurf / area3DSurf)
    assert s.computed_area_2d() == pytest.approx(s.area_2d, rel=1e-9)
    assert s.computed_area_3d() == pytest.approx(s.area_3d, rel=1e-9)


def test_tin_to_grid_reproduces_the_tin():
    # two triangles of the plane z = 2x + 3y + 1 over a 10 x 10 square
    xyz = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float)
    xyz[:, 2] = 2 * xyz[:, 0] + 3 * xyz[:, 1] + 1
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    arr, gt = tin.tin_to_grid(xyz, tri, 1.0)
    assert arr.shape == (10, 10) and gt == (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)
    xs = gt[0] + (np.arange(10) + 0.5)
    ys = gt[3] - (np.arange(10) + 0.5)
    gx, gy = np.meshgrid(xs, ys)
    assert np.allclose(arr, 2 * gx + 3 * gy + 1)
    # one triangle only: the other half is NoData
    arr2, _ = tin.tin_to_grid(xyz, tri[:1], 1.0)
    assert (arr2 == tin.NODATA).sum() == 45
    with pytest.raises(ValueError, match="larger cell size"):
        tin.tin_to_grid(xyz, tri, 0.001, max_cells=1000)


def test_2dm_and_outline(tmp_path):
    xyz = np.array([[0, 0, 1], [10, 0, 2], [10, 10, 3], [0, 10, 4]], float)
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    p = tmp_path / "t.2dm"
    tin.write_2dm(str(p), xyz, tri, "EG")
    lines = p.read_text().splitlines()
    assert lines[0] == "MESH2D" and lines[1] == 'MESHNAME "EG"'
    assert sum(ln.startswith("ND ") for ln in lines) == 4
    assert "E3T 2 1 3 4 1" in lines
    edges = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    u, c = np.unique(edges, axis=0, return_counts=True)
    rings = tin.outline_rings(xyz, tri, u[c == 1])
    assert len(rings) == 1 and len(rings[0]) == 5
    area = 0.5 * abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(rings[0][:-1], rings[0][1:], strict=True)
        )
    )
    assert area == pytest.approx(100.0) and math.isclose(rings[0][0][0], 0.0)
