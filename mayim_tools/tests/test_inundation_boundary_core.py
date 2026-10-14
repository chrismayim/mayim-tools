"""
Tests for mayim_tools/hydrology/inundation_boundary/core.py and export.py.
Zero QGIS dependency - run under plain pytest.

Areas and volumes are checked against analytic shapes (cone, box pits);
the boundary rings against the analytic contour radius; polygon nesting
with dry islands; the two inundation modes; bounding-polygon clipping.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from mayim_tools.hydrology.inundation_boundary.core import (
    InundationError,
    fill_field,
    level_boundaries,
    nest_rings,
    run_inundation,
    seed_cells,
)
from mayim_tools.hydrology.inundation_boundary.export import (
    SUMMARY_FIELDS,
    write_summary_csv,
)

CELL = 1.0
GT = (0.0, CELL, 0.0, 100.0, 0.0, -CELL)  # origin top-left at (0, 100)


def _cone(n=101, z0=100.0, slope=0.1):
    """Cone with its apex (lowest point) at the grid centre."""
    r, c = np.mgrid[0:n, 0:n]
    centre = (n - 1) / 2
    d = np.hypot(r - centre, c - centre) * CELL
    return z0 + slope * d, centre


def _centre_xy(centre):
    return GT[0] + (centre + 0.5) * CELL, GT[3] - (centre + 0.5) * CELL


def _ring_area(ring):
    x, y = np.asarray(ring)[:, 0], np.asarray(ring)[:, 1]
    return 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def _two_pits(n=60):
    """Flat ground at 10 m with two separate 2 m deep box pits."""
    z = np.full((n, n), 10.0)
    z[10:20, 10:20] = 8.0  # 100 cells
    z[30:50, 30:45] = 8.0  # 300 cells
    return z


# ---------------------------------------------------------------------------
# Modes and seeds
# ---------------------------------------------------------------------------


def test_all_mode_floods_both_pits():
    z = _two_pits()
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 9.0)
    lv = res
    assert lv.wet_cells == 400
    assert lv.area_m2 == pytest.approx(400.0)
    assert lv.volume_m3 == pytest.approx(400.0)
    assert lv.max_depth == pytest.approx(1.0)
    assert len(lv.polygons) == 2


def test_connected_mode_floods_only_seeded_pit():
    z = _two_pits()
    x, y = GT[0] + 15.5, GT[3] - 15.5  # inside the small pit
    res = run_inundation(
        z, -9999.0, GT, (0, 0, *z.shape), 9.0, mode="connected", seeds=[(x, y)]
    )
    lv = res
    assert res.n_seeds == 1
    assert lv.wet_cells == 100
    assert len(lv.polygons) == 1
    assert not res.warnings


def test_connected_mode_without_seed_uses_lowest_cell():
    z = _two_pits()
    z[40, 40] = 7.0  # the big pit holds the lowest cell
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 9.0, mode="connected")
    assert res.wet_cells == 300
    assert any("lowest cell" in w for w in res.warnings)


def test_connected_through_saddle_once_level_rises():
    z = _two_pits()
    z[20:30, 15] = 9.5  # channel joining the pits, sill at 9.5
    z[30, 15:31] = 9.5
    seed = (GT[0] + 15.5, GT[3] - 15.5)
    window = (0, 0, *z.shape)
    low = run_inundation(z, -9999.0, GT, window, 9.0, mode="connected", seeds=[seed])
    high = run_inundation(z, -9999.0, GT, window, 9.75, mode="connected", seeds=[seed])
    assert low.wet_cells == 100
    assert high.wet_cells > 400  # both pits plus the channel


def test_seed_outside_domain_is_reported():
    z = _two_pits()
    cells, outside = seed_cells([(500.0, 500.0), (15.5, 84.5)], GT, z < 100)
    assert cells == [(15, 15)]
    assert outside == 1


def test_fill_field_modes():
    z = _two_pits()
    dom = np.ones_like(z, dtype=bool)
    assert np.array_equal(fill_field(z, dom, "all"), z)
    f = fill_field(z, dom, "connected", [(15, 15)])
    assert f[15, 15] == pytest.approx(8.0)
    assert f[40, 40] == pytest.approx(10.0)  # reached over the 10 m flat
    with pytest.raises(InundationError):
        fill_field(z, dom, "bogus")


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def test_cone_contour_matches_analytic_circle():
    z, centre = _cone()
    level = 102.0  # radius 20 m
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), level, smoothing=3)
    lv = res
    assert len(lv.lines) == 1 and len(lv.polygons) == 1
    ring = np.asarray(lv.lines[0])
    assert np.allclose(ring[0], ring[-1])
    cx, cy = _centre_xy(centre)
    radii = np.hypot(ring[:, 0] - cx, ring[:, 1] - cy)
    assert radii.mean() == pytest.approx(20.0, abs=0.3)
    assert _ring_area(ring) == pytest.approx(math.pi * 400, rel=0.02)
    # cell-based area and the cone volume pi r^2 h / 3
    assert lv.area_m2 == pytest.approx(math.pi * 400, rel=0.03)
    assert lv.volume_m3 == pytest.approx(math.pi * 400 * 2 / 3, rel=0.03)
    assert lv.max_depth == pytest.approx(2.0, abs=0.01)


def test_island_becomes_hole():
    z = np.full((60, 60), 10.0)
    z[10:50, 10:50] = 5.0
    z[25:35, 25:35] = 12.0  # island
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 8.0)
    lv = res
    assert len(lv.lines) == 2
    assert len(lv.polygons) == 1 and len(lv.polygons[0]) == 2
    outer, hole = lv.polygons[0]
    assert _ring_area(outer) > _ring_area(hole)
    assert lv.wet_cells == 1600 - 100


def test_nest_rings_three_levels():
    def square(h):
        return np.array([(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)], float)

    polys = nest_rings([square(1), square(10), square(5)])
    # 10 = water with hole 5; 1 = water again (lake on the island)
    assert len(polys) == 2
    assert len(polys[0]) == 2 and _ring_area(polys[0][1]) == pytest.approx(100)
    assert len(polys[1]) == 1 and _ring_area(polys[1][0]) == pytest.approx(4)


def test_small_rings_dropped():
    z = np.full((30, 30), 10.0)
    z[15, 15] = 5.0  # single-cell pit
    rings, polys = level_boundaries(z, 9.0, GT, 0, 3.0)
    assert rings == [] and polys == []
    rings, _ = level_boundaries(z, 9.0, GT, 0, 0.0)
    assert len(rings) == 1


def test_no_smoothing_ring_through_midpoints():
    z = np.full((20, 20), 10.0)
    z[5:10, 5:10] = 0.0
    rings, _ = level_boundaries(z, 5.0, GT, smoothing=0, min_cells=0)
    ring = np.asarray(rings[0])
    # edge exactly half way between the pit and flat cell centres
    assert ring[:, 0].min() == pytest.approx(GT[0] + 5.0)
    assert ring[:, 0].max() == pytest.approx(GT[0] + 10.0)


# ---------------------------------------------------------------------------
# Bounding polygon
# ---------------------------------------------------------------------------


def test_bounding_polygon_limits_and_closes_rings():
    z, centre = _cone()
    cx, cy = _centre_xy(centre)
    # half of the circle: the polygon covers x <= cx only
    bound = [
        [
            [
                (cx - 60, cy - 60),
                (cx, cy - 60),
                (cx, cy + 60),
                (cx - 60, cy + 60),
                (cx - 60, cy - 60),
            ]
        ]
    ]
    full = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 102.0)
    half = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 102.0, bounding=bound)
    assert half.area_m2 == pytest.approx(full.area_m2 / 2, rel=0.03)
    assert half.volume_m3 == pytest.approx(full.volume_m3 / 2, rel=0.03)
    # the ring is closed and extends at most a cell beyond the polygon edge
    ring = np.asarray(half.lines[0])
    assert np.allclose(ring[0], ring[-1])
    assert ring[:, 0].max() <= cx + CELL


def test_bounding_off_dem_raises():
    z = _two_pits()
    bound = [[[(500, 500), (510, 500), (510, 510), (500, 510), (500, 500)]]]
    with pytest.raises(InundationError):
        run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 9.0, bounding=bound)


def test_nodata_is_dry():
    z = _two_pits()
    z[12:18, 12:18] = -9999.0
    lv = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 9.0)
    assert lv.wet_cells == 400 - 36
    assert len(lv.polygons) == 2
    assert len(lv.polygons[0]) + len(lv.polygons[1]) == 3  # NoData hole


# ---------------------------------------------------------------------------
# Warnings, rasters, CSV
# ---------------------------------------------------------------------------


def test_depth_and_water_surface_rasters():
    z, _ = _cone()
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 103.0)
    depth, wse = res.depth, res.water_surface
    assert np.nanmax(depth) == pytest.approx(3.0, abs=0.01)
    assert np.nansum(depth) * CELL * CELL == pytest.approx(res.volume_m3, rel=1e-6)
    assert np.nanmin(wse) == np.nanmax(wse) == pytest.approx(103.0)
    assert np.array_equal(np.isfinite(wse), np.isfinite(depth))
    assert res.mean_depth == pytest.approx(res.volume_m3 / res.area_m2)


def test_level_must_be_finite():
    z = _two_pits()
    with pytest.raises(InundationError):
        run_inundation(z, -9999.0, GT, (0, 0, *z.shape), float("nan"))


def test_warnings_dry_and_overtopped():
    z = _two_pits()
    dry = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 7.0)
    assert dry.wet_cells == 0 and dry.mean_depth is None
    assert any("Nothing is inundated" in w for w in dry.warnings)
    over = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 11.0)
    assert any("at or above the highest ground" in w for w in over.warnings)


def test_seed_above_level_is_explained():
    z = _two_pits()
    bank = (GT[0] + 5.5, GT[3] - 5.5)  # on the 10 m flat, not in a pit
    res = run_inundation(
        z, -9999.0, GT, (0, 0, *z.shape), 9.0, mode="connected", seeds=[bank]
    )
    assert res.wet_cells == 0
    assert res.seed_ground == [10.0]
    assert any("ground at the seed point" in w for w in res.warnings)


def test_summary_csv(tmp_path):
    z = _two_pits()
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 9.0)
    path = tmp_path / "summary.csv"
    write_summary_csv(str(path), res, (401.2, 80.4, 2))
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == SUMMARY_FIELDS
    assert row["level_m"] == "9.000"
    assert row["volume_m3"] == "400.0"
    assert row["mean_depth_m"] == "1.000"
    assert row["n_polygons"] == "2"
    assert "," not in row["area_m2"]  # no thousands separators
    write_summary_csv(str(path), res)
    with open(path, encoding="utf-8") as f:
        assert next(csv.DictReader(f))["n_polygons"] == ""


# ---------------------------------------------------------------------------
# Word report
# ---------------------------------------------------------------------------


def test_depth_classes_sum_to_area():
    from mayim_tools.hydrology.inundation_boundary.report import depth_classes

    z, _ = _cone()
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 103.0)
    classes = depth_classes(res.depth, 1.0, res.weights)
    assert sum(c["area_m2"] for c in classes) == pytest.approx(res.area_m2)
    assert sum(c["pct"] for c in classes) == pytest.approx(100.0)
    assert classes[-1]["to_m"] >= 3.0 - 1e-9


def test_report_docx(tmp_path):
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    import docx

    from mayim_tools.hydrology.inundation_boundary.report import write_report_docx

    z, centre = _cone()
    cx, cy = _centre_xy(centre)
    bound = [[[(cx - 40, cy - 40), (cx + 40, cy - 40), (cx + 40, cy + 40),
               (cx - 40, cy + 40), (cx - 40, cy - 40)]]]  # fmt: skip
    res = run_inundation(
        z, -9999.0, GT, (0, 0, *z.shape), 103.0, bounding=bound,
        mode="connected", seeds=[(cx, cy)],
    )  # fmt: skip
    path = tmp_path / "report.docx"
    write_report_docx(
        str(path),
        res,
        {
            "tool": "DEM Inundation Boundary",
            "version": "0.4.0",
            "dem": {"path": "dem.tif", "crs": "EPSG:32735", "cell_x": 1, "cell_y": 1},
            "bounding": "box (1 feature(s))",
            "seeds": {"layer": "seed"},
            "outputs": [("Summary table (CSV)", "x.csv")],
            "warnings": res.warnings,
        },
        (res.area_m2, 100.0, len(res.polygons)),
        bounding=bound,
        seeds_xy=[(cx, cy)],
    )
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "DEM Inundation Boundary Assessment" in text
    assert "103.000 m" in text
    assert "range" not in text.lower()
    assert len(doc.inline_shapes) == 2  # plan, depth distribution
    cells = [c.text for t in doc.tables for r in t.rows for c in r.cells]
    assert "Water level" in cells and "103.000 m" in cells
    assert not any("," in c and c.replace(",", "").isdigit() for c in cells)


def test_report_dry_level(tmp_path):
    pytest.importorskip("docx")
    from mayim_tools.hydrology.inundation_boundary.report import write_report_docx

    z = _two_pits()
    res = run_inundation(z, -9999.0, GT, (0, 0, *z.shape), 7.0)
    path = tmp_path / "dry.docx"
    write_report_docx(str(path), res, {"warnings": res.warnings})
    assert path.exists()
