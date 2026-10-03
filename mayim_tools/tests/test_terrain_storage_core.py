"""
Tests for mayim_tools/hydrology/terrain_storage/core.py and export.py.
Zero QGIS dependency - run under plain pytest.

Volumes are checked against hand calculations on box-shaped pits and
against a brute-force cell-by-cell sum; the priority flood is checked
against an independent iterative (Bellman-Ford style) minimax solver.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from mayim_tools.hydrology.terrain_storage.core import (
    StageStorage,
    StorageArea,
    StorageError,
    compute_storage,
    coverage_fraction,
    edge_cells,
    mosaic_depths,
    polygon_window,
    priority_flood_levels,
    stage_levels,
    window_geotransform,
)
from mayim_tools.hydrology.terrain_storage.export import (
    write_hec_table,
    write_stage_table_csv,
)

CELL = 2.0
GT = (500.0, CELL, 0.0, 900.0, 0.0, -CELL)
NODATA = -9999.0


def rect(c0, r0, c1, r1, gt=GT):
    """Closed anticlockwise rectangle ring from cell-index corners."""
    x0, x1 = gt[0] + c0 * gt[1], gt[0] + c1 * gt[1]
    y0, y1 = gt[3] + r0 * gt[5], gt[3] + r1 * gt[5]
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0), (x0, y1)]


def run(z, polygons, interval=0.5, **kw):
    z = np.asarray(z, dtype=np.float64)
    area = StorageArea("S1", polygons, 0)
    window = (0, 0, *z.shape)
    return compute_storage(z, NODATA, GT, window, area, interval, **kw)


def box_pit(rows=11, cols=11, rim=10.0, floor=0.0, pit=(3, 3, 8, 8)):
    """Flat ground at ``rim`` with a vertical-sided pit at ``floor``."""
    z = np.full((rows, cols), rim)
    r0, c0, r1, c1 = pit
    z[r0:r1, c0:c1] = floor
    return z


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_aligned_square_is_exact():
    cov = coverage_fraction([[rect(2, 3, 6, 5)]], GT, (8, 8), supersample=8)
    expected = np.zeros((8, 8))
    expected[3:5, 2:6] = 1.0
    np.testing.assert_allclose(cov, expected)


def test_coverage_half_cells():
    # polygon edge through the middle of column 4
    ring = [
        (GT[0] + 1 * CELL, GT[3] - 4 * CELL),
        (GT[0] + 4.5 * CELL, GT[3] - 4 * CELL),
        (GT[0] + 4.5 * CELL, GT[3] - 1 * CELL),
        (GT[0] + 1 * CELL, GT[3] - 1 * CELL),
        (GT[0] + 1 * CELL, GT[3] - 4 * CELL),
    ]
    cov = coverage_fraction([[ring]], GT, (6, 6), supersample=8)
    np.testing.assert_allclose(cov[1:4, 4], 0.5)
    np.testing.assert_allclose(cov[1:4, 1:4], 1.0)
    assert cov.sum() * CELL**2 == pytest.approx(3.5 * 3 * CELL**2)


def test_coverage_triangle_area_and_centre_rule():
    ring = [
        (GT[0] + 0.3 * CELL, GT[3] - 9.7 * CELL),
        (GT[0] + 9.1 * CELL, GT[3] - 8.2 * CELL),
        (GT[0] + 4.4 * CELL, GT[3] - 0.6 * CELL),
        (GT[0] + 0.3 * CELL, GT[3] - 9.7 * CELL),
    ]
    a = np.asarray(ring)
    exact = 0.5 * abs(np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1]))
    fine = coverage_fraction([[ring]], GT, (10, 10), supersample=16)
    assert fine.sum() * CELL**2 == pytest.approx(exact, rel=0.01)
    centre = coverage_fraction([[ring]], GT, (10, 10), supersample=1)
    assert set(np.unique(centre)) <= {0.0, 1.0}
    assert centre.sum() * CELL**2 == pytest.approx(exact, rel=0.15)


def test_coverage_hole_and_multipart():
    outer = rect(0, 0, 6, 6)
    hole = rect(2, 2, 4, 4)[::-1]
    other = rect(7, 7, 9, 9)
    cov = coverage_fraction([[outer, hole], [other]], GT, (10, 10), supersample=4)
    assert cov.sum() == pytest.approx(36 - 4 + 4)
    assert cov[2:4, 2:4].sum() == 0
    assert cov[7:9, 7:9].sum() == 4


def test_coverage_unclosed_ring_is_closed():
    ring = rect(1, 1, 3, 3)[:-1]
    cov = coverage_fraction([[ring]], GT, (4, 4), supersample=2)
    assert cov.sum() == pytest.approx(4)


def test_polygon_window_pads_and_clips():
    assert polygon_window([[rect(2, 3, 6, 5)]], GT, (20, 20)) == (2, 1, 4, 6)
    assert polygon_window([[rect(-5, -5, 2, 2)]], GT, (20, 20)) == (0, 0, 3, 3)
    assert polygon_window([[rect(30, 30, 32, 32)]], GT, (20, 20)) is None


def test_window_geotransform():
    gt = window_geotransform(GT, 3, 4)
    assert gt[0] == GT[0] + 4 * CELL and gt[3] == GT[3] - 3 * CELL


# ---------------------------------------------------------------------------
# Priority flood
# ---------------------------------------------------------------------------


def brute_force_levels(z, domain, seed):
    rows, cols = z.shape
    level = np.full(z.shape, np.inf)
    level[seed] = z[seed]
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if not domain[r, c]:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (dr or dc) and 0 <= rr < rows and 0 <= cc < cols:
                            if domain[rr, cc]:
                                cand = max(z[r, c], level[rr, cc])
                                if cand < level[r, c]:
                                    level[r, c] = cand
                                    changed = True
    return level


def test_priority_flood_matches_brute_force():
    rng = np.random.default_rng(42)
    z = rng.uniform(0, 10, (12, 14))
    domain = rng.uniform(size=z.shape) > 0.15
    seed = tuple(int(i) for i in np.argwhere(domain)[0])
    fast, _ = priority_flood_levels(z, domain, seed)
    slow = brute_force_levels(z, domain, seed)
    np.testing.assert_array_equal(np.isinf(fast), np.isinf(slow))
    np.testing.assert_allclose(fast[np.isfinite(fast)], slow[np.isfinite(slow)])


def test_priority_flood_spill_is_lowest_rim():
    z = box_pit()
    z[0:3, 5] = 7.0  # notch through the rim to the polygon edge
    domain = np.ones(z.shape, bool)
    domain[[0, -1], :] = False
    domain[:, [0, -1]] = False
    edge = edge_cells(domain, ~domain)
    levels, spill = priority_flood_levels(z, domain, (5, 5), edge=edge)
    assert spill == (1, 5)
    assert levels[spill] == 7.0


def test_priority_flood_rejects_seed_outside_domain():
    with pytest.raises(StorageError):
        priority_flood_levels(np.zeros((3, 3)), np.zeros((3, 3), bool), (1, 1))


# ---------------------------------------------------------------------------
# Stages and table
# ---------------------------------------------------------------------------


def test_stage_levels_round_numbers():
    assert stage_levels(101.23, 102.0, 0.25) == [101.23, 101.25, 101.5, 101.75, 102.0]
    assert stage_levels(5.0, 6.0, 0.5) == [5.0, 5.5, 6.0]
    assert stage_levels(5.0, 5.0, 0.5) == [5.0]
    with pytest.raises(StorageError):
        stage_levels(0, 1, 0)
    with pytest.raises(StorageError):
        stage_levels(0, 1000, 0.001)


def test_stage_storage_matches_brute_force():
    rng = np.random.default_rng(7)
    z = rng.uniform(0, 5, 200)
    w = rng.uniform(0, 1, 200)
    s = StageStorage(z, w, z, 4.0)
    for h in (0.0, 1.3, 2.5, 4.99, 6.0):
        wet = z < h
        area, vol, n = s.at(h)
        assert area == pytest.approx(4.0 * w[wet].sum())
        assert vol == pytest.approx(4.0 * (w[wet] * (h - z[wet])).sum(), abs=1e-9)
        assert n == wet.sum()


# ---------------------------------------------------------------------------
# compute_storage
# ---------------------------------------------------------------------------


def test_box_pit_hand_calculation():
    z = box_pit()
    z[0:3, 5] = 6.0  # notch through the rim: spill at 6 m
    res = run(z, [[rect(1, 1, 10, 10)]], interval=1.0)
    assert res.floor_z == 0.0
    assert res.spill_z == 6.0 and res.top_source == "spill"
    assert res.top_stage == 6.0
    stages = [row["stage_m"] for row in res.table]
    assert stages == [0, 1, 2, 3, 4, 5, 6]
    pit_area = 25 * CELL**2
    for row in res.table:
        assert row["area_m2"] == pytest.approx(pit_area if row["stage_m"] > 0 else 0)
        assert row["volume_m3"] == pytest.approx(pit_area * row["stage_m"])
    assert res.table[3]["inc_vol_m3"] == pytest.approx(pit_area)
    assert res.attributes["vol_ML"] == pytest.approx(pit_area * 6 / 1000)
    # spill reported in the notch, where the pool leaves the polygon
    assert res.spill_xy == pytest.approx((GT[0] + 5.5 * CELL, GT[3] - 1.5 * CELL))
    assert not res.warnings


def test_terraced_pit_area_steps():
    z = box_pit()
    z[4:7, 4:7] = -2.0  # deeper 3x3 sump
    res = run(z, [[rect(1, 1, 10, 10)]], interval=1.0)
    by_stage = {row["stage_m"]: row for row in res.table}
    assert res.floor_z == -2.0 and res.top_stage == 10.0
    assert by_stage[-1.0]["area_m2"] == pytest.approx(9 * CELL**2)
    assert by_stage[0.0]["volume_m3"] == pytest.approx(9 * CELL**2 * 2)
    assert by_stage[1.0]["area_m2"] == pytest.approx(25 * CELL**2)
    assert by_stage[1.0]["volume_m3"] == pytest.approx(CELL**2 * (9 * 3 + 16 * 1))


def test_generous_polygon_finds_crest_not_outer_toe():
    # rim at 10 (ring of cells), ground falls to 2 outside it, polygon well beyond
    z = np.full((15, 15), 2.0)
    z[3:12, 3:12] = 10.0
    z[5:10, 5:10] = 0.0
    z[3:5, 7] = 8.0  # notch: lowest point on the crest
    res = run(z, [[rect(1, 1, 14, 14)]], interval=1.0)
    assert res.spill_z == 8.0
    assert res.spill_xy == pytest.approx((GT[0] + 7.5 * CELL, GT[3] - 3.5 * CELL))
    assert res.table[-1]["volume_m3"] == pytest.approx(25 * CELL**2 * 8)
    assert not any("cuts through" in w for w in res.warnings)
    # level-pool "all cells" counts the outer toe below 8 m too - and says so
    res_all = run(z, [[rect(1, 1, 14, 14)]], interval=1.0, mode="all")
    assert res_all.table[-1]["volume_m3"] > res.table[-1]["volume_m3"]
    assert any("not connected" in w for w in res_all.warnings)


def test_tight_polygon_is_flagged():
    z = np.full((11, 11), 10.0)
    for r in range(11):
        for c in range(11):
            z[r, c] = max(abs(r - 5), abs(c - 5)) * 2.0  # pyramid bowl
    res = run(z, [[rect(3, 3, 8, 8)]], interval=0.5)  # inside the 10 m crest
    assert res.spill_z == pytest.approx(4.0)
    assert any("cuts through" in w for w in res.warnings)


def test_connected_mode_excludes_hollow_behind_ridge():
    z = np.full((9, 15), 10.0)
    z[2:7, 2:6] = 0.0  # main pond, floor 0
    z[2:7, 6] = 5.0  # ridge
    z[2:7, 7:12] = 2.0  # second hollow, floor 2
    poly = [[rect(1, 1, 13, 8)]]
    con = run(z, poly, interval=1.0)
    allc = run(z, poly, interval=1.0, mode="all")
    pond, hollow, ridge = 20 * CELL**2, 25 * CELL**2, 5 * CELL**2
    vc = {r["stage_m"]: r["volume_m3"] for r in con.table}
    va = {r["stage_m"]: r["volume_m3"] for r in allc.table}
    assert vc[4.0] == pytest.approx(pond * 4)
    assert va[4.0] == pytest.approx(pond * 4 + hollow * 2)
    # above the ridge both agree
    assert vc[10.0] == pytest.approx(va[10.0])
    assert vc[10.0] == pytest.approx(pond * 10 + ridge * 5 + hollow * 8)


def test_user_max_stage_above_and_below_spill():
    z = box_pit()
    z[0:3, 5] = 6.0
    poly = [[rect(1, 1, 10, 10)]]
    lower = run(z, poly, interval=1.0, max_stage=3.0)
    assert lower.top_stage == 3.0 and lower.top_source == "user"
    assert not lower.warnings
    higher = run(z, poly, interval=1.0, max_stage=8.0)
    assert any("vertical wall" in w for w in higher.warnings)
    assert higher.table[-1]["volume_m3"] == pytest.approx(
        25 * CELL**2 * 8 + 2 * CELL**2 * 2
    )
    with pytest.raises(StorageError):
        run(z, poly, max_stage=-1.0)


def test_floor_on_edge_fails_no_depression():
    z = np.tile(np.arange(10, dtype=float), (6, 1))  # plane sloping east
    with pytest.raises(StorageError, match="no depression found"):
        run(z, [[rect(0, 0, 10, 6)]], interval=0.5)
    # a maximum stage above the floor still computes storage behind the edge
    res = run(z, [[rect(0, 0, 10, 6)]], interval=0.5, max_stage=2.0)
    assert res.table[-1]["volume_m3"] > 0
    with pytest.raises(StorageError, match="no storage"):
        run(z, [[rect(0, 0, 10, 6)]], interval=0.5, max_stage=0.0)


def test_polygon_on_slope_fails_with_extend_hint():
    # pond floor 0, crest 10 at radius 4 cells; polygon cut at radius 2
    z = np.zeros((11, 11))
    for r in range(11):
        for c in range(11):
            z[r, c] = max(abs(r - 5), abs(c - 5)) * 2.5
    # tilt so the polygon edge holds the lowest cell
    z[5, 3] = -1.0
    with pytest.raises(StorageError, match="extend it beyond the crest"):
        run(z, [[rect(3, 3, 8, 8)]], interval=0.5)


def test_nodata_inside_polygon_is_wall_and_warned():
    z = box_pit()
    z[5, 5] = NODATA
    res = run(z, [[rect(1, 1, 10, 10)]], interval=1.0)
    assert any("NoData" in w for w in res.warnings)
    assert res.table[-1]["area_m2"] == pytest.approx(24 * CELL**2)


def test_polygon_beyond_dem_warns():
    z = box_pit()
    res = run(z, [[rect(-2, 1, 10, 10)]], interval=1.0)
    assert any("beyond the DEM" in w for w in res.warnings)


def test_no_valid_cells_raises():
    z = np.full((5, 5), NODATA)
    with pytest.raises(StorageError):
        run(z, [[rect(1, 1, 4, 4)]])


def test_bad_mode_raises():
    with pytest.raises(StorageError):
        run(box_pit(), [[rect(1, 1, 10, 10)]], mode="flat")


def test_partial_cells_weighted():
    z = box_pit()
    # polygon edge through the middle of the pit's first column
    x0 = GT[0] + 3.5 * CELL
    ring = [
        (x0, GT[3] - 10 * CELL),
        (GT[0] + 10 * CELL, GT[3] - 10 * CELL),
        (GT[0] + 10 * CELL, GT[3] - 1 * CELL),
        (x0, GT[3] - 1 * CELL),
        (x0, GT[3] - 10 * CELL),
    ]
    res = run(z, [[ring]], interval=1.0, max_stage=2.0)
    assert res.table[-1]["area_m2"] == pytest.approx(22.5 * CELL**2)


def test_volume_integrates_area():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:30, 0:30]
    z = 0.02 * ((xx - 15) ** 2 + (yy - 14) ** 2) + rng.uniform(0, 0.05, (30, 30))
    res = run(z, [[rect(1, 1, 29, 29)]], interval=0.02)
    h = np.array([r["stage_m"] for r in res.table])
    a = np.array([r["area_m2"] for r in res.table])
    v = np.array([r["volume_m3"] for r in res.table])
    assert np.all(np.diff(a) >= -1e-9) and np.all(np.diff(v) >= 0)
    trap = np.concatenate([[0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(h))])
    assert trap[-1] == pytest.approx(v[-1], rel=0.01)


def test_water_surface_polygons_match_table():
    shapely = pytest.importorskip("shapely")
    from shapely import wkt

    from mayim_tools.hydrology._common.grid_polygons import (
        rings_to_multipolygon_wkt,
    )

    z = box_pit()
    z[4:7, 4:7] = -2.0
    res = run(z, [[rect(1, 1, 10, 10)]], interval=0.5, polygon_interval=1.0)
    stages = [h for h, _, _ in res.ws_polygons]
    assert stages == [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for _, row, polys in res.ws_polygons:
        geom = wkt.loads(rings_to_multipolygon_wkt(polys))
        assert geom.is_valid
        assert geom.area == pytest.approx(row["area_m2"])
    assert shapely is not None


def test_depth_grid_and_mosaic():
    z = box_pit()
    z[0:3, 5] = 6.0
    res = run(z, [[rect(1, 1, 10, 10)]], interval=1.0)
    assert np.nanmax(res.depth) == pytest.approx(6.0)
    assert np.isfinite(res.depth).sum() == 25
    # a second, overlapping, deeper window shifted by (2, 3)
    other = run(z, [[rect(1, 1, 10, 10)]], interval=1.0, max_stage=8.0)
    other.window = (2, 3, *other.depth.shape)
    merged, origin = mosaic_depths([res, other], nodata=NODATA)
    assert origin == (0, 0)
    assert merged.shape == (13, 14)
    assert merged[5, 6] == pytest.approx(8.0)  # other's cell (3, 3) is pit
    assert merged[0, 0] == NODATA
    assert mosaic_depths([]) is None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_exports(tmp_path):
    z = box_pit()
    z[0:3, 5] = 6.0
    res = run(z, [[rect(1, 1, 10, 10)]], interval=2.0)
    csv_path = tmp_path / "stage.csv"
    write_stage_table_csv(str(csv_path), [res])
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["stage_m"] for r in rows] == ["0.0", "2.0", "4.0", "6.0"]
    assert rows[0]["storage_id"] == "S1"
    assert math.isclose(float(rows[-1]["volume_m3"]), 25 * CELL**2 * 6)

    hec_path = tmp_path / "hec.txt"
    write_hec_table(str(hec_path), [res])
    text = hec_path.read_text(encoding="utf-8").splitlines()
    assert text[2] == "Elevation (m)\tVolume (1000 m3)\tArea (1000 m2)"
    assert text[-1] == "6.000\t0.6000\t0.1000"


# ---------------------------------------------------------------------------
# Report data and Word report
# ---------------------------------------------------------------------------


def test_power_fit_recovers_exponent():
    from mayim_tools.hydrology.terrain_storage.core import power_fit

    d = np.linspace(0, 5, 26)
    fit = power_fit(d, 3.0 * d**2.5)
    assert fit["a"] == pytest.approx(3.0) and fit["b"] == pytest.approx(2.5)
    assert fit["r2"] == pytest.approx(1.0)
    assert power_fit([0, 1], [0, 1]) is None


def test_depth_distribution_shares():
    from mayim_tools.hydrology.terrain_storage.core import depth_distribution

    bins = depth_distribution([0.5, 1.5, 1.6, 3.9], [1, 1, 2, 4], n_bins=4)
    assert [b["area_m2"] for b in bins] == [1, 3, 0, 4]
    assert sum(b["pct"] for b in bins) == pytest.approx(100)
    assert depth_distribution([], []) == []


def test_result_detail_for_box_pit():
    z = box_pit()
    z[0:3, 5] = 6.0
    res = run(z, [[rect(1, 1, 10, 10)]], interval=0.5)
    # vertical-sided pit: V = 25 a d, exponent 1
    assert res.detail["fit_volume"]["b"] == pytest.approx(1.0)
    assert res.attributes["mean_dep"] == pytest.approx(6.0)
    assert res.attributes["wet_pct"] == pytest.approx(100 * 25 / 81, abs=0.01)
    assert res.detail["depth_hist"][-1]["pct"] == pytest.approx(100)
    assert res.window_gt == GT and res.polygons


def test_report_rows_thinning():
    from mayim_tools.hydrology.terrain_storage.report import report_rows

    table = [{"i": i} for i in range(101)]
    rows = report_rows(table, max_rows=30)
    assert len(rows) <= 30 and rows[0]["i"] == 0 and rows[-1]["i"] == 100
    assert report_rows(table[:5]) == table[:5]


def test_shape_description():
    from mayim_tools.hydrology.terrain_storage.report import shape_description

    assert "prismatic" in shape_description(1.1)
    assert "trough" in shape_description(2.0)
    assert "bowl" in shape_description(2.9)
    assert shape_description(None) == "not determined"


def test_word_report(tmp_path):
    docx = pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from mayim_tools.hydrology.terrain_storage.report import write_report_docx

    yy, xx = np.mgrid[0:30, 0:30]
    z = 0.02 * ((xx - 15) ** 2 + (yy - 14) ** 2)
    res = run(z, [[rect(1, 1, 29, 29)]], interval=0.1)
    z2 = box_pit()
    z2[0:3, 5] = 6.0
    res2 = run(z2, [[rect(1, 1, 10, 10)]], interval=1.0, max_stage=8.0)
    path = tmp_path / "storage.docx"
    write_report_docx(
        str(path),
        [res, res2],
        {
            "tool": "DEM depression stage-storage",
            "version": "test",
            "dem": {"path": "dem.tif", "crs": "EPSG:32735", "cell_x": 2, "cell_y": 2},
            "storage": {"layer": "ponds", "n_features": 2, "id_field": "name"},
            "interval": "0.1",
            "mode": "connected",
            "coverage": "Fractional cell coverage (8 x 8 sub-cells)",
            "outputs": [("Stage table", "table.csv")],
            "warnings": res2.warnings,
        },
    )
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "DEM Depression Stage-Storage Assessment" in text
    assert "depth-area curve" in text and "stage-storage curve" in text
    assert "vertical wall" in text
    assert len(doc.inline_shapes) == 14  # 7 figures x 2 storages
    assert len(doc.tables) >= 2 + 2 * 3


# ---------------------------------------------------------------------------
# Smoothed water-edge contours
# ---------------------------------------------------------------------------


def test_marching_squares_circle_and_closure():
    from mayim_tools.hydrology.terrain_storage.core import chaikin, marching_squares

    yy, xx = np.mgrid[0:60, 0:80]
    v = np.hypot(yy - 30, xx - 40)
    lines = marching_squares(v, 20.0)
    assert len(lines) == 1
    line = lines[0]
    assert np.allclose(line[0], line[-1])  # closed
    r = np.hypot(line[:, 0] - 30, line[:, 1] - 40)
    assert r.min() > 19.9 and r.max() <= 20.0 + 1e-9
    smooth = chaikin(line, 3)
    assert np.allclose(smooth[0], smooth[-1])
    assert len(smooth) == 8 * (len(line) - 1) + 1
    # a line running off the grid stays open
    open_lines = marching_squares(v, 35.0)
    assert open_lines and not any(np.allclose(ln[0], ln[-1]) for ln in open_lines)


def test_marching_squares_matches_contourpy_lengths():
    contourpy = pytest.importorskip("contourpy")
    from mayim_tools.hydrology.terrain_storage.core import marching_squares

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:40, 0:50]
    v = np.hypot(yy - 20.3, xx - 25.2) + rng.normal(0, 0.3, yy.shape)
    for level in (5.5, 12.0, 18.0):
        mine = marching_squares(v, level)
        ref = contourpy.contour_generator(z=v, line_type="Separate").lines(level)

        def total(ls):
            return sum(np.hypot(*np.diff(ln, axis=0).T).sum() for ln in ls)

        assert len(mine) == len(ref)
        assert total(mine) == pytest.approx(total(ref))


def test_chaikin_open_line_keeps_ends():
    from mayim_tools.hydrology.terrain_storage.core import chaikin

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    out = chaikin(pts, 2)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])
    assert np.array_equal(chaikin(pts, 0), pts)


def test_storage_contours_follow_water_edge():
    # paraboloid bowl: water edge at stage h is a circle of known radius
    yy, xx = np.mgrid[0:61, 0:61]
    rr = np.hypot(yy - 30, xx - 30)
    z = 0.01 * rr**2
    z[rr > 28] = 7.84 + (rr[rr > 28] - 28) * 0.0  # flat rim at 7.84 m
    res = run(z, [[rect(1, 1, 60, 60)]], interval=0.1, polygon_interval=1.0)
    by_stage = {h: lines for h, _, lines in res.contours}
    assert 4.0 in by_stage
    line = np.asarray(by_stage[4.0][0])
    cx, cy = GT[0] + 30.5 * CELL, GT[3] - 30.5 * CELL
    radius_cells = np.hypot(line[:, 0] - cx, line[:, 1] - cy) / CELL
    assert radius_cells.mean() == pytest.approx(20.0, abs=0.3)  # sqrt(4 / 0.01)
    assert np.allclose(line[0], line[-1])


def test_contours_skip_disconnected_hollow():
    z = np.full((9, 15), 10.0)
    z[2:7, 2:6] = 0.0
    z[2:7, 6] = 5.0
    z[2:7, 7:12] = 2.0
    res = run(z, [[rect(1, 1, 13, 8)]], interval=1.0, polygon_interval=1.0)
    lines = dict((h, ln) for h, _, ln in res.contours)
    assert len(lines[4.0]) == 1  # main pond only; hollow not yet joined
    xs = np.asarray(lines[4.0][0])[:, 0]
    assert xs.max() < GT[0] + 6.5 * CELL


def test_contour_interval_independent_of_polygons():
    yy, xx = np.mgrid[0:41, 0:41]
    z = 0.01 * np.hypot(yy - 20, xx - 20) ** 2
    poly = [[rect(1, 1, 40, 40)]]
    res = run(z, poly, interval=0.1, polygon_interval=1.0, contour_interval=0.25)
    ws_stages = [h for h, _, _ in res.ws_polygons]
    c_stages = [h for h, _, _ in res.contours]
    assert ws_stages[:3] == [1.0, 2.0, 3.0]
    assert c_stages[:4] == [0.25, 0.5, 0.75, 1.0]
    assert c_stages[-1] == res.top_stage
    # default follows the polygon spacing
    same = run(z, poly, interval=0.1, polygon_interval=1.0)
    assert [h for h, _, _ in same.contours] == ws_stages


def test_floor_point_is_lowest_cell():
    z = box_pit()
    z[6, 4] = -1.5
    res = run(z, [[rect(1, 1, 10, 10)]], interval=1.0)
    assert res.floor_z == -1.5
    assert res.floor_xy == pytest.approx((GT[0] + 4.5 * CELL, GT[3] - 6.5 * CELL))
    assert res.attributes["x_floor"] == pytest.approx(res.floor_xy[0])


def test_contour_levels_round_depths_then_spill_and_top():
    from mayim_tools.hydrology.terrain_storage.core import contour_levels

    lv = contour_levels(1559.94, 1561.3, 1561.3, 0.5)
    assert [k for _, k in lv] == ["depth", "depth", "spill"]
    assert [round(h - 1559.94, 6) for h, _ in lv] == [0.5, 1.0, 1.36]
    # specified maximum above the spill level: spill contour, then the top
    lv = contour_levels(100.0, 103.0, 101.2, 1.0)
    assert lv == [(101.0, "depth"), (101.2, "spill"), (102.0, "depth"), (103.0, "top")]
    # a depth level coinciding with the spill level is reported once, as spill
    lv = contour_levels(100.0, 101.0, 101.0, 0.5)
    assert lv == [(100.5, "depth"), (101.0, "spill")]


def test_storage_contours_at_depths_above_floor():
    yy, xx = np.mgrid[0:41, 0:41]
    z = 1559.94 + 0.01 * np.hypot(yy - 20, xx - 20) ** 2
    z[0:2, 20] = 1559.94 + 1.7  # notch: spill 1.7 m above the floor
    res = run(z, [[rect(0, 0, 41, 41)]], interval=0.1, contour_interval=0.5)
    depths = [row["depth_m"] for _, row, _ in res.contours]
    kinds = [row["kind"] for _, row, _ in res.contours]
    assert depths[:3] == [0.5, 1.0, 1.5]
    assert kinds[-1] == "spill" and res.contours[-1][0] == res.spill_z
