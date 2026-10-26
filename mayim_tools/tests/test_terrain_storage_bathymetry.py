"""
Tests for mayim_tools/hydrology/terrain_storage/bathymetry.py and
bathymetry_export.py (the bed estimate of DEM depression stage-storage).
Zero QGIS dependency - run under plain pytest.

Synthetic water bodies with a known bed are "flooded" to a water level
(DEM = max(bed, water level)), and the estimates are checked against the
true bed: a cone (shoreline slope and hypsometry both exact), a
paraboloid (hypsometry exact, slope 2 x too deep - by design an upper
bound) and a dammed V-shaped valley (valley method).
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from mayim_tools.hydrology.terrain_storage.bathymetry import (
    BathymetryError,
    WaterBody,
    area_rank,
    distance_to_false,
    estimate_bathymetry,
    fit_hypsometry,
    hypsometric_bed,
    mosaic_depths,
    sample_line,
    valley_bed,
)
from mayim_tools.hydrology.terrain_storage.bathymetry_export import (
    write_bathymetry_summary_csv,
    write_profile_csv,
)

NODATA = -9999.0


def _dict_rows(path):
    """Rows of a CSV that starts with a '#' column key (UTF-8 with BOM)."""
    raw = open(path, "rb").read()
    assert raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8-sig").splitlines()
    assert lines[0] == "# Column key"
    return list(csv.DictReader(ln for ln in lines if not ln.startswith("#")))


def grid(cell):
    return (1000.0, cell, 0.0, 5000.0, 0.0, -cell)


def rect(gt, c0, r0, c1, r1):
    x0, x1 = gt[0] + c0 * gt[1], gt[0] + c1 * gt[1]
    y0, y1 = gt[3] + r0 * gt[5], gt[3] + r1 * gt[5]
    return [(x0, y1), (x1, y1), (x1, y0), (x0, y0), (x0, y1)]


def run(z, gt, body=None, **kw):
    z = np.asarray(z, dtype=float)
    if body is None:
        body = WaterBody("A", [[rect(gt, 1, 1, z.shape[1] - 1, z.shape[0] - 1)]], 0)
    return estimate_bathymetry(z, NODATA, gt, (0, 0, *z.shape), body, **kw)


def bowl(kind, n=101, cell=2.0, wl=102.0):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n // 2, xx - n // 2) * cell
    if kind == "cone":
        true = 100.0 + 0.05 * r
    else:
        true = 100.0 + 0.0025 * r**2
    true[r > 0.9 * n] = true[r <= 0.9 * n].max()
    return np.maximum(true, wl), true, grid(cell)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_distance_to_false_exact_and_off_grid():
    m = np.ones((5, 7), dtype=bool)
    m[2, 3] = False
    d = distance_to_false(m, 2.0, 2.0, off_grid=True)
    assert d[2, 3] == 0
    assert d[2, 4] == pytest.approx(2.0)
    assert d[0, 0] == pytest.approx(np.hypot(4.0, 6.0))
    # off-grid as a target: the corner is one cell from the edge
    d2 = distance_to_false(m, 2.0, 2.0)
    assert d2[0, 0] == pytest.approx(2.0)


def test_distance_to_false_numpy_fallback_matches(monkeypatch):
    import builtins

    rng = np.random.default_rng(1)
    m = rng.uniform(size=(30, 40)) > 0.2
    ref = distance_to_false(m, 3.0, 2.0)
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    fallback = distance_to_false(m, 3.0, 2.0)
    np.testing.assert_allclose(fallback, ref)


def test_area_rank_ties_and_bounds():
    d = np.array([5.0, 3.0, 3.0, 1.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    r = area_rank(d, w)
    assert r.tolist() == [0.25, 0.75, 0.75, 1.0]


def test_fit_hypsometry_recovers_power_law():
    depth, n = 3.0, 1.7
    h = np.linspace(0.1, 4.0, 30)
    a = 1000.0 * ((h + depth) / depth) ** n
    fit = fit_hypsometry(h, a, 1000.0, 0.05, 50.0)
    assert fit["depth"] == pytest.approx(depth, rel=0.02)
    assert fit["n"] == pytest.approx(n, rel=0.02)
    assert fit["bound"] is None
    assert fit_hypsometry(h[:2], a[:2], 1000.0, 0.05, 50.0) is None


def test_hypsometric_bed_shape():
    rank = np.array([1.0, 0.25, 1e-12])
    bed = hypsometric_bed(10.0, 2.0, 2.0, rank)
    np.testing.assert_allclose(bed, [10.0, 9.0, 8.0], atol=1e-5)


def test_valley_bed_v_shape():
    t_xy = np.array([[0.0, 0.0]])
    t_z = np.array([5.0])
    cells = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    shore = np.array([10.0, 5.0, 0.0])
    bed = valley_bed(10.0, t_xy, t_z, cells, shore)
    np.testing.assert_allclose(bed, [5.0, 7.5, 10.0])


def test_sample_line():
    s = sample_line([(0, 0), (10, 0), (10, 5)], 2.5)
    assert len(s) == 7 and s[-1, 2] == pytest.approx(15.0)
    assert s[-1, 0] == 10 and s[-1, 1] == 5


# ---------------------------------------------------------------------------
# Whole-body estimates
# ---------------------------------------------------------------------------


def test_cone_both_methods_exact():
    z, true, gt = bowl("cone")
    res = run(z, gt, flat_tol=0.001)
    assert res.water_level == pytest.approx(102.0)
    assert res.methods["slope"]["max_depth"] == pytest.approx(2.0, rel=0.1)
    assert res.methods["hypsometry"]["max_depth"] == pytest.approx(2.0, rel=0.05)
    assert res.methods["hypsometry"]["n"] == pytest.approx(2.0, rel=0.05)
    assert res.basis == "slope+hypsometry"
    c = res.scenarios["central"]
    true_vol = np.clip(102.0 - true, 0, None)[res.water_mask].sum() * 4.0
    assert c["max_depth"] == pytest.approx(2.0, rel=0.05)
    assert c["volume_m3"] == pytest.approx(true_vol, rel=0.05)
    assert res.scenarios["low"]["max_depth"] == pytest.approx(0.5 * c["max_depth"])
    assert res.scenarios["high"]["max_depth"] >= c["max_depth"]


def test_paraboloid_slope_is_upper_bound_hypsometry_exact():
    z, true, gt = bowl("paraboloid")
    res = run(z, gt, flat_tol=0.001)
    d_slope = res.methods["slope"]["max_depth"]
    d_hyps = res.methods["hypsometry"]["max_depth"]
    assert d_hyps == pytest.approx(2.0, rel=0.05)
    assert d_slope > 1.8 * d_hyps  # linear extrapolation of a bowl: 2 x
    c = res.scenarios["central"]
    true_vol = np.clip(102.0 - true, 0, None)[res.water_mask].sum() * 4.0
    assert c["volume_m3"] == pytest.approx(true_vol, rel=0.05)
    # the bed is deepest farthest from shore and never above the water
    bed = res.beds["central"]
    assert np.nanmax(bed) <= 102.0 + 1e-9
    assert np.nanargmin(bed) == np.ravel_multi_index((50, 50), bed.shape)


def test_flat_tolerance_keeps_estimates_close():
    z, _, gt = bowl("cone")
    res = run(z, gt, flat_tol=0.1)
    assert res.scenarios["central"]["max_depth"] == pytest.approx(2.0, rel=0.15)


def test_known_max_depth_overrides():
    z, _, gt = bowl("cone")
    body = WaterBody("A", [[rect(gt, 1, 1, 100, 100)]], 0, max_depth=3.0)
    res = run(z, gt, body=body, flat_tol=0.001)
    assert res.basis == "user"
    assert res.scenarios["central"]["max_depth"] == pytest.approx(3.0, rel=1e-6)
    assert res.scenarios["low"]["max_depth"] == pytest.approx(2.4, rel=1e-6)
    assert res.scenarios["high"]["max_depth"] == pytest.approx(3.6, rel=1e-6)


def test_given_water_level():
    z, _, gt = bowl("cone")
    res = run(z, gt, flat_tol=0.001, water_level=102.0)
    assert res.water_level == 102.0
    with pytest.raises(BathymetryError):
        run(z, gt, water_level=90.0)


def test_dry_depression_fails():
    _, true, gt = bowl("cone")
    with pytest.raises(BathymetryError, match="no flat water surface"):
        run(true, gt, flat_tol=0.001)


def dammed_valley():
    cell = 5.0
    gt = grid(cell)
    rows, cols = 160, 80
    rr, cc = np.mgrid[0:rows, 0:cols]
    true = 118.0 - 0.02 * rr * cell + 0.15 * np.abs(cc * cell - 200.0)
    wall = 110
    z = true.copy()
    z[wall : wall + 4, :] = np.maximum(z[wall : wall + 4, :], 114.0)
    water = (rr < wall) & (true < 112.0)
    z[water] = 112.0

    def xy(r, c):
        return (gt[0] + (c + 0.5) * cell, gt[3] - (r + 0.5) * cell)

    poly = [[[xy(5, 5), xy(5, 75), xy(wall + 2, 75), xy(wall + 2, 5), xy(5, 5)]]]
    line = [xy(2, 40), xy(158, 40)]
    true_depth = 112.0 - (118.0 - 0.02 * (wall - 0.5) * cell)
    return z, true, gt, poly, line, true_depth


def test_dam_valley_method():
    z, true, gt, poly, line, true_depth = dammed_valley()
    body = WaterBody("Dam", poly, 0, centrelines=[line])
    res = run(z, gt, body=body, flat_tol=0.05)
    assert res.basis == "valley"
    v = res.methods["valley"]
    assert v["depth_upstream"] == pytest.approx(true_depth, rel=0.05)
    assert v["depth_downstream"] == pytest.approx(true_depth, rel=0.05)
    assert v["gradient_upstream"] == pytest.approx(0.02, rel=0.05)
    c = res.scenarios["central"]
    true_vol = np.clip(112.0 - true, 0, None)[res.water_mask].sum() * 25.0
    assert c["max_depth"] == pytest.approx(true_depth, rel=0.05)
    assert c["volume_m3"] == pytest.approx(true_vol, rel=0.08)
    assert res.scenarios["low"]["max_depth"] < c["max_depth"]
    assert res.scenarios["high"]["max_depth"] > c["max_depth"]
    assert res.profile and any(row["in_water"] for row in res.profile)


def test_dam_line_reversed_gives_same_result():
    z, _, gt, poly, line, true_depth = dammed_valley()
    body = WaterBody("Dam", poly, 0, centrelines=[line[::-1]])
    res = run(z, gt, body=body, flat_tol=0.05)
    assert res.methods["valley"]["max_depth"] == pytest.approx(true_depth, rel=0.05)


def test_centreline_missing_the_water_warns():
    z, _, gt, poly, _, _ = dammed_valley()
    far = [(gt[0] + 10, gt[3] - 10), (gt[0] + 20, gt[3] - 10)]
    body = WaterBody("Dam", poly, 0, centrelines=[far])
    res = run(z, gt, body=body, flat_tol=0.05)
    assert res.basis != "valley"
    assert any("centreline" in w for w in res.warnings)


def test_hydrolakes_check_warns_outside_range():
    z, _, gt = bowl("cone")
    res = run(z, gt, flat_tol=0.001, hydrolakes={"depth_avg_m": 25.0})
    assert any("HydroLAKES" in w for w in res.warnings)
    res = run(
        z,
        gt,
        flat_tol=0.001,
        hydrolakes={"depth_avg_m": None, "volume_m3": 1.0},
    )
    assert not any("HydroLAKES" in w for w in res.warnings)


def test_mosaic_and_exports(tmp_path):
    z, _, gt, poly, line, _ = dammed_valley()
    body = WaterBody("Dam", poly, 0, centrelines=[line])
    res = run(z, gt, body=body, flat_tol=0.05)
    merged, origin = mosaic_depths([res])
    assert origin == (0, 0) and merged.shape == z.shape
    assert merged.max() == pytest.approx(
        res.scenarios["central"]["max_depth"], rel=1e-5
    )

    path = tmp_path / "summary.csv"
    write_bathymetry_summary_csv(str(path), [res])
    rows = _dict_rows(path)
    assert [r["name"] for r in rows[:3]] == ["low", "central", "high"]
    assert "," not in rows[1]["volume_m3"]  # no thousands separators

    path = tmp_path / "profile.csv"
    write_profile_csv(str(path), [res])
    prof = _dict_rows(path)
    assert prof and prof[0]["storage_id"] == "Dam"


# ---------------------------------------------------------------------------
# Integration with DEM depression stage-storage
# ---------------------------------------------------------------------------


def _storage_area(gt, n):
    from mayim_tools.hydrology.terrain_storage.core import StorageArea

    return StorageArea("A", [[rect(gt, 1, 1, n - 1, n - 1)]], 0)


def test_stage_storage_with_estimated_bed():
    from mayim_tools.hydrology.terrain_storage.core import compute_storage_with_bed

    z, true, gt = bowl("cone")
    # a rim so the depression spills above the water: true cone up to 104.5 m
    res = compute_storage_with_bed(
        z, NODATA, gt, (0, 0, *z.shape), _storage_area(gt, 101), 0.1, flat_tol=0.001
    )
    assert res.bathymetry is not None
    assert res.floor_z == pytest.approx(100.0, abs=0.15)  # estimated bed
    by = {row["stage_m"]: row for row in res.table}
    assert 102.0 in by
    # volume at the survey water level = central volume below the water
    assert by[102.0]["volume_m3"] == pytest.approx(
        res.bathymetry.scenarios["central"]["volume_m3"], rel=0.02
    )
    # low <= central <= high at every stage, all equal well above the water
    for row in res.table:
        assert row["volume_low_m3"] <= row["volume_m3"] + 1e-6
        assert row["volume_m3"] <= row["volume_high_m3"] + 1e-6
    top = res.table[-1]
    spread = top["volume_high_m3"] - top["volume_low_m3"]
    below = (
        res.bathymetry.scenarios["high"]["volume_m3"]
        - res.bathymetry.scenarios["low"]["volume_m3"]
    )
    assert spread == pytest.approx(below, rel=0.05)
    assert res.attributes["z_survey"] == pytest.approx(102.0)
    assert res.attributes["vws_c"] > 0


def test_stage_storage_with_bed_dam_and_csv(tmp_path):
    from mayim_tools.hydrology.terrain_storage.core import (
        StorageArea,
        compute_storage_with_bed,
    )
    from mayim_tools.hydrology.terrain_storage.export import write_stage_table_csv

    z, true, gt, poly, line, true_depth = dammed_valley()
    area = StorageArea("Dam", poly, 0)
    res = compute_storage_with_bed(
        z, NODATA, gt, (0, 0, *z.shape), area, 0.5, centrelines=[line], flat_tol=0.05
    )
    assert res.bathymetry.basis == "valley"
    assert 112.0 - res.floor_z == pytest.approx(true_depth, rel=0.06)
    path = tmp_path / "table.csv"
    write_stage_table_csv(str(path), [res])
    rows = _dict_rows(path)
    assert "volume_low_m3" in rows[0] and "volume_high_m3" in rows[0]


def test_stage_storage_csv_without_bed_has_no_range_columns(tmp_path):
    from mayim_tools.hydrology.terrain_storage.core import compute_storage
    from mayim_tools.hydrology.terrain_storage.export import write_stage_table_csv

    _, true, gt = bowl("cone")
    res = compute_storage(
        true, NODATA, gt, (0, 0, *true.shape), _storage_area(gt, 101), 0.5
    )
    path = tmp_path / "table.csv"
    write_stage_table_csv(str(path), [res])
    header = [
        ln
        for ln in path.read_text(encoding="utf-8-sig").splitlines()
        if not ln.startswith("#")
    ][0]
    assert "volume_low_m3" not in header


def test_report_with_bed_estimate(tmp_path):
    docx = pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from mayim_tools.hydrology.terrain_storage.core import (
        StorageArea,
        compute_storage_with_bed,
    )
    from mayim_tools.hydrology.terrain_storage.report import write_report_docx

    z, true, gt, poly, line, _ = dammed_valley()
    res = compute_storage_with_bed(
        z,
        NODATA,
        gt,
        (0, 0, *z.shape),
        StorageArea("Dam", poly, 0),
        0.5,
        centrelines=[line],
        flat_tol=0.05,
    )
    path = tmp_path / "report.docx"
    write_report_docx(
        str(path),
        [res],
        {
            "tool": "DEM depression stage-storage",
            "version": "test",
            "dem": {"path": "dem.tif", "crs": "EPSG:32735", "cell_x": 5, "cell_y": 5},
            "storage": {"layer": "dams", "n_features": 1, "id_field": "name"},
            "interval": "0.5",
            "mode": "connected",
            "coverage": "Fractional cell coverage (8 x 8 sub-cells)",
            "outputs": [],
            "warnings": res.warnings,
        },
    )
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Estimated bed below the surveyed water surface" in text
    assert "Hollister" in text and "Heathcote" in text
    assert "along the stream centreline" in text
    assert len(doc.inline_shapes) == 8  # 7 standard figures + the bed profile
