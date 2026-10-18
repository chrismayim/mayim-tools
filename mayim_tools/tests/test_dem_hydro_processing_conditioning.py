"""Tests for conditioning, keep-sink matching, CSV export and the Word
report of the DEM Hydrological Processing tool (no QGIS needed)."""

import csv

import numpy as np
import pytest

from mayim_tools.hydrology.dem_hydro_processing.core.conditioning import (
    MODE_CONTINUOUS,
    MODE_PRESERVE,
    ConditioningSettings,
    run_conditioning,
)
from mayim_tools.hydrology.dem_hydro_processing.core.diagnostics import (
    find_unresolvable_cells,
)
from mayim_tools.hydrology.dem_hydro_processing.core.keep_sinks import (
    STATUS_MATCHED,
    STATUS_OUTSIDE,
    STATUS_SINK_ONLY,
    KeepFeature,
)
from mayim_tools.hydrology.dem_hydro_processing.core.stream_evidence import (
    RasterTransform,
)
from mayim_tools.hydrology.dem_hydro_processing.export import (
    fmt,
    write_sectioned_csv,
)

NODATA = -9999.0
N = 80


def make_dem(seed=1, noise=0.3, pan=True):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N]
    dem = 100 - 0.05 * x - 0.03 * y + rng.normal(0, noise, (N, N))
    if pan:  # a deep, smooth pan near the centre
        dem -= 6.0 * np.exp(-(((x - 40) ** 2 + (y - 40) ** 2) / 30.0))
    return dem.astype("float64")


TR = RasterTransform(0.0, float(N), 1.0, 1.0)


def run(dem=None, **kw):
    dem = make_dem() if dem is None else dem
    settings = kw.pop("settings", ConditioningSettings())
    return run_conditioning(dem, NODATA, TR, (1.0, 1.0), settings=settings, **kw)


def test_continuous_mode_leaves_no_unresolved_sinks():
    r = run()
    assert r.stats["mode"] == MODE_CONTINUOUS
    assert r.stats["unresolved_sink_clusters"] == 0
    assert r.sinks == []
    assert find_unresolvable_cells(r.corrected, NODATA).sum() == 0
    assert r.stats["cells_changed"] > 0


def test_nodata_cells_unchanged():
    dem = make_dem()
    dem[:5, :5] = NODATA
    r = run(dem)
    assert np.all(r.corrected[:5, :5] == NODATA)


def test_polygon_keeps_pan_and_reports_match():
    poly = KeepFeature(
        "pan",
        "polygon",
        rings=[[(30, 50), (50, 50), (50, 30), (30, 30), (30, 50)]],
    )
    r = run(keep_features=[poly])
    assert r.keep_matches[0].status == STATUS_MATCHED
    assert r.keep_matches[0].leaf_ids
    assert r.stats["kept_depressions"] >= 1
    # the pan bottom stays a sink and is reported as confirmed, not unresolved
    assert any(k.kind == "kept" for k in r.sinks)
    assert r.stats["unresolved_sink_clusters"] == 0
    assert r.corrected[40, 40] == pytest.approx(make_dem()[40, 40], abs=1e-6)


def test_point_snaps_to_depression_within_radius():
    pt = KeepFeature("pan point", "point", xy=(43.5, N - 40.5))
    r = run(keep_features=[pt], settings=ConditioningSettings(keep_search_cells=4))
    assert r.keep_matches[0].status == STATUS_MATCHED


def test_point_on_slope_becomes_sink_only():
    dem = make_dem(pan=False, noise=0.0)
    pt = KeepFeature("slope", "point", xy=(10.5, N - 70.5))
    r = run(dem, keep_features=[pt])
    assert r.keep_matches[0].status == STATUS_SINK_ONLY


def test_feature_outside_dem_reported():
    pt = KeepFeature("far", "point", xy=(5000.0, 5000.0))
    r = run(keep_features=[pt])
    assert r.keep_matches[0].status == STATUS_OUTSIDE


def test_every_feature_gets_a_match():
    feats = [
        KeepFeature("a", "point", xy=(5000.0, 5000.0)),
        KeepFeature("b", "point", xy=(43.5, N - 40.5)),
    ]
    r = run(keep_features=feats)
    assert [m.label for m in r.keep_matches] == ["a", "b"]


def test_preserve_mode_runs_and_classifies():
    s = ConditioningSettings(mode=MODE_PRESERVE, n_iterations=20)
    r = run(settings=s)
    assert r.classifications is not None
    assert r.stats["classified"]
    assert r.stats["mode"] == MODE_PRESERVE


def test_fmt_has_no_thousands_separator():
    assert fmt(1234567) == "1234567"
    assert fmt(1234567.5) == "1.23457e+06" or "," not in fmt(1234567.5)
    assert "," not in fmt(123456.789)
    assert fmt(None) == ""
    assert fmt(float("nan")) == ""


def test_sectioned_csv_sections_and_encoding(tmp_path):
    poly = KeepFeature(
        "pan", "polygon", rings=[[(30, 50), (50, 50), (50, 30), (30, 30), (30, 50)]]
    )
    r = run(keep_features=[poly])
    path = tmp_path / "r.csv"
    write_sectioned_csv(str(path), r, {"tool": "T", "version": "1"})
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    titles = [
        row[0] for row in rows if row and row[0][:2] in {f"{i}." for i in range(1, 9)}
    ]
    assert len(titles) == 8
    text = path.read_text(encoding="utf-8-sig")
    assert "pan" in text
    assert text.count("DEPRESSIONS RESOLVED") == 1


def test_word_report_written(tmp_path):
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from mayim_tools.hydrology.dem_hydro_processing.report import write_report_docx

    r = run()
    path = tmp_path / "r.docx"
    write_report_docx(str(path), r, {"tool": "T", "version": "1"})
    assert path.stat().st_size > 10000


def test_breach_search_fails_fast_when_pit_is_lowest_point():
    from mayim_tools.hydrology.dem_hydro_processing.core.breaching import (
        least_cost_breach_from_pit,
    )

    dem = np.full((400, 400), 10.0)
    dem[200, 200] = 1.0  # the lowest cell anywhere: nothing to escape to
    res = least_cost_breach_from_pit(dem, 200, 200, nodata=NODATA)
    assert not res.success
    assert res.escape_cell is None


def test_global_minimum_basin_is_resolved_quickly():
    import time

    y, x = np.mgrid[0:300, 0:300].astype("float64")
    dem = 100 - 0.01 * x - 25 * np.exp(-(((x - 150) ** 2 + (y - 150) ** 2) / 900.0))
    t0 = time.time()
    r = run_conditioning(dem, NODATA, TR, (1.0, 1.0), settings=ConditioningSettings())
    assert r.stats["unresolved_sink_clusters"] == 0
    assert time.time() - t0 < 60
