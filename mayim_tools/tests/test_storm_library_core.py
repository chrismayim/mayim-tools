"""
Tests for mayim_tools/rainfall/storm_library/core.py and the shared
mayim_tools/rainfall/_common/ddf.read_ddf_csv reader. Zero QGIS
dependency - run under plain pytest.

The two central tests use synthetic series with a known answer:
- identity: the reference DDF is fitted from the series itself, so the
  product's storm structure IS the reference structure -> F ~ 1 at all
  durations and a "consistent" verdict;
- smoothing: the same series after a 3-hour moving average (mass
  preserving, peaks flattened) against the same reference -> F < 1 at
  short durations, ~1 at the anchor, and "sharpening indicated".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mayim_tools.rainfall._common.ddf import DDFTable, read_ddf_csv
from mayim_tools.rainfall._common.rfa.ams import (
    compute_year_completeness,
    extract_ams_for_duration,
)
from mayim_tools.rainfall.storm_library import export
from mayim_tools.rainfall.storm_library.core import (
    DECISION_CONSISTENT,
    DECISION_FLATTER,
    INDICATIVE_STAMP,
    AnchorMapping,
    FittedCurve,
    StormLibraryConfig,
    StormLibraryEngine,
    ams_by_year,
    embedded_burst_check,
    prepare_series,
    weiss_factor,
)

DURATIONS = [60, 120, 180, 360, 720, 1440, 2880]
ARIS = [2, 5, 10, 20, 50, 100]


# ----------------------------------------------------------------------
# Synthetic data helpers
# ----------------------------------------------------------------------


def synthetic_series(years=30, seed=1):
    """Hourly series of peaked storms (Poisson arrivals, gamma depths,
    random peak position and peakedness)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        "1980-01-01", f"{1980 + years}-01-01", freq="h", inclusive="left"
    )
    v = np.zeros(len(idx))
    t = 0
    while True:
        t += int(rng.exponential(24 * 12))
        if t >= len(v) - 48:
            break
        dur = int(rng.integers(2, 30))
        depth = rng.gamma(0.8, 15)
        x = (np.arange(dur) + 0.5) / dur
        w = np.exp(-np.abs(x - rng.uniform(0.1, 0.9)) * rng.uniform(3, 12))
        v[t : t + dur] += depth * w / w.sum()
    return pd.Series(v, index=idx)


def ddf_from_series(series, durations=DURATIONS, aris=ARIS):
    """Reference DDF fitted (GEV L-moments, Weiss-corrected sliding AMS)
    from the series itself - the 'identity' reference."""
    years = series.index.year.values
    valid = np.unique(years)
    rows = []
    for d in durations:
        n = int(d / 60)
        _, mx = ams_by_year(series.values, years, valid, n)
        c = FittedCurve(mx * weiss_factor(n), "GEV", "L-moments")
        for t in aris:
            rows.append(
                {
                    "duration_min": d,
                    "ari_years": t,
                    "aep_percent": 100.0 / t,
                    "depth_mm": c.depth_at_return_period(t),
                }
            )
    return DDFTable(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def series():
    return synthetic_series()


@pytest.fixture(scope="module")
def reference(series):
    return ddf_from_series(series)


@pytest.fixture(scope="module")
def identity_result(series, reference):
    cfg = StormLibraryConfig(n_bootstrap=60, random_seed=7)
    return StormLibraryEngine(cfg).run(series, 60.0, reference)


@pytest.fixture(scope="module")
def smoothed_result(series, reference):
    smoothed = series.rolling(3, center=True, min_periods=1).mean()
    cfg = StormLibraryConfig(n_bootstrap=60, random_seed=7)
    return StormLibraryEngine(cfg).run(smoothed, 60.0, reference)


def _rows(result, duration):
    return [r for r in result.consistency_rows if r["duration_min"] == duration]


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def test_weiss_factor_values():
    assert weiss_factor(1) == pytest.approx(1 / (1 - 1 / 8))  # 1.1429
    assert weiss_factor(2) == pytest.approx(1.0667, abs=1e-4)
    assert weiss_factor(24) == pytest.approx(1.0052, abs=1e-4)
    with pytest.raises(ValueError):
        weiss_factor(0)


def test_ams_by_year_matches_shared_rfa_extraction():
    s = synthetic_series(years=6, seed=3)
    s.iloc[500:540] = np.nan  # a gap: windows touching it are invalid
    recs = compute_year_completeness(s, 60.0, 0.5)
    valid = [r.year for r in recs if r.included]
    for n in (1, 3, 24):
        ds = extract_ams_for_duration(s, n * 60, 60.0, recs)
        yy, mx = ams_by_year(s.values, s.index.year.values, valid, n)
        assert list(yy) == list(ds.years)
        assert np.allclose(mx, ds.values_mm)


def test_anchor_mapping_range_handling():
    rng = np.random.default_rng(0)
    data = rng.gumbel(50, 15, 60)
    curve = FittedCurve(data, "GEV", "L-moments")
    ref = DDFTable(
        pd.DataFrame(
            [
                {
                    "duration_min": 1440,
                    "ari_years": t,
                    "aep_percent": 100 / t,
                    "depth_mm": 80 + 25 * np.log(t),
                }
                for t in ARIS
            ]
        )
    )
    m = AnchorMapping(curve, ref, 1440)
    small, mid, huge = 5.0, curve.depth_at_return_period(10), 10 * data.max()
    s, t, flags = m.scale([small, mid, huge])
    assert flags[0] == "below_range" and s[0] == pytest.approx(m.s_floor)
    assert flags[1] == "in_range"
    assert mid * s[1] == pytest.approx(ref.depth_at_ari(1440, 10)[0], rel=1e-3)
    assert flags[2] == "extrap_aep"


def test_embedded_burst_check_flags_spiky_pattern():
    ref = DDFTable(
        pd.DataFrame(
            [
                {
                    "duration_min": d,
                    "ari_years": t,
                    "aep_percent": 100 / t,
                    "depth_mm": base * (1 + 0.3 * np.log(t)),
                }
                for d, base in ((60, 20.0), (1440, 60.0))
                for t in ARIS
            ]
        )
    )
    burst_t = 10.0
    total = ref.depth_at_ari(1440, burst_t)[0]
    uniform = np.full(24, total / 24)
    spiky = np.zeros(24)
    spiky[5] = total * 0.9
    spiky[6:] = total * 0.1 / 18
    no_w = lambda n: 1.0  # noqa: E731
    assert embedded_burst_check(uniform, 60, 1440, burst_t, ref, [60], no_w) == (
        False,
        "",
    )
    flag, detail = embedded_burst_check(spiky, 60, 1440, burst_t, ref, [60], no_w)
    assert flag and "1 h burst" in detail


# ----------------------------------------------------------------------
# End-to-end: identity and smoothing
# ----------------------------------------------------------------------


def test_identity_gives_flatness_near_one(identity_result):
    for r in identity_result.consistency_rows:
        assert r["F"] == pytest.approx(1.0, abs=0.05), r
    assert identity_result.self_check_passed
    decisions = {r["decision"] for r in identity_result.duration_summary}
    assert DECISION_FLATTER not in decisions
    assert identity_result.overall_verdict.startswith("consistent")


def test_smoothing_is_detected_as_flatter(smoothed_result):
    one_hour = _rows(smoothed_result, 60)
    assert all(r["F_p50"] < 0.7 for r in one_hour)
    assert all(r["decision"] == DECISION_FLATTER for r in one_hour)
    for r in _rows(smoothed_result, 1440):  # anchor still ~1
        assert r["F"] == pytest.approx(1.0, abs=0.03)
    assert smoothed_result.overall_verdict.startswith("sharpening indicated")
    summary = {
        r["duration_min"]: r["decision"] for r in smoothed_result.duration_summary
    }
    assert summary[60] == DECISION_FLATTER
    assert summary[720] == DECISION_CONSISTENT


def test_flatness_increases_towards_anchor(smoothed_result):
    f = [
        np.mean([r["F_p50"] for r in _rows(smoothed_result, d)]) for d in (60, 120, 360)
    ]
    assert f[0] < f[1] < f[2]


def test_anchor_self_check_rows(identity_result):
    rows = _rows(identity_result, 1440)
    assert rows and all(r["is_anchor"] for r in rows)
    assert all(r["decision"] == "anchor (self-check)" for r in rows)
    assert identity_result.metadata["self_check_max_deviation"] <= 0.03


def test_bootstrap_is_reproducible(series, reference):
    cfg = StormLibraryConfig(n_bootstrap=15, random_seed=11, build_library=False)
    a = StormLibraryEngine(cfg).run(series, 60.0, reference)
    b = StormLibraryEngine(cfg).run(series, 60.0, reference)
    fa = [r["F_p50"] for r in a.consistency_rows]
    fb = [r["F_p50"] for r in b.consistency_rows]
    assert np.allclose(fa, fb, equal_nan=True)


# ----------------------------------------------------------------------
# Storm library
# ----------------------------------------------------------------------


def test_library_patterns_are_well_formed(identity_result):
    lib = identity_result.library_rows
    assert len(lib) > 50
    for row in lib[:500]:
        incs = [v for k, v in row.items() if k.startswith("inc_")]
        assert len(incs) == row["n_increments"]
        assert row["n_increments"] == row["duration_min"] / 60
        assert row["n_increments"] >= 3
        assert sum(incs) == pytest.approx(100.0, abs=1e-6)
        assert row["shape"] in ("Early", "Middle", "Late")
        assert row["aep_band"] in ("frequent", "intermediate", "rare")
        assert 1 <= row["month"] <= 12


def test_library_durations_respect_native_step(identity_result):
    durations = {r["duration_min"] for r in identity_result.library_rows}
    assert 180 in durations and 1440 in durations
    assert 270 not in durations  # 4.5 h is not a whole number of hours
    assert all(d >= 180 for d in durations)  # >= 3 native steps


# ----------------------------------------------------------------------
# Missing data, tiers, config validation
# ----------------------------------------------------------------------


def test_missing_data_never_zero_and_years_excluded(reference):
    s = synthetic_series(years=12, seed=5)
    s.loc["1985-02-01":"1985-06-30"] = np.nan  # ~40% of 1985 missing
    cfg = StormLibraryConfig(n_bootstrap=10)
    res = StormLibraryEngine(cfg).run(s, 60.0, reference)
    assert "1985" in res.metadata["years_excluded"]
    assert res.metadata["n_years_used"] == 11
    gap = s.loc["1985-02-01":"1985-06-30"].index
    for row in res.library_rows:
        assert not (row["start"] <= gap[-1] and row["end"] >= gap[0])


def test_tier3_stamps_every_output(series, reference):
    cfg = StormLibraryConfig(n_bootstrap=5, reference_tier=3)
    res = StormLibraryEngine(cfg).run(series, 60.0, reference)
    assert res.status_stamp == INDICATIVE_STAMP
    assert res.metadata["status"] == INDICATIVE_STAMP
    for rows in (res.consistency_rows, res.library_rows, res.event_rows):
        assert rows and all(r["status"] == INDICATIVE_STAMP for r in rows)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        StormLibraryEngine(StormLibraryConfig(distribution="Weibull"))
    with pytest.raises(ValueError):
        StormLibraryEngine(StormLibraryConfig(reference_tier=4))


def test_anchor_must_be_multiple_of_native_step(series, reference):
    cfg = StormLibraryConfig(anchor_min=1450, n_bootstrap=0)
    with pytest.raises(ValueError, match="multiple"):
        StormLibraryEngine(cfg).run(series, 60.0, reference)


# ----------------------------------------------------------------------
# Series preparation
# ----------------------------------------------------------------------


def test_prepare_series_extractor_format_multisite_and_tz():
    t = pd.date_range("2000-01-01 00:00", periods=48, freq="h")
    df = pd.concat(
        [
            pd.DataFrame({"Site": "A", "ValidTime": t, "PrecipitationMM": 1.0}),
            pd.DataFrame({"Site": "B", "ValidTime": t, "PrecipitationMM": 2.0}),
        ]
    )
    df.loc[3, "PrecipitationMM"] = np.nan
    grid, native, site, warns = prepare_series(df, timezone_offset_h=2)
    assert site == "A" and any("2 sites" in w for w in warns)
    assert native == 60.0
    assert grid.index[0] == pd.Timestamp("2000-01-01 02:00")
    assert np.isnan(grid.iloc[3])  # missing stays missing
    grid_b, _, site_b, _ = prepare_series(df, site="B")
    assert site_b == "B" and grid_b.max() == 2.0


# ----------------------------------------------------------------------
# Shared reference DDF reader
# ----------------------------------------------------------------------

DRESA_SINGLE = """Latitude,Longitude,MAP (mm),Altitude (m),Cluster
-29.6,30.3,850,1000,5

Duration,2yr Depth (mm),2yr Lower (mm),2yr Upper (mm),10yr Depth (mm),10yr Lower (mm),10yr Upper (mm)
1 h,30,25,35,45,38,52
24 h,70,60,80,110,95,125
1 day,60,52,68,95,82,108
2 day,85,75,95,130,115,145

Nearest Rainfall Stations (reference only - not used in the grid-based calculation)
Station,Distance
X,1.2
"""

DRESA_MULTI = """Site,Latitude,Longitude,MAP (mm),Altitude (m),Cluster,Duration,2yr Depth (mm),2yr Lower (mm),2yr Upper (mm)
P1,-29,30,800,900,5,1 h,30,25,35
P1,-29,30,800,900,5,24 h,70,60,80
P2,-28,31,900,950,6,1 h,40,35,45
P2,-28,31,900,950,6,24 h,90,80,100
"""

RECOMMENDED = """Duration,RecommendedDistribution,2yr Depth (mm),5yr Depth (mm)
1 h,GEV,20,28
24 h,GLO,60,80
"""


def test_read_ddf_csv_design_rainfall_single_site(tmp_path):
    p = tmp_path / "dresa.csv"
    p.write_text(DRESA_SINGLE)
    long_df, info = read_ddf_csv(p)
    one_day = long_df[long_df["duration_min"] == 1440]
    # continuous 24 h kept, fixed 1 day dropped
    assert set(one_day["duration_label"]) == {"24 h"}
    assert "1 day" in info["dropped_durations"]
    # central estimates only, bounds carried separately
    row = long_df[(long_df["duration_min"] == 60) & (long_df["ari_years"] == 2)].iloc[0]
    assert (row.depth_mm, row.lower_mm, row.upper_mm) == (30, 25, 35)
    assert len(long_df) == 6  # 3 durations x 2 return periods
    # 2 day is fixed-interval -> converted with Weiss n=2
    two_day = long_df[long_df["duration_min"] == 2880].iloc[0]
    assert two_day.depth_mm == pytest.approx(85 * weiss_factor(2))
    assert info["fixed_day_durations"] and info["has_bounds"]
    assert row.aep_percent == pytest.approx(50.0)  # AMS convention 100/T


def test_read_ddf_csv_multisite_and_recommended(tmp_path):
    p = tmp_path / "multi.csv"
    p.write_text(DRESA_MULTI)
    long_df, info = read_ddf_csv(p, site="P2")
    assert info["site"] == "P2" and set(info["sites_available"]) == {"P1", "P2"}
    assert long_df[long_df["duration_min"] == 60]["depth_mm"].iloc[0] == 40
    with pytest.raises(ValueError):
        read_ddf_csv(p, site="nope")

    q = tmp_path / "rec.csv"
    q.write_text(RECOMMENDED)
    long_df, info = read_ddf_csv(q)
    assert sorted(long_df["ari_years"].unique()) == [2, 5]
    assert not info["has_bounds"]
    table = DDFTable(long_df)
    assert table.depth_at_ari(1440, 5)[0] == pytest.approx(80)
    t, ok = table.ari_of_depth(1440, 70)
    assert ok and 2 < t < 5


def test_exports_write_blank_for_missing(tmp_path, identity_result):
    export.write_consistency_csv(identity_result, tmp_path / "c.csv")
    export.write_library_csv(identity_result, tmp_path / "l.csv")
    export.write_events_csv(identity_result, tmp_path / "e.csv")
    export.write_metadata_csv(identity_result, tmp_path / "m.csv")
    c = pd.read_csv(tmp_path / "c.csv")
    assert {"F", "F_p05", "F_p50", "F_p95", "decision", "status"} <= set(c.columns)
    lib = pd.read_csv(tmp_path / "l.csv")
    assert len(lib) == len(identity_result.library_rows)
    m = pd.read_csv(tmp_path / "m.csv")
    assert "overall_verdict" in set(m["key"])
    # NaN reference bounds are written as empty cells, never zero
    assert c["reference_lower_mm"].isna().all()
    pytest.importorskip("matplotlib")
    assert export.write_flatness_png(identity_result, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 1000
