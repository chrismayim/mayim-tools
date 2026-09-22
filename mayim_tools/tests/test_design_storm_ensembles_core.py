"""
Tests for mayim_tools/rainfall/design_storm_ensembles/core.py. Zero
QGIS dependency - run under plain pytest.

Covers the individual building blocks (AEP/ARI conversion, duration
parsing, shape classification, window dedupe, DDF lookup) plus a full
end-to-end run of DesignStormEngine against synthetic data, mirroring
the source plugin's own dev/test_synthetic.py sanity checks (ensemble
row count, increment sums ~100, increment counts matching
TARGET_DURATIONS, expected bin-status categories present) as real
pytest assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mayim_tools.rainfall.design_storm_ensembles.core import (
    TARGET_DURATIONS,
    DDFTable,
    DesignStormEngine,
    aep_band,
    aep_from_ari,
    ari_from_aep,
    assign_aep_empirical,
    classify_shape,
    dedupe_windows,
    load_ddf_table,
    load_timeseries,
    parse_duration_to_minutes,
)

# ----------------------------------------------------------------------
# AEP <-> ARI conversion
# ----------------------------------------------------------------------


def test_aep_from_ari_known_values():
    # AEP = 1 - e^(-1/ARI); ARI=1 -> ~63.2%, ARI=100 -> ~0.995%
    assert abs(aep_from_ari(1.0) - 63.212) < 0.01
    assert abs(aep_from_ari(100.0) - 0.995) < 0.01


def test_ari_from_aep_is_inverse_of_aep_from_ari():
    for ari in [1.5, 2.0, 5.0, 10.0, 50.0, 100.0]:
        aep = aep_from_ari(ari)
        assert abs(ari_from_aep(aep) - ari) < 1e-6


def test_aep_band_thresholds():
    assert aep_band(50.0) == "frequent"
    assert aep_band(20.0) == "frequent"
    assert aep_band(19.9) == "intermediate"
    assert aep_band(5.0) == "intermediate"
    assert aep_band(4.9) == "rare"
    assert aep_band(0.5) == "rare"


def test_assign_aep_empirical_weibull_position():
    population = [10.0, 20.0, 30.0, 40.0]
    # this_depth=15.0 is exceeded by 3 of 4 (20, 30, 40) -> rank=3+1=4, n=4
    # -> 100*4/5=80
    assert assign_aep_empirical(population, 15.0) == 80.0
    # this_depth=5.0 is exceeded by all 4 -> rank=4+1=5, n=4 -> 100*5/5=100
    assert assign_aep_empirical(population, 5.0) == 100.0
    # highest value: exceeded by 0 -> rank=0+1=1 -> 100*1/5=20
    assert assign_aep_empirical(population, 40.0) == 20.0


# ----------------------------------------------------------------------
# Duration parsing
# ----------------------------------------------------------------------


def test_parse_duration_to_minutes_hours():
    assert parse_duration_to_minutes("3 h") == 180
    assert parse_duration_to_minutes("3hr") == 180


def test_parse_duration_to_minutes_days():
    assert parse_duration_to_minutes("1 day") == 1440


def test_parse_duration_to_minutes_bare_number():
    assert parse_duration_to_minutes("180") == 180.0


def test_parse_duration_to_minutes_invalid_raises():
    with pytest.raises(ValueError):
        parse_duration_to_minutes("not a duration")


# ----------------------------------------------------------------------
# Shape classification
# ----------------------------------------------------------------------


def test_classify_shape_early():
    window = {"values": np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0])}
    assert classify_shape(window) == "Early"


def test_classify_shape_middle():
    window = {"values": np.array([1.0, 1.0, 10.0, 1.0, 1.0, 1.0])}
    assert classify_shape(window) == "Middle"


def test_classify_shape_late():
    window = {"values": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])}
    assert classify_shape(window) == "Late"


def test_classify_shape_empty_returns_none():
    assert classify_shape({"values": np.array([])}) is None


# ----------------------------------------------------------------------
# Window dedupe
# ----------------------------------------------------------------------


def test_dedupe_windows_collapses_overlap_keeping_highest_depth():
    windows = [
        {
            "start": pd.Timestamp("2020-01-01 00:00"),
            "end": pd.Timestamp("2020-01-01 06:00"),
            "depth_mm": 10.0,
        },
        {
            "start": pd.Timestamp("2020-01-01 02:00"),
            "end": pd.Timestamp("2020-01-01 08:00"),
            "depth_mm": 25.0,
        },
        {
            "start": pd.Timestamp("2020-01-02 00:00"),
            "end": pd.Timestamp("2020-01-02 06:00"),
            "depth_mm": 5.0,
        },
    ]
    result = dedupe_windows(windows)
    assert len(result) == 2
    depths = sorted(w["depth_mm"] for w in result)
    assert depths == [5.0, 25.0]


def test_dedupe_windows_empty_returns_empty():
    assert dedupe_windows([]) == []


# ----------------------------------------------------------------------
# DDF table loading and lookup
# ----------------------------------------------------------------------


def make_ddf_csv(tmp_path):
    rows = []
    for d_h in [1, 3, 6, 24]:
        row = {"Duration": f"{d_h} h"}
        for ari, depth in [
            (2, 20.0 * d_h**0.3),
            (10, 35.0 * d_h**0.3),
            (100, 60.0 * d_h**0.3),
        ]:
            row[f"{ari}yr Depth (mm)"] = round(depth, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_path / "ddf.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_ddf_table_parses_durations_and_aris(tmp_path):
    path = make_ddf_csv(tmp_path)
    long_df = load_ddf_table(path)
    assert set(long_df["duration_min"].unique()) == {60.0, 180.0, 360.0, 1440.0}
    assert set(long_df["ari_years"].unique()) == {2.0, 10.0, 100.0}


def test_load_ddf_table_no_ari_columns_raises(tmp_path):
    df = pd.DataFrame({"Duration": ["1 h", "3 h"], "Notes": ["a", "b"]})
    path = tmp_path / "bad_ddf.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)return-period"):
        load_ddf_table(str(path))


def test_ddf_table_assign_aep_in_range(tmp_path):
    path = make_ddf_csv(tmp_path)
    long_df = load_ddf_table(path)
    table = DDFTable(long_df)
    # A depth roughly at the 10yr curve for 3h duration should be in-range
    depth_10yr, _ = table.depth_at(180.0, aep_from_ari(10.0))
    aep_pct, in_range = table.assign_aep(180.0, depth_10yr)
    assert in_range is True
    assert abs(aep_pct - aep_from_ari(10.0)) < 0.5


def test_ddf_table_assign_aep_out_of_range(tmp_path):
    path = make_ddf_csv(tmp_path)
    long_df = load_ddf_table(path)
    table = DDFTable(long_df)
    _, in_range = table.assign_aep(180.0, 1e6)  # absurdly large depth
    assert in_range is False


# ----------------------------------------------------------------------
# Time series loading
# ----------------------------------------------------------------------


def test_load_timeseries_detects_hourly_native_interval(tmp_path):
    index = pd.date_range("2020-01-01", periods=48, freq="h")
    values = np.zeros(48)
    values[5:8] = [1.0, 2.0, 1.0]
    df = pd.DataFrame({"Date_time": index, "Precip": values})
    path = tmp_path / "ts.csv"
    df.to_csv(path, index=False)

    series, native_minutes = load_timeseries(str(path))
    assert native_minutes == 60.0
    assert len(series) == 48


def test_load_timeseries_respects_explicit_datetime_column(tmp_path):
    """Auto-detection of the datetime column is NOT robust to column
    order when the other column is purely numeric: pd.to_datetime()
    silently treats small integers as nanosecond-epoch timestamps, so
    a numeric column listed before the real datetime column can be
    mis-detected as the datetime column (this is inherited, documented
    behaviour, not something this test suite tries to change). Given
    that, this test confirms the auto-detection instead does the right
    thing on its OWN, more reliable half: finding the value column
    automatically once the datetime column is given explicitly."""
    index = pd.date_range("2020-01-01", periods=24, freq="h")
    values = np.arange(24, dtype=float)
    df = pd.DataFrame({"SomeValue": values, "WhenItHappened": index})
    path = tmp_path / "ts2.csv"
    df.to_csv(path, index=False)

    series, native_minutes = load_timeseries(str(path), datetime_col="WhenItHappened")
    assert native_minutes == 60.0
    assert len(series) == 24


# ----------------------------------------------------------------------
# Full end-to-end engine run against synthetic data (mirrors the source
# plugin's own dev/test_synthetic.py smoke test, as real assertions)
# ----------------------------------------------------------------------


def _make_synthetic_timeseries(rng, years=8):
    start = pd.Timestamp("2010-01-01 00:00")
    n_hours = years * 365 * 24
    index = pd.date_range(start, periods=n_hours, freq="h")
    values = np.zeros(n_hours)
    n_events = int(years * 60)
    for _ in range(n_events):
        start_idx = rng.integers(0, n_hours - 200)
        dur_hours = int(rng.integers(1, 168))
        dur_hours = min(dur_hours, n_hours - start_idx - 1)
        if dur_hours < 1:
            continue
        ari_like = rng.choice(
            [1.2, 2, 5, 10, 20, 50, 100, 300],
            p=[0.30, 0.25, 0.18, 0.12, 0.08, 0.04, 0.02, 0.01],
        )
        target_depth = (
            5.0 * np.sqrt(dur_hours) * (ari_like**0.25) * rng.uniform(0.6, 1.1)
        )
        peak_frac = rng.uniform(0, 1)
        t = np.linspace(0, 1, dur_hours)
        shape = np.exp(-((t - peak_frac) ** 2) / (2 * 0.08**2))
        shape += rng.uniform(0, 0.15, size=dur_hours)
        shape = np.clip(shape, 0, None)
        if shape.sum() <= 0:
            continue
        profile = shape / shape.sum() * target_depth
        values[start_idx : start_idx + dur_hours] += profile
    return pd.Series(np.round(values, 4), index=index, name="Precip")


def _make_synthetic_ddf():
    durations_h = [1, 2, 3, 6, 9, 12, 18, 24, 48, 72, 120, 168]
    aris = [1.01, 1.111, 2, 5, 10, 20, 50, 100, 200, 500]

    def ddf_depth(duration_hours, ari_years):
        return 5.0 * np.sqrt(duration_hours) * (ari_years**0.25)

    rows = []
    for d in durations_h:
        row = {"Duration": f"{d} h"}
        for a in aris:
            row[f"{a}yr Depth (mm)"] = round(ddf_depth(d, a), 3)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_engine_result(tmp_path_factory):
    """Runs the full DesignStormEngine pipeline once against synthetic
    data (fewer Monte Carlo sims than production defaults, for test
    speed) and shares the result across every test in this module."""
    tmp_path = tmp_path_factory.mktemp("design_storm_synthetic")
    rng = np.random.default_rng(1234)

    ts = _make_synthetic_timeseries(rng)
    ts_df = ts.reset_index()
    ts_df.columns = ["Date_time", "Precip"]
    ts_path = tmp_path / "synthetic_timeseries.csv"
    ts_df.to_csv(ts_path, index=False)

    ddf_df = _make_synthetic_ddf()
    ddf_path = tmp_path / "synthetic_ddf.csv"
    ddf_df.to_csv(ddf_path, index=False)

    engine = DesignStormEngine(mc_sims=50)  # fewer sims for test speed
    return engine.run(str(ts_path), str(ddf_path))


def test_engine_run_produces_expected_row_count(synthetic_engine_result):
    assert len(synthetic_engine_result["ensemble_rows"]) == 24 * 3 * 3


def test_engine_run_identifies_events(synthetic_engine_result):
    assert synthetic_engine_result["n_events"] > 0


def test_engine_run_increment_counts_match_target_durations(synthetic_engine_result):
    duration_to_n = {d: n for d, ts, n in TARGET_DURATIONS}
    for row in synthetic_engine_result["ensemble_rows"]:
        if not row["Increments"]:
            continue
        assert len(row["Increments"]) == duration_to_n[row["Duration"]]


def test_engine_run_increments_sum_to_100(synthetic_engine_result):
    for row in synthetic_engine_result["ensemble_rows"]:
        if not row["Increments"]:
            continue
        total = sum(row["Increments"])
        assert abs(total - 100.0) < 0.5, (
            f"Duration={row['Duration']} AEP={row['AEP']} Shape={row['Shape']} "
            f"sums to {total:.2f}, not ~100"
        )


def test_engine_run_bin_summary_has_expected_status_categories(synthetic_engine_result):
    statuses = {row["status"] for row in synthetic_engine_result["bin_summary_rows"]}
    # With a reasonably long synthetic record, expect at least direct and
    # monte-carlo-disaggregated bins to appear (insufficient-data bins may
    # or may not, depending on random draw, so only assert the two
    # guaranteed-by-construction categories).
    assert "direct_from_data" in statuses or "direct_from_data_pooled" in statuses
    assert "monte_carlo_disaggregated" in statuses


def test_engine_run_catalogue_has_all_three_shapes_and_bands(synthetic_engine_result):
    catalogue = synthetic_engine_result["catalogue_rows"]
    shapes = {row["shape"] for row in catalogue}
    bands = {row["aep_band"] for row in catalogue}
    assert shapes == {"Early", "Middle", "Late"}
    assert bands.issubset({"frequent", "intermediate", "rare"})
    assert len(bands) > 0


def test_engine_run_raises_when_no_events_identified(tmp_path):
    index = pd.date_range("2020-01-01", periods=48, freq="h")
    values = np.zeros(48)  # entirely dry record
    df = pd.DataFrame({"Date_time": index, "Precip": values})
    ts_path = tmp_path / "dry_ts.csv"
    df.to_csv(ts_path, index=False)

    ddf_df = _make_synthetic_ddf()
    ddf_path = tmp_path / "ddf.csv"
    ddf_df.to_csv(ddf_path, index=False)

    engine = DesignStormEngine(mc_sims=10)
    with pytest.raises(ValueError, match="(?i)no storm events"):
        engine.run(str(ts_path), str(ddf_path))
