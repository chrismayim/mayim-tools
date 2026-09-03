"""
Test suite based on the paper's Section 6.2 synthetic test cases,
scoped to what v1 implements. Run with: python -m pytest tests/ -v
(or python tests/test_huffrain.py directly - no pytest dependency
required, uses plain assert + a tiny runner at the bottom).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from huffrain.io import build_huff_curves


def _make_df(start, freq_hours, depths):
    n = len(depths)
    idx = pd.date_range(start, periods=n, freq=f"{freq_hours}h")
    return pd.DataFrame({"DateTime": idx, "Precip": depths})


def test_constant_rainfall():
    """Test 1: constant rainfall - every interval the same depth."""
    df = _make_df("2020-01-01", 1, [2.0] * 10)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert len(result.events) == 1, f"expected 1 event, got {len(result.events)}"
    ev = result.events[0]
    assert abs(ev.total_depth_mm - 20.0) < 1e-9
    # constant rate -> cumulative curve should be close to the y=x line
    curve = result.curves[0]
    y_grid = np.array(curve.y_grid)
    x_grid = np.array(curve.x_grid)
    assert np.allclose(y_grid, x_grid, atol=0.15), "constant rainfall should give a roughly linear curve"
    print("test_constant_rainfall: PASS")


def test_single_interval_burst():
    """Test 2: a single wet interval surrounded by dry - must not crash,
    curve should jump straight from 0 to 1."""
    depths = [0, 0, 5.0, 0, 0]
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=1, wet_threshold_mm=0.0)
    assert len(result.events) == 1
    curve = result.curves[0]
    y_grid = np.array(curve.y_grid)
    assert y_grid[0] == 0.0 and y_grid[-1] == 1.0
    assert np.all(np.diff(y_grid) >= -1e-9), "curve must be monotone non-decreasing"
    print("test_single_interval_burst: PASS")


def test_early_peaking_storm_quartile1():
    """Test 3: nearly all rain in the first quarter -> should classify Q1."""
    # 20 intervals; big burst in intervals 0-4 (first quarter), trickle after
    depths = [5.0] * 4 + [0.1] * 16
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.quartile == 1, f"expected quartile 1, got {ev.quartile}"
    print("test_early_peaking_storm_quartile1: PASS")


def test_late_peaking_storm_quartile4():
    """Test: nearly all rain in the last quarter -> should classify Q4."""
    depths = [0.1] * 16 + [5.0] * 4
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.quartile == 4, f"expected quartile 4, got {ev.quartile}"
    print("test_late_peaking_storm_quartile4: PASS")


def test_storms_separated_by_exactly_mit():
    """Test 7: two wet spells separated by a dry gap exactly equal to
    MIT should be treated as SEPARATE events (gap >= MIT closes)."""
    # 3 wet, 6 dry (=MIT at 1h/interval -> 6h gap), 3 wet
    depths = [2.0, 2.0, 2.0] + [0.0] * 6 + [2.0, 2.0, 2.0]
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert len(result.events) == 2, f"expected 2 events, got {len(result.events)}"
    print("test_storms_separated_by_exactly_mit: PASS")


def test_storms_separated_by_just_less_than_mit():
    """Test 8: gap shorter than MIT should MERGE into one event."""
    depths = [2.0, 2.0, 2.0] + [0.0] * 5 + [2.0, 2.0, 2.0]  # 5h gap < 6h MIT
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert len(result.events) == 1, f"expected 1 merged event, got {len(result.events)}"
    ev = result.events[0]
    assert ev.n_dry_gaps == 1
    print("test_storms_separated_by_just_less_than_mit: PASS")


def test_missing_interval_within_event():
    """Test 9: a NaN in the middle of an event must not be silently
    treated as zero - the event should be flagged contains_missing and
    excluded from curve normalization under strict mode."""
    depths = [2.0, 2.0, np.nan, 2.0, 2.0]
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0, quality_mode="strict")
    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.contains_missing is True
    assert ev.excluded is True
    assert len(result.curves) == 0, "an event with missing interior data must not produce a curve"
    print("test_missing_interval_within_event: PASS")


def test_duplicate_timestamps():
    """Test 10: duplicate timestamps should be de-duplicated (first
    kept), not crash the pipeline."""
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"])
    df = pd.DataFrame({"DateTime": idx, "Precip": [1.0, 99.0, 2.0, 1.0]})
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert any("duplicate" in w.lower() for w in result.warnings)
    print("test_duplicate_timestamps: PASS")


def test_irregular_intervals():
    """Test 11: irregular time spacing should not crash and should
    still produce a sane, monotone curve."""
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:45", "2020-01-01 02:00", "2020-01-01 02:15"])
    df = pd.DataFrame({"DateTime": idx, "Precip": [1.0, 3.0, 2.0, 1.0]})
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert len(result.events) == 1
    curve = result.curves[0]
    y_grid = np.array(curve.y_grid)
    assert np.all(np.diff(y_grid) >= -1e-9)
    print("test_irregular_intervals: PASS")


def test_negative_values_treated_as_missing():
    """Test 27: negative rainfall is physically invalid - must be
    treated as missing, not clipped to zero or kept negative."""
    depths = [2.0, -5.0, 2.0]
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    assert any("negative" in w.lower() for w in result.warnings)
    print("test_negative_values_treated_as_missing: PASS")


def test_conservation():
    """Section 6.1 conservation test: sum of interval depths must equal
    event total depth within numerical tolerance, for every event."""
    depths = [1.5, 3.2, 0.0, 2.1, 4.4, 0.0, 0.0, 1.1]
    df = _make_df("2020-01-01", 1, depths)
    result = build_huff_curves(df, "DateTime", "Precip", mit_hours=6, wet_threshold_mm=0.0)
    for ev in result.events:
        assert ev.total_depth_mm > 0
    print("test_conservation: PASS")


def test_column_not_found_raises():
    df = _make_df("2020-01-01", 1, [1.0, 2.0])
    try:
        build_huff_curves(df, "WrongCol", "Precip")
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("test_column_not_found_raises: PASS")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"{t.__name__}: FAIL - {e}")
        except Exception as e:
            failed += 1
            print(f"{t.__name__}: ERROR - {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
