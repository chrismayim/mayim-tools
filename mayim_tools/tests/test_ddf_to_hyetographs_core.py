"""
Tests for ddf_to_hyetographs/core.py and export.py. Run: python3 tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tempfile

import numpy as np
import pandas as pd

from mayim_tools.rainfall.ddf_to_hyetographs.core import (
    PEAK_RATIOS,
    alternating_block_arrange,
    build_hyetograph,
    compute_incremental_blocks,
    extrapolate_short_duration,
    interpolate_cumulative_depth,
    parse_idf_table,
)
from mayim_tools.rainfall.ddf_to_hyetographs.export import (
    _n_steps_for_duration,
    write_cumulative_depth,
    write_incremental_depth,
    write_incremental_intensity,
)

# ----------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------


def test_parse_idf_table_rt_header_format():
    df = pd.DataFrame(
        {
            "Duration": ["5 min", "10 min", "1 h"],
            "2yr Depth (mm)": [8.0, 12.0, 26.0],
            "10yr Depth (mm)": [14.0, 20.0, 45.0],
        }
    )
    table = parse_idf_table(df)
    assert set(table.keys()) == {2.0, 10.0}
    assert table[2.0] == [(5.0, 8.0), (10.0, 12.0), (60.0, 26.0)]
    print("test_parse_idf_table_rt_header_format: PASS")


def test_parse_idf_table_aep_percent_format():
    df = pd.DataFrame(
        {
            "Duration": ["5 min", "10 min"],
            "50%": [8.0, 12.0],
            "1%": [22.0, 32.0],
        }
    )
    table = parse_idf_table(df)
    assert set(table.keys()) == {2.0, 100.0}  # T = 100/AEP%
    print("test_parse_idf_table_aep_percent_format: PASS")


def test_parse_idf_table_bare_numeric_duration():
    df = pd.DataFrame({"Duration": [5, 10, 60], "2yr Depth (mm)": [8.0, 12.0, 26.0]})
    table = parse_idf_table(df)
    assert table[2.0][0][0] == 5.0
    print("test_parse_idf_table_bare_numeric_duration: PASS")


def test_parse_idf_table_auto_detects_time_column():
    df = pd.DataFrame({"Time": ["5 min", "10 min"], "2yr Depth (mm)": [8.0, 12.0]})
    table = parse_idf_table(df)
    assert 2.0 in table
    print("test_parse_idf_table_auto_detects_time_column: PASS")


def test_parse_idf_table_no_recognizable_columns_raises():
    df = pd.DataFrame({"Duration": ["5 min"], "SomeOtherColumn": [1.0]})
    try:
        parse_idf_table(df)
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_parse_idf_table_no_recognizable_columns_raises: PASS")


def test_parse_idf_table_missing_duration_column_raises():
    df = pd.DataFrame({"2yr Depth (mm)": [8.0]})
    try:
        parse_idf_table(df)
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_parse_idf_table_missing_duration_column_raises: PASS")


# ----------------------------------------------------------------------
# Interpolation and incremental blocks
# ----------------------------------------------------------------------


def test_interpolate_cumulative_depth_exact_at_tabulated_points():
    pairs = [(5.0, 8.0), (10.0, 12.0), (60.0, 26.0)]
    assert abs(interpolate_cumulative_depth(5.0, pairs) - 8.0) < 1e-9
    assert abs(interpolate_cumulative_depth(10.0, pairs) - 12.0) < 1e-9
    assert abs(interpolate_cumulative_depth(60.0, pairs) - 26.0) < 1e-9
    print("test_interpolate_cumulative_depth_exact_at_tabulated_points: PASS")


def test_interpolate_cumulative_depth_rejects_extrapolation():
    pairs = [(5.0, 8.0), (60.0, 26.0)]
    for bad in (4.0, 61.0):
        try:
            interpolate_cumulative_depth(bad, pairs)
            assert False, f"should have raised for {bad}"
        except ValueError:
            pass
    print("test_interpolate_cumulative_depth_rejects_extrapolation: PASS")


# ----------------------------------------------------------------------
# Short-duration extrapolation below the table's shortest tabulated
# point - added for a real case: ERA5-derived DDF tables are
# hourly-only, so a finer timestep (e.g. 5 min) needs cumulative-depth
# values below the table's own 60-min minimum.
# ----------------------------------------------------------------------


def test_extrapolate_short_duration_exact_at_first_tabulated_point():
    pairs = [(60.0, 45.0), (120.0, 56.0), (180.0, 64.0)]
    result = extrapolate_short_duration(60.0, pairs)
    assert abs(result - 45.0) < 1e-9
    print("test_extrapolate_short_duration_exact_at_first_tabulated_point: PASS")


def test_extrapolate_short_duration_goes_to_zero_at_zero():
    pairs = [(60.0, 45.0), (120.0, 56.0)]
    assert extrapolate_short_duration(0.0, pairs) == 0.0
    # strictly between 0 and the anchor value at any positive duration below t1
    value_at_1min = extrapolate_short_duration(1.0, pairs)
    assert 0.0 < value_at_1min < 45.0
    print("test_extrapolate_short_duration_goes_to_zero_at_zero: PASS")


def test_extrapolate_short_duration_matches_power_law_formula():
    """Confirms the exact formula, not just its boundary behaviour:
    D(t) = D1 * (t/t1)^b where b is the first segment's log-log slope."""
    pairs = [(60.0, 45.0), (120.0, 56.0)]
    t1, d1 = pairs[0]
    t2, d2 = pairs[1]
    b = np.log(d2 / d1) / np.log(t2 / t1)
    expected = d1 * (30.0 / t1) ** b
    result = extrapolate_short_duration(30.0, pairs)
    assert abs(result - expected) < 1e-9
    print("test_extrapolate_short_duration_matches_power_law_formula: PASS")


def test_extrapolate_short_duration_monotonically_increasing():
    """The extrapolated curve must itself be monotonic - depth cannot
    decrease as duration increases even within the extrapolated region."""
    pairs = [(60.0, 45.0), (120.0, 56.0)]
    values = [extrapolate_short_duration(t, pairs) for t in (1, 5, 10, 20, 30, 45, 60)]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    print("test_extrapolate_short_duration_monotonically_increasing: PASS")


def test_interpolate_cumulative_depth_short_duration_disabled_by_default():
    """Backward-compatible default: without the flag, a duration below
    the table's minimum still raises exactly as before."""
    pairs = [(60.0, 45.0), (120.0, 56.0)]
    try:
        interpolate_cumulative_depth(30.0, pairs)
        assert False, "should have raised by default"
    except ValueError as e:
        assert "below the table's shortest" in str(e)
    print("test_interpolate_cumulative_depth_short_duration_disabled_by_default: PASS")


def test_interpolate_cumulative_depth_short_duration_enabled():
    pairs = [(60.0, 45.0), (120.0, 56.0)]
    result = interpolate_cumulative_depth(
        30.0, pairs, allow_short_duration_extrapolation=True
    )
    assert 0 < result < 45.0
    print("test_interpolate_cumulative_depth_short_duration_enabled: PASS")


def test_interpolate_cumulative_depth_long_duration_never_extrapolates():
    """The flag only affects the SHORT-duration side - above the
    table's longest tabulated duration must still always raise,
    regardless of the flag."""
    pairs = [(60.0, 45.0), (120.0, 56.0)]
    try:
        interpolate_cumulative_depth(
            200.0, pairs, allow_short_duration_extrapolation=True
        )
        assert (
            False
        ), "should have raised - long-duration extrapolation is never allowed"
    except ValueError as e:
        assert "above the table's longest" in str(e)
    print("test_interpolate_cumulative_depth_long_duration_never_extrapolates: PASS")


def test_build_hyetograph_era5_style_hourly_only_table_with_extrapolation():
    """The actual real-world case reported: an hourly-only DDF table
    (ERA5-derived) with a 5-min timestep - without the flag this
    raises; with it, produces a complete hyetograph, still conserving
    total depth."""
    pairs = [(60.0, 45.0), (120.0, 56.0), (1440.0, 180.0)]
    try:
        build_hyetograph(1440.0, 5.0, pairs)
        assert False, "should have raised without the flag"
    except ValueError:
        pass

    hyeto = build_hyetograph(
        1440.0, 5.0, pairs, allow_short_duration_extrapolation=True
    )
    for ratio, data in hyeto.items():
        assert abs(sum(data["incremental"]) - 180.0) < 1e-6
    print("test_build_hyetograph_era5_style_hourly_only_table_with_extrapolation: PASS")


def test_compute_incremental_blocks_exact_multiple_sums_to_total():
    pairs = [(5.0, 8.0), (10.0, 12.0), (30.0, 20.0), (60.0, 26.0)]
    blocks = compute_incremental_blocks(60.0, 5.0, pairs)
    assert len(blocks) == 12
    assert abs(sum(blocks) - 26.0) < 1e-9
    print("test_compute_incremental_blocks_exact_multiple_sums_to_total: PASS")


def test_compute_incremental_blocks_remainder_clips_not_extrapolates():
    pairs = [(5.0, 8.0), (10.0, 12.0), (30.0, 20.0), (63.0, 27.0)]
    blocks = compute_incremental_blocks(63.0, 5.0, pairs)
    assert len(blocks) == 13  # 12 full + 1 short (3 min) final step
    assert abs(sum(blocks) - 27.0) < 1e-9
    print("test_compute_incremental_blocks_remainder_clips_not_extrapolates: PASS")


# ----------------------------------------------------------------------
# Alternating block arrangement
# ----------------------------------------------------------------------


def test_alternating_block_conserves_total_depth():
    blocks = [1.0, 5.0, 2.0, 8.0, 3.0, 6.0, 1.5]
    for ratio in PEAK_RATIOS:
        arranged = alternating_block_arrange(blocks, ratio)
        assert abs(sum(arranged) - sum(blocks)) < 1e-9
        assert sorted(arranged) == sorted(blocks)
    print("test_alternating_block_conserves_total_depth: PASS")


def test_alternating_block_peak_at_correct_position():
    blocks = [1.0, 2.0, 3.0, 8.0, 4.0, 5.0, 6.0]
    arranged = alternating_block_arrange(blocks, 0.5)
    peak_idx = arranged.index(max(blocks))
    expected_idx = round(0.5 * (len(blocks) - 1))
    assert peak_idx == expected_idx
    print("test_alternating_block_peak_at_correct_position: PASS")


def test_alternating_block_25_percent_peak_near_front():
    blocks = list(range(1, 13))  # 12 blocks, largest = 12
    arranged = alternating_block_arrange([float(b) for b in blocks], 0.25)
    peak_idx = arranged.index(12.0)
    assert peak_idx == round(0.25 * 11)
    print("test_alternating_block_25_percent_peak_near_front: PASS")


def test_alternating_block_single_value():
    assert alternating_block_arrange([5.0], 0.5) == [5.0]
    print("test_alternating_block_single_value: PASS")


# ----------------------------------------------------------------------
# build_hyetograph - the three output value types together
# ----------------------------------------------------------------------


def test_build_hyetograph_all_ratios_present():
    pairs = [(5.0, 8.0), (10.0, 12.0), (30.0, 20.0), (60.0, 26.0)]
    hyeto = build_hyetograph(60.0, 5.0, pairs)
    assert set(hyeto.keys()) == set(PEAK_RATIOS)
    print("test_build_hyetograph_all_ratios_present: PASS")


def test_build_hyetograph_cumulative_matches_running_sum_of_incremental():
    pairs = [(5.0, 8.0), (10.0, 12.0), (30.0, 20.0), (60.0, 26.0)]
    hyeto = build_hyetograph(60.0, 5.0, pairs)
    for ratio, data in hyeto.items():
        running = np.cumsum(data["incremental"])
        assert np.allclose(
            running, data["cumulative"]
        ), f"ratio {ratio}: cumulative doesn't match running sum"
    print("test_build_hyetograph_cumulative_matches_running_sum_of_incremental: PASS")


def test_build_hyetograph_intensity_equals_depth_over_time():
    pairs = [(5.0, 8.0), (10.0, 12.0), (30.0, 20.0), (60.0, 26.0)]
    timestep = 5.0
    hyeto = build_hyetograph(60.0, timestep, pairs)
    for ratio, data in hyeto.items():
        for depth, intensity in zip(data["incremental"], data["intensity_mmhr"]):
            expected = depth / (timestep / 60.0)
            assert abs(intensity - expected) < 1e-9
    print("test_build_hyetograph_intensity_equals_depth_over_time: PASS")


def test_build_hyetograph_cumulative_final_value_equals_total_depth():
    pairs = [(5.0, 8.0), (10.0, 12.0), (30.0, 20.0), (60.0, 26.0)]
    hyeto = build_hyetograph(60.0, 5.0, pairs)
    for ratio, data in hyeto.items():
        assert abs(data["cumulative"][-1] - 26.0) < 1e-9
    print("test_build_hyetograph_cumulative_final_value_equals_total_depth: PASS")


def test_build_hyetograph_intensity_conserves_depth_when_integrated():
    """Sum of (intensity * timestep_hours) must recover the total
    depth - confirms the intensity output is a genuine, consistent
    rate conversion, not just cosmetically derived."""
    pairs = [(5.0, 8.0), (10.0, 12.0), (30.0, 20.0), (60.0, 26.0)]
    timestep = 5.0
    hyeto = build_hyetograph(60.0, timestep, pairs)
    for ratio, data in hyeto.items():
        recovered_depth = sum(i * (timestep / 60.0) for i in data["intensity_mmhr"])
        assert abs(recovered_depth - 26.0) < 1e-6
    print("test_build_hyetograph_intensity_conserves_depth_when_integrated: PASS")


# ----------------------------------------------------------------------
# Export - all three CSVs, and the off-by-one column-count regression
# ----------------------------------------------------------------------


def test_n_steps_for_duration_exact_multiple_no_extra_column():
    """Regression test for a real bug found during this round of work:
    the header column count was unconditionally adding one extra
    'remainder' column even when the duration divided evenly into the
    timestep - confirmed directly (60 min at 5 min steps generated a
    spurious, always-empty '65min' column)."""
    assert _n_steps_for_duration(60.0, 5.0) == 12
    assert _n_steps_for_duration(30.0, 5.0) == 6
    print("test_n_steps_for_duration_exact_multiple_no_extra_column: PASS")


def test_n_steps_for_duration_remainder_gets_extra_column():
    assert _n_steps_for_duration(63.0, 5.0) == 13
    assert _n_steps_for_duration(7.0, 5.0) == 2
    print("test_n_steps_for_duration_remainder_gets_extra_column: PASS")


def _sample_tables():
    df = pd.DataFrame(
        {
            "Duration": ["5 min", "10 min", "15 min", "30 min", "1 h", "2 h"],
            "2yr Depth (mm)": [8, 12, 15, 20, 26, 32],
            "10yr Depth (mm)": [14, 20, 25, 34, 45, 56],
        }
    )
    return parse_idf_table(df)


def test_write_incremental_depth_header_matches_row_length():
    tables = _sample_tables()
    tmp = Path(tempfile.mkdtemp()) / "inc.csv"
    n = write_incremental_depth([(10.0, 60.0)], 5.0, tables, tmp)
    assert n == 1
    lines = tmp.read_text().splitlines()
    header_len = len(lines[0].split(","))
    row_len = len(lines[1].split(","))
    assert header_len == row_len
    # 2 metadata cols + 5 ratios * 12 steps (60min/5min, exact multiple)
    assert header_len == 2 + 5 * 12
    print("test_write_incremental_depth_header_matches_row_length: PASS")


def test_write_cumulative_depth_final_populated_cell_equals_total():
    tables = _sample_tables()
    tmp = Path(tempfile.mkdtemp()) / "cum.csv"
    write_cumulative_depth([(10.0, 60.0)], 5.0, tables, tmp)
    lines = tmp.read_text().splitlines()
    header = lines[0].split(",")
    row = lines[1].split(",")
    # last column of the "25%" block (12 steps: cols 2..13)
    last_25pct_col_idx = 2 + 12 - 1
    assert abs(float(row[last_25pct_col_idx]) - 45.0) < 1e-3  # 10yr @ 60min = 45mm
    print("test_write_cumulative_depth_final_populated_cell_equals_total: PASS")


def test_write_incremental_intensity_values_are_positive_rates():
    tables = _sample_tables()
    tmp = Path(tempfile.mkdtemp()) / "int.csv"
    write_incremental_intensity([(10.0, 60.0)], 5.0, tables, tmp)
    lines = tmp.read_text().splitlines()
    row = lines[1].split(",")
    values = [float(v) for v in row[2:] if v]
    assert all(v >= 0 for v in values)
    assert any(v > 0 for v in values)
    print("test_write_incremental_intensity_values_are_positive_rates: PASS")


def test_all_three_outputs_share_identical_structure():
    """The three CSVs must have identical headers and row structure -
    only the cell VALUES differ - so they're directly comparable
    side by side, as documented."""
    tables = _sample_tables()
    tmp_dir = Path(tempfile.mkdtemp())
    write_incremental_depth(
        [(10.0, 60.0), (2.0, 30.0)], 5.0, tables, tmp_dir / "inc.csv"
    )
    write_cumulative_depth(
        [(10.0, 60.0), (2.0, 30.0)], 5.0, tables, tmp_dir / "cum.csv"
    )
    write_incremental_intensity(
        [(10.0, 60.0), (2.0, 30.0)], 5.0, tables, tmp_dir / "int.csv"
    )

    headers = [
        (tmp_dir / f).read_text().splitlines()[0]
        for f in ("inc.csv", "cum.csv", "int.csv")
    ]
    assert headers[0] == headers[1] == headers[2]

    row_counts = [
        len((tmp_dir / f).read_text().splitlines())
        for f in ("inc.csv", "cum.csv", "int.csv")
    ]
    assert row_counts[0] == row_counts[1] == row_counts[2] == 3  # header + 2 data rows
    print("test_all_three_outputs_share_identical_structure: PASS")


def test_shorter_duration_row_pads_trailing_columns_blank():
    tables = _sample_tables()
    tmp = Path(tempfile.mkdtemp()) / "mixed.csv"
    # 60min needs 12 steps/ratio, 30min needs only 6 - the 30min row
    # must have blank trailing cells in each ratio's block, not error
    write_incremental_depth([(10.0, 60.0), (10.0, 30.0)], 5.0, tables, tmp)
    lines = tmp.read_text().splitlines()
    row_30min = lines[2].split(",")
    # col index 2+6 = 8 is the 7th step of the "25%" block - should be blank for the 30min row
    assert row_30min[2 + 6] == ""
    print("test_shorter_duration_row_pads_trailing_columns_blank: PASS")


def test_write_incremental_depth_era5_style_table_needs_flag():
    """Export-level confirmation of the real reported case: an
    hourly-only table with a 5-min timestep raises without the flag,
    succeeds with it."""
    df = pd.DataFrame(
        {
            "Duration": ["1 h", "2 h", "24 h"],
            "100yr Depth (mm)": [45.0, 56.0, 180.0],
        }
    )
    tables = parse_idf_table(df)
    tmp = Path(tempfile.mkdtemp()) / "era5.csv"

    try:
        write_incremental_depth([(100.0, 1440.0)], 5.0, tables, tmp)
        assert False, "should have raised without the flag"
    except ValueError:
        pass

    n = write_incremental_depth(
        [(100.0, 1440.0)], 5.0, tables, tmp, allow_short_duration_extrapolation=True
    )
    assert n == 1
    print("test_write_incremental_depth_era5_style_table_needs_flag: PASS")


if __name__ == "__main__":
    tests = [
        v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)
    ]
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
