"""
Tests for d8_flow_direction/core.py. Run: python3 tests/test_core.py

Every direction-code assertion below is checked against the exact
encoding confirmed from WhiteboxTools' own source (see core.py's
module docstring) - not a generic/assumed D8 convention.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from mayim_tools.hydrology.d8_flow_direction.core import (
    OUTPUT_NODATA,
    cell_sizes_from_geotransform,
    compute_d8_pointer,
    compute_d8_pointer_from_file,
    read_raster,
    write_raster,
)

NAN = np.nan


def _flat9(center=10.0, fill=10.0):
    """A 3x3 array with the given fill elevation everywhere, centre
    cell overridden - the minimal case for testing a single centre
    cell's 8 neighbours in isolation."""
    a = np.full((3, 3), fill, dtype=np.float64)
    a[1, 1] = center
    return a


# ----------------------------------------------------------------------
# Direction encoding - each of the 8 directions individually, confirmed
# against WhiteboxTools' own source-confirmed encoding
# ----------------------------------------------------------------------


def test_direction_ne_is_1():
    a = _flat9(center=10.0)
    a[0, 2] = 5.0  # NE neighbour, strictly lower
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 1
    print("test_direction_ne_is_1: PASS")


def test_direction_e_is_2():
    a = _flat9(center=10.0)
    a[1, 2] = 5.0  # E
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 2
    print("test_direction_e_is_2: PASS")


def test_direction_se_is_4():
    a = _flat9(center=10.0)
    a[2, 2] = 5.0  # SE
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 4
    print("test_direction_se_is_4: PASS")


def test_direction_s_is_8():
    a = _flat9(center=10.0)
    a[2, 1] = 5.0  # S
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 8
    print("test_direction_s_is_8: PASS")


def test_direction_sw_is_16():
    a = _flat9(center=10.0)
    a[2, 0] = 5.0  # SW
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 16
    print("test_direction_sw_is_16: PASS")


def test_direction_w_is_32():
    a = _flat9(center=10.0)
    a[1, 0] = 5.0  # W
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 32
    print("test_direction_w_is_32: PASS")


def test_direction_nw_is_64():
    a = _flat9(center=10.0)
    a[0, 0] = 5.0  # NW
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 64
    print("test_direction_nw_is_64: PASS")


def test_direction_n_is_128():
    a = _flat9(center=10.0)
    a[0, 1] = 5.0  # N
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 128
    print("test_direction_n_is_128: PASS")


def test_esri_style_e_is_1():
    """ESRI convention: E=1 (vs default's E=2) - confirms the
    esri_style flag actually changes the output encoding, not just
    accepts the parameter."""
    a = _flat9(center=10.0)
    a[1, 2] = 5.0  # E
    result = compute_d8_pointer(
        a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0, esri_style=True
    )
    assert result[1, 1] == 1
    print("test_esri_style_e_is_1: PASS")


def test_esri_style_n_is_64():
    a = _flat9(center=10.0)
    a[0, 1] = 5.0  # N
    result = compute_d8_pointer(
        a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0, esri_style=True
    )
    assert result[1, 1] == 64
    print("test_esri_style_n_is_64: PASS")


# ----------------------------------------------------------------------
# Tie-breaking - the exact, order-dependent rule confirmed from source
# ----------------------------------------------------------------------


def test_tie_break_ne_beats_e_when_equal_slope():
    """NE (i=0) is checked before E (i=1) in WhiteboxTools' own
    iteration order - with a GENUINELY exact tie, NE must win.

    Deliberately avoids sqrt()-based construction: an earlier version
    of this test tried to tune a diagonal (NE) elevation to match a
    cardinal (E) slope via `5.0 * (1 - 1/sqrt(2))`-style arithmetic,
    but that construction itself introduces floating-point rounding,
    so the two slopes end up merely very close (differing in the last
    bit), not bit-exactly equal - the test was then unintentionally
    checking "whichever slope rounds a hair higher wins" rather than
    the actual tie-breaking rule. Comparing two DIAGONAL directions
    instead (both using the identical `diag` distance) with identical
    elevation drops keeps the arithmetic exactly symmetric - both
    slopes are computed as the exact same expression, so equality is
    genuine, not approximate."""
    a = _flat9(center=10.0, fill=10.0)
    a[0, 2] = 5.0  # NE (i=0) - drop of 5, distance diag
    a[2, 2] = (
        5.0  # SE (i=2) - identical drop, identical distance diag - a genuine, exact tie
    )
    diag = np.hypot(1.0, 1.0)
    slope_ne = (10.0 - a[0, 2]) / diag
    slope_se = (10.0 - a[2, 2]) / diag
    assert slope_ne == slope_se, "test setup did not produce a bit-exact tie"
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert (
        result[1, 1] == 1
    )  # NE's code (1), not SE's (4) - NE comes first in iteration order
    print("test_tie_break_ne_beats_e_when_equal_slope: PASS")


def test_tie_break_four_cardinal_directions_equal_e_wins():
    """A genuine, bit-exact tie among the four CARDINAL directions
    (E, S, W, N) - deliberately avoids the four diagonals here, since
    diagonal distances involve sqrt(2), and round-tripping an
    elevation constructed as `10.0 - drop*diag` back through
    `(10.0 - elevation)/diag` does NOT reliably reproduce the exact
    same float as a plain cardinal `(10.0 - elevation)/1.0` - confirmed
    directly (the diagonal round-trip gave 1.0000000000000004, not
    1.0, when this test was first written), which meant the original
    version of this test was inadvertently checking "whichever slope
    rounds a hair higher" rather than the tie-breaking rule itself.
    With all four ties confined to cardinal directions (exact
    arithmetic, no irrational distances), E must win since it is
    checked first (i=1) among the tied set."""
    a = _flat9(center=10.0, fill=10.0)
    a[1, 2] = 5.0  # E (i=1)
    a[2, 1] = 5.0  # S (i=3)
    a[1, 0] = 5.0  # W (i=5)
    a[0, 1] = 5.0  # N (i=7)
    slopes = [
        (10.0 - a[1, 2]) / 1.0,
        (10.0 - a[2, 1]) / 1.0,
        (10.0 - a[1, 0]) / 1.0,
        (10.0 - a[0, 1]) / 1.0,
    ]
    assert len(set(slopes)) == 1, "test setup did not produce a bit-exact 4-way tie"
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 2  # E's code - first among the tied cardinal directions
    print("test_tie_break_four_cardinal_directions_equal_e_wins: PASS")


# ----------------------------------------------------------------------
# No valid downslope neighbour -> direction 0 (not NoData, not
# arbitrary) - pits, flats, and edges all produce this per WhiteboxTools
# ----------------------------------------------------------------------


def test_pit_produces_zero():
    """Every neighbour is HIGHER than the centre - a local pit."""
    a = _flat9(center=1.0, fill=10.0)
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 0
    print("test_pit_produces_zero: PASS")


def test_flat_produces_zero():
    """Every neighbour is EQUAL to the centre - a flat area, not a
    pit, but the same output per WhiteboxTools (no STRICTLY lower
    neighbour exists)."""
    a = np.full((3, 3), 10.0)
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[1, 1] == 0
    print("test_flat_produces_zero: PASS")


def test_corner_cell_with_no_lower_neighbour_is_zero_not_crash():
    """A corner cell has only 3 real neighbours (the other 5 are
    off-grid) - if none of those 3 are lower, must resolve to 0
    cleanly, not error or wrap around."""
    a = np.array(
        [
            [10.0, 15.0],
            [15.0, 15.0],
        ]
    )
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[0, 0] == 0
    print("test_corner_cell_with_no_lower_neighbour_is_zero_not_crash: PASS")


def test_corner_cell_with_lower_neighbour_works_correctly():
    """Same corner, but now WITH a valid lower neighbour among the 3
    real ones - confirms edge/corner cells still compute a real
    direction correctly, not just fail safely."""
    a = np.array(
        [
            [10.0, 5.0],
            [15.0, 15.0],
        ]
    )
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert result[0, 0] == 2  # E, the only real lower neighbour
    print("test_corner_cell_with_lower_neighbour_works_correctly: PASS")


# ----------------------------------------------------------------------
# NoData handling - centre cell, neighbour cells, and NaN vs sentinel
# ----------------------------------------------------------------------


def test_nodata_centre_produces_output_nodata():
    a = _flat9(center=-9999.0, fill=10.0)
    result = compute_d8_pointer(
        a, nodata_value=-9999.0, cell_size_x=1.0, cell_size_y=1.0
    )
    assert result[1, 1] == OUTPUT_NODATA
    print("test_nodata_centre_produces_output_nodata: PASS")


def test_nodata_neighbour_is_excluded_not_treated_as_extreme():
    """A NoData neighbour must simply be skipped, not treated as
    infinitely low (which would wrongly make it the automatic winner)
    or infinitely high (which would just as wrongly make it never a
    candidate for a reason unrelated to its real, unknown elevation)."""
    a = _flat9(center=10.0, fill=10.0)
    a[1, 2] = -9999.0  # E is NoData
    a[2, 1] = 5.0  # S is a genuine, real lower neighbour
    result = compute_d8_pointer(
        a, nodata_value=-9999.0, cell_size_x=1.0, cell_size_y=1.0
    )
    assert result[1, 1] == 8  # S, not influenced by E's NoData value at all
    print("test_nodata_neighbour_is_excluded_not_treated_as_extreme: PASS")


def test_nan_nodata_value_handled_correctly():
    """NaN can't be compared with == (NaN != NaN) - confirms this is
    handled via isnan, not silently broken."""
    a = _flat9(center=np.nan, fill=10.0)
    result = compute_d8_pointer(
        a, nodata_value=np.nan, cell_size_x=1.0, cell_size_y=1.0
    )
    assert result[1, 1] == OUTPUT_NODATA
    print("test_nan_nodata_value_handled_correctly: PASS")


# ----------------------------------------------------------------------
# Non-square cell sizes - confirms cardinal vs diagonal distances are
# genuinely direction-specific, not a single assumed cell size
# ----------------------------------------------------------------------


def test_non_square_cells_affect_slope_correctly():
    """With cell_size_x=1, cell_size_y=10 (tall thin pixels), an E
    neighbour's slope is NOT scaled by the (irrelevant) y size, and an
    N neighbour's slope IS scaled by y, not x - set up so the 'wrong'
    cell size choice would flip which neighbour wins."""
    a = _flat9(center=10.0, fill=10.0)
    a[1, 2] = 9.0  # E: drop of 1 over distance cell_size_x=1 -> slope 1.0
    a[0, 1] = 5.0  # N: drop of 5 over distance cell_size_y=10 -> slope 0.5
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=10.0)
    assert (
        result[1, 1] == 2
    )  # E wins (slope 1.0 > 0.5), only correct if y-size wasn't misapplied to E
    print("test_non_square_cells_affect_slope_correctly: PASS")


# ----------------------------------------------------------------------
# A small, realistic multi-cell surface - confirms the vectorised
# whole-array computation matches expectations across many cells at
# once, not just isolated 3x3 single-cell tests
# ----------------------------------------------------------------------


def test_tilted_plane_all_cells_point_downhill_consistently():
    """A simple tilted plane (elevation decreases to the east) - every
    interior cell should point E (code 2), since E is the single
    steepest, and no diagonal or other direction ties it here."""
    rows, cols = 5, 5
    a = np.zeros((rows, cols))
    for c in range(cols):
        a[:, c] = (
            cols - c
        ) * 10.0  # strictly decreasing eastward, uniform down each column
    result = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    interior = result[1:-1, 1:-1]
    assert np.all(
        interior == 2
    ), f"expected all-E (2) interior, got unique values {np.unique(interior)}"
    print("test_tilted_plane_all_cells_point_downhill_consistently: PASS")


# ----------------------------------------------------------------------
# Raster I/O round-trip (GDAL) - only run if GDAL is available in this
# environment, matching the pattern used elsewhere in this project for
# environment-dependent tests
# ----------------------------------------------------------------------


def test_raster_round_trip_matches_direct_computation():
    try:
        from osgeo import gdal  # noqa: F401
    except ImportError:
        print(
            "test_raster_round_trip_matches_direct_computation: "
            "SKIPPED (GDAL not installed here)"
        )
        return

    import tempfile

    a = np.array(
        [
            [10.0, 9.0, 8.0],
            [11.0, 10.0, 9.0],
            [12.0, 11.0, 10.0],
        ]
    )
    tmp_dir = Path(tempfile.mkdtemp())
    in_path = str(tmp_dir / "dem.tif")
    out_path = str(tmp_dir / "d8.tif")

    geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)  # 1x1 cells, north-up
    projection = ""  # no CRS needed for this numerical round-trip check

    write_raster(in_path, a.astype(np.int16), geotransform, projection)
    # write_raster always sets NoData to OUTPUT_NODATA and writes int16 -
    # fine here since the test DEM has no NoData cells to worry about.

    compute_d8_pointer_from_file(in_path, out_path)

    result_direct = compute_d8_pointer(
        a, nodata_value=float(OUTPUT_NODATA), cell_size_x=1.0, cell_size_y=1.0
    )
    result_from_file, _, _, _ = read_raster(out_path)

    assert np.array_equal(result_direct, result_from_file.astype(np.int16))
    print("test_raster_round_trip_matches_direct_computation: PASS")


def test_cell_sizes_from_geotransform_uses_absolute_values():
    """GDAL geotransforms store pixel height as NEGATIVE for a
    north-up raster - must be returned as a positive cell size, not a
    negative one that would silently corrupt every slope calculation."""
    geotransform = (0.0, 2.5, 0.0, 0.0, 0.0, -2.5)
    cx, cy = cell_sizes_from_geotransform(geotransform)
    assert cx == 2.5 and cy == 2.5
    print("test_cell_sizes_from_geotransform_uses_absolute_values: PASS")


# ----------------------------------------------------------------------
# Default colour/label palette (used by the QGIS algorithm's
# postProcessAlgorithm() to style its output automatically)
# ----------------------------------------------------------------------


def test_direction_palette_default_scheme_has_nine_entries():
    from mayim_tools.hydrology.d8_flow_direction.core import direction_palette

    palette = direction_palette(esri_style=False)
    assert len(palette) == 9  # 8 directions + pit/flat
    print("test_direction_palette_default_scheme_has_nine_entries: PASS")


def test_direction_palette_default_scheme_value_2_is_e():
    from mayim_tools.hydrology.d8_flow_direction.core import direction_palette

    palette = direction_palette(esri_style=False)
    by_value = {value: (color, label) for value, color, label in palette}
    assert by_value[2] == ("#0076fd", "E")
    print("test_direction_palette_default_scheme_value_2_is_e: PASS")


def test_direction_palette_esri_scheme_value_1_is_e():
    """ESRI convention: E=1 (vs default's E=2) - confirms the palette
    correctly follows whichever scheme was actually used, not a fixed
    value->colour mapping regardless of esri_style."""
    from mayim_tools.hydrology.d8_flow_direction.core import direction_palette

    palette = direction_palette(esri_style=True)
    by_value = {value: (color, label) for value, color, label in palette}
    assert by_value[1] == ("#0076fd", "E")
    print("test_direction_palette_esri_scheme_value_1_is_e: PASS")


def test_direction_palette_includes_pit_entry_at_zero():
    from mayim_tools.hydrology.d8_flow_direction.core import direction_palette

    palette = direction_palette(esri_style=False)
    by_value = {value: (color, label) for value, color, label in palette}
    assert 0 in by_value
    assert by_value[0][1] == "Pit / flat (no lower neighbour)"
    print("test_direction_palette_includes_pit_entry_at_zero: PASS")


def test_direction_palette_all_values_unique():
    from mayim_tools.hydrology.d8_flow_direction.core import direction_palette

    for esri_style in (False, True):
        palette = direction_palette(esri_style=esri_style)
        values = [value for value, _, _ in palette]
        assert len(values) == len(
            set(values)
        ), f"duplicate values for esri_style={esri_style}"
    print("test_direction_palette_all_values_unique: PASS")


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
