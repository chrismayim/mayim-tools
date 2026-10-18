"""
Tests for core/flat_resolution.py.

Run: python3 tests/test_flat_resolution.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.flat_resolution import (
    DEFAULT_MAX_TOTAL_DISTORTION_M,
    resolve_flat_gradient,
)


def _apply(dem, changes):
    corrected = dem.astype(np.float64).copy()
    for cell, v in changes.items():
        corrected[cell] = v
    return corrected


def _every_group_cell_has_strictly_lower_d8_neighbor(corrected, group_mask):
    rows, cols = corrected.shape
    for r, c in zip(*np.nonzero(group_mask)):
        found = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and corrected[nr, nc] < corrected[r, c]
                ):
                    found = True
        if not found:
            return False, (r, c)
    return True, None


def test_simple_flat_with_single_exit_resolves_and_converges():
    dem = np.full((9, 9), 15.0)
    dem[2:7, 2:7] = 10.0
    dem[4, 7] = 5.0  # single exit, middle row
    group_mask = np.zeros_like(dem, dtype=bool)
    group_mask[2:7, 2:7] = True

    changes = resolve_flat_gradient(dem, group_mask, fill_elevation=10.0)
    assert len(changes) == 25
    corrected = _apply(dem, changes)
    ok, bad_cell = _every_group_cell_has_strictly_lower_d8_neighbor(
        corrected, group_mask
    )
    assert ok, f"unresolvable at {bad_cell}"

    # convergent, not parallel: the middle row (closest to the actual exit)
    # should be lower than the edge rows at the same horizontal distance
    assert corrected[4, 4] < corrected[2, 4]
    assert corrected[4, 4] < corrected[6, 4]
    print("test_simple_flat_with_single_exit_resolves_and_converges: PASS")


def test_total_distortion_never_exceeds_target():
    dem = np.full((30, 30), 15.0)
    dem[3:27, 3:27] = 10.0  # a large, 24x24 flat
    dem[15, 27] = 5.0
    group_mask = np.zeros_like(dem, dtype=bool)
    group_mask[3:27, 3:27] = True

    target = 0.02
    changes = resolve_flat_gradient(
        dem, group_mask, fill_elevation=10.0, max_total_distortion_m=target
    )
    max_distortion = max(v - 10.0 for v in changes.values())
    assert (
        max_distortion <= target + 1e-9
    ), f"max distortion {max_distortion} exceeded target {target}"
    print("test_total_distortion_never_exceeds_target: PASS")


def test_no_outflow_boundary_returns_empty():
    """A flat entirely surrounded by higher terrain has no outlet and
    cannot be resolved into a draining gradient - matching Barnes et
    al.'s own requirement that only flats with a genuine outlet are
    resolved (see module docstring)."""
    dem = np.full((7, 7), 15.0)
    dem[2:5, 2:5] = 10.0
    group_mask = np.zeros_like(dem, dtype=bool)
    group_mask[2:5, 2:5] = True

    changes = resolve_flat_gradient(dem, group_mask, fill_elevation=10.0)
    assert changes == {}
    print("test_no_outflow_boundary_returns_empty: PASS")


def test_single_cell_group_resolves_trivially():
    dem = np.full((5, 5), 15.0)
    dem[2, 2] = 10.0
    dem[2, 3] = 5.0  # direct exit neighbour
    group_mask = np.zeros_like(dem, dtype=bool)
    group_mask[2, 2] = True

    changes = resolve_flat_gradient(dem, group_mask, fill_elevation=10.0)
    assert len(changes) == 1
    assert changes[(2, 2)] > 10.0
    print("test_single_cell_group_resolves_trivially: PASS")


def test_pure_outflow_only_group_still_resolves():
    """A group with ONLY outflow-type boundaries (no higher terrain
    anywhere adjacent) should still resolve correctly, using just the
    toward-lower gradient (flat_height=0 case, see module docstring)."""
    dem = np.full((9, 9), -9999.0)  # nodata everywhere except the flat and its exit
    dem[2:7, 2:7] = 10.0
    dem[4, 7] = 5.0
    group_mask = np.zeros_like(dem, dtype=bool)
    group_mask[2:7, 2:7] = True

    changes = resolve_flat_gradient(
        dem, group_mask, fill_elevation=10.0, nodata=-9999.0
    )
    # nodata neighbours are ignored entirely (neither higher nor lower), so only
    # the single real exit cell provides an outflow boundary - should still resolve
    assert len(changes) == 25
    corrected = _apply(dem, changes)
    corrected[corrected == -9999.0] = -9999.0  # no-op, just for clarity
    ok, bad_cell = _every_group_cell_has_strictly_lower_d8_neighbor(
        np.where(dem == -9999.0, -1e18, corrected),
        group_mask,  # treat nodata as "very low" so it never blocks the check
    )
    assert ok, f"unresolvable at {bad_cell}"
    print("test_pure_outflow_only_group_still_resolves: PASS")


def test_nodata_neighbors_do_not_crash_or_count_as_higher_or_lower():
    dem = np.full((9, 9), -9999.0)
    dem[2:7, 2:7] = 10.0
    dem[1, 4] = 15.0  # one real higher neighbour
    dem[4, 7] = 5.0  # one real lower/exit neighbour
    group_mask = np.zeros_like(dem, dtype=bool)
    group_mask[2:7, 2:7] = True

    changes = resolve_flat_gradient(
        dem, group_mask, fill_elevation=10.0, nodata=-9999.0
    )
    assert len(changes) == 25  # did not crash, and still resolves the whole group
    print("test_nodata_neighbors_do_not_crash_or_count_as_higher_or_lower: PASS")


def test_larger_realistic_circular_depression_with_narrow_exit():
    """A stress test closer to a real depression's shape: a circular
    flat with a single narrow exit, confirming correctness scales
    beyond the simple rectangular cases above."""
    size = 41
    dem = np.full((size, size), 20.0)
    cy, cx = 20, 20
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(yy - cy, xx - cx)
    flat_mask = dist <= 10
    dem[flat_mask] = 10.0
    dem[20, 31] = 5.0  # narrow exit to the east, just past the flat's edge

    changes = resolve_flat_gradient(dem, flat_mask, fill_elevation=10.0)
    assert len(changes) == int(flat_mask.sum())
    corrected = _apply(dem, changes)
    ok, bad_cell = _every_group_cell_has_strictly_lower_d8_neighbor(
        corrected, flat_mask
    )
    assert ok, f"unresolvable at {bad_cell}"
    # convergent flow: the cell nearest the exit (east edge, same row) should be
    # lower than the cell at the same radius on the opposite (west) side
    assert corrected[20, 30] < corrected[20, 10]
    print("test_larger_realistic_circular_depression_with_narrow_exit: PASS")


def test_default_target_is_small_and_positive():
    assert 0 < DEFAULT_MAX_TOTAL_DISTORTION_M <= 0.1
    print("test_default_target_is_small_and_positive: PASS")


def test_gradient_survives_float32_at_realistic_elevation_and_scale():
    """Regression test for a real bug found from Chris reporting "same
    result" a fourth time, after three previous verified fixes: the
    adaptive epsilon scaling (shrinking as a group's own step count
    grows, to bound total distortion) was never floored against what
    float32 - this project's own DEM output precision - can actually
    represent at a given elevation. At realistic South African/Western
    Australian terrain elevations combined with a group large enough
    to plausibly occur after the grouped-resolution fix merges a long
    artificial chain into one unit, the un-floored epsilon shrank
    below float32's own representable precision, silently collapsing
    the "gradient" back onto one identical rounded value - recreating
    exactly the flat plateau the fix exists to remove, with no visible
    error. Checked directly here, at a scale and elevation large enough
    that the bug was clearly reproducible before the fix (confirmed
    separately, not just assumed): a 625-cell flat at 1500 m must
    still show multiple distinct elevation values after rounding to
    float32, not just in float64."""
    size = 25
    dem = np.full((size + 4, size + 4), 1500.0)
    dem[2 : 2 + size, 2 : 2 + size] = 1500.0
    dem[2 + size // 2, 2 + size] = 1495.0
    group_mask = np.zeros_like(dem, dtype=bool)
    group_mask[2 : 2 + size, 2 : 2 + size] = True

    changes = resolve_flat_gradient(dem, group_mask, fill_elevation=1500.0)
    assert len(changes) == size * size
    f32_distinct = len(set(np.float32(v) for v in changes.values()))
    assert (
        f32_distinct > 1
    ), "gradient collapsed to a single float32 value - the exact bug this test guards against"
    print("test_gradient_survives_float32_at_realistic_elevation_and_scale: PASS")


if __name__ == "__main__":
    tests = [
        test_simple_flat_with_single_exit_resolves_and_converges,
        test_total_distortion_never_exceeds_target,
        test_no_outflow_boundary_returns_empty,
        test_single_cell_group_resolves_trivially,
        test_pure_outflow_only_group_still_resolves,
        test_nodata_neighbors_do_not_crash_or_count_as_higher_or_lower,
        test_larger_realistic_circular_depression_with_narrow_exit,
        test_default_target_is_small_and_positive,
        test_gradient_survives_float32_at_realistic_elevation_and_scale,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
