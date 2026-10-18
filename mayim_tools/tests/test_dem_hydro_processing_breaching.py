"""
Tests for core/breaching.py.

Every expected value below follows directly from this module's own
documented cost rule (the excavated channel stays flat at the pit's
own elevation until a cell whose NATURAL elevation is already lower
is found - confirmed to be the cost-minimizing choice by direct
comparison during development, see this module's own docstring) -
not copied from any other tool's output.

Run: python3 tests/test_breaching.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.breaching import (
    _excavation_along_path,
    least_cost_breach_from_pit,
)


def test_no_barrier_immediate_escape():
    """A pit with an already-lower neighbour needs no excavation at
    all - the cheapest possible case."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 3.0, 0.5],
            [5.0, 5.0, 5.0],
        ]
    )
    result = least_cost_breach_from_pit(dem, 1, 1)
    assert result.success
    assert result.total_cost == 0.0
    assert result.escape_cell == (1, 2)
    assert result.escape_elevation == 0.5
    print("test_no_barrier_immediate_escape: PASS")


def test_single_cell_barrier_breach_cost():
    """pit(1) -> barrier(10) -> escape(0.5), walled on both sides with
    NoData so this is the ONLY possible route (an earlier version of
    this test left the surrounding cells open at a low-enough
    elevation that a cheaper route around the barrier existed, which
    the algorithm correctly found - a flaw in the test's construction,
    not the algorithm; walling the corridor is the fix). The barrier's
    excavation TARGET is the pit's own level (1), so the search's own
    cost (used to compare candidate routes) is 10-1=9 exactly -
    unaffected by the flat-channel gradient fix below, since that fix
    only adjusts the final excavation applied to the chosen path, not
    the search's own cost accounting. The ACTUAL excavation applied is
    9.01, not 9.0 exactly: a tiny channel-gradient step (see
    breaching.py's own _excavation_along_path docstring for why a flat
    breach channel is its own real bug, found the same way as the
    fill-flat one) nudges the barrier very slightly lower than the
    pit's own level, so the channel has a genuine, if tiny,
    steepest-descent direction rather than sitting exactly flat with
    the pit."""
    nd = -9999.0
    dem = np.array(
        [
            [nd, nd, nd, nd],
            [nd, 1.0, 10.0, 0.5],
            [nd, nd, nd, nd],
        ]
    )
    result = least_cost_breach_from_pit(dem, 1, 1, nodata=nd)
    assert result.success
    assert result.total_cost == 9.0, f"got {result.total_cost}"
    assert result.escape_cell == (1, 3)
    barrier_cell = (1, 2)
    assert (
        abs(result.excavation[barrier_cell] - 9.01) < 1e-9
    ), f"got {result.excavation[barrier_cell]}"
    assert result.excavation[(1, 1)] == 0.0, "the pit itself must never be excavated"
    assert result.excavation[(1, 3)] == 0.0, "the escape cell must never be excavated"
    print("test_single_cell_barrier_breach_cost: PASS")


def test_cheaper_route_is_chosen():
    """Two possible directions out of the pit, each in its own
    NoData-isolated corridor so they cannot diagonally shortcut into
    each other: one is a genuine dead end (walled off after a single
    high barrier cell, no escape reachable that way at all), the
    other is a working, low-cost multi-cell corridor to a real escape.
    The algorithm must find the working route and its correct cost,
    not get stuck exploring the dead end."""
    nd = -9999.0
    dem = np.array(
        [
            [nd, nd, nd, nd, nd, nd],
            [nd, 20.0, nd, nd, nd, nd],
            [nd, 1.0, nd, nd, nd, nd],
            [nd, nd, 3.0, 3.0, 3.0, 0.2],
            [nd, nd, nd, nd, nd, nd],
        ]
    )
    result = least_cost_breach_from_pit(dem, 2, 1, nodata=nd)
    assert result.success
    # excavate (3,2)=3, (3,3)=3, (3,4)=3 each down to pit level 1 -> cost 2 each = 6 total
    assert result.total_cost == 6.0, f"got {result.total_cost}"
    assert result.escape_cell == (3, 5)
    assert (
        1,
        1,
    ) not in result.path_cells, "must not route through the dead-end barrier"
    print("test_cheaper_route_is_chosen: PASS")


def test_max_breach_length_constraint():
    """A barrier requiring a longer path than max_breach_length must
    fail, even though an unconstrained search would succeed."""
    dem = np.array([[5.0] * 8, [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.2], [5.0] * 8])
    unconstrained = least_cost_breach_from_pit(dem, 1, 0)
    assert unconstrained.success
    constrained = least_cost_breach_from_pit(dem, 1, 0, max_breach_length=2)
    assert (
        not constrained.success
    ), "a 2-cell length limit must not reach an escape 7 cells away"
    print("test_max_breach_length_constraint: PASS")


def test_max_breach_depth_constraint():
    """A barrier taller than the allowed single-cell excavation depth
    must fail, forcing the caller to fall back to filling for this
    depression. Walled with NoData so there is no alternative route -
    an earlier version of this test left the surrounding cells open,
    where they provided a legitimate cheaper route within the depth
    cap, which the algorithm correctly used - a flaw in the test, not
    the algorithm."""
    nd = -9999.0
    dem = np.array(
        [
            [nd, nd, nd, nd],
            [nd, 1.0, 50.0, 0.2],
            [nd, nd, nd, nd],
        ]
    )
    result = least_cost_breach_from_pit(dem, 1, 1, nodata=nd, max_breach_depth=10.0)
    assert (
        not result.success
    ), "a 49-unit excavation must not be allowed under a 10-unit depth cap"
    print("test_max_breach_depth_constraint: PASS")


def test_nodata_not_traversed():
    nd = -9999.0
    dem = np.array(
        [
            [nd, nd, nd, nd],
            [nd, 1.0, nd, 0.2],
            [nd, nd, nd, nd],
        ]
    )
    result = least_cost_breach_from_pit(dem, 1, 1, nodata=nd)
    # the only possible route is blocked by NoData - must fail, not route through it
    assert not result.success
    print("test_nodata_not_traversed: PASS")


def test_never_raises_anything_and_gradient_stays_within_bound():
    """General invariant, checked on a random configuration with a
    confirmed multi-cell SUCCESSFUL breach (an earlier version of this
    test used a seed/configuration where the search actually failed,
    so its own assertions never ran at all - vacuously passing
    regardless of what they checked; fixed by using a configuration
    confirmed to succeed with several intermediate cells, so the
    invariant below is actually exercised).

    Every excavation amount must be >= 0 (breaching only ever lowers
    cells, never raises them). The resulting elevation invariant has
    genuinely changed from an earlier version of this test: cells are
    no longer required to stay AT OR ABOVE the pit's own level exactly
    - the flat-channel gradient fix (see breaching.py's own
    _excavation_along_path docstring) deliberately nudges intermediate
    cells slightly below the pit's level, by design, to give the
    channel a genuine steepest-descent direction. What must still hold
    is that this nudge stays small: no cell may be cut more than
    max_total_distortion_m below the pit's own level."""
    nd = -9999.0
    dem = np.full((5, 10), nd)
    dem[1:4, 1:9] = 20.0  # a wide corridor
    dem[2, 2:7] = 10.0  # a 5-cell-long flat barrier run across it
    dem[2, 7] = (
        -2.0
    )  # escape - must be below the PIT's own level, not just the barrier's
    pit_row, pit_col = 2, 1
    dem[pit_row, pit_col] = -1.0  # the pit itself

    result = least_cost_breach_from_pit(dem, pit_row, pit_col, nodata=nd)
    assert result.success
    assert (
        len(result.path_cells) >= 5
    ), "test setup expects a multi-cell channel to meaningfully exercise this"

    pit_elev = dem[pit_row, pit_col]
    for cell, amount in result.excavation.items():
        assert amount >= 0.0
        if amount == 0.0:
            continue  # the pit and the escape cell themselves - never excavated, keep their own natural elevation
        resulting_elev = dem[cell] - amount
        assert (
            resulting_elev >= pit_elev - 0.01 - 1e-9
        ), f"cell {cell} excavated more than the allowed 0.01 m below the pit's own level: {resulting_elev} vs pit {pit_elev}"
    print("test_never_raises_anything_and_gradient_stays_within_bound: PASS")


def test_allowed_mask_forces_the_route_even_when_a_cheaper_one_exists():
    """Two routes from the pit, each in its own NoData-isolated
    corridor: a cheap one (barrier 2 m) and a dearer one (barrier 8 m).
    Unrestricted, the cheap one wins; restricted to the dear route's
    cells, the search must use it - this is how a breach gets guided
    along a mapped stream."""
    nd = -9999.0
    dem = np.array(
        [
            [nd, nd, nd, nd, nd, nd],
            [
                nd,
                nd,
                9.0,
                9.0,
                0.2,
                nd,
            ],  # dear route: 9 m barriers (excavation 8 each) to an escape
            [nd, 1.0, nd, nd, nd, nd],  # pit
            [
                nd,
                nd,
                3.0,
                3.0,
                0.2,
                nd,
            ],  # cheap route: 3 m barriers (excavation 2 each) to an escape
            [nd, nd, nd, nd, nd, nd],
        ]
    )
    free = least_cost_breach_from_pit(dem, 2, 1, nodata=nd)
    assert free.success and free.total_cost == 4.0 and free.escape_cell == (3, 4), free

    allowed = np.zeros(dem.shape, dtype=bool)
    allowed[1, 2:5] = True  # only the dear route's cells
    guided = least_cost_breach_from_pit(dem, 2, 1, nodata=nd, allowed_mask=allowed)
    assert guided.success and guided.escape_cell == (1, 4), guided
    assert guided.total_cost == 16.0, guided.total_cost
    assert all(
        allowed[c] for c in guided.path_cells[1:]
    ), "every cell after the pit must lie in the allowed mask"
    print("test_allowed_mask_forces_the_route_even_when_a_cheaper_one_exists: PASS")


def test_allowed_mask_with_no_valid_route_fails_cleanly():
    nd = -9999.0
    dem = np.array(
        [
            [nd, nd, nd, nd, nd],
            [nd, 1.0, 3.0, 0.2, nd],
            [nd, nd, nd, nd, nd],
        ]
    )
    allowed = np.zeros(dem.shape, dtype=bool)  # nothing allowed
    result = least_cost_breach_from_pit(dem, 1, 1, nodata=nd, allowed_mask=allowed)
    assert not result.success
    print("test_allowed_mask_with_no_valid_route_fails_cleanly: PASS")


def test_breach_channel_is_d8_resolvable_along_its_own_length():
    """THE direct test for the actual property this fix exists for:
    Chris reported flow accumulation disconnected specifically "within
    flowpaths and at confluences" even after the fill-flat fix was
    applied - because a breach channel, before this fix, was excavated
    perfectly flat along its own length (every intermediate cell
    excavated to exactly the pit's own elevation), which a D8 flow-
    direction algorithm cannot resolve a direction across, same as a
    filled flat. Checks the actual property that matters (each
    consecutive pair of cells along the channel has a strictly
    decreasing elevation) rather than only checking total excavation
    cost or that a route was found."""
    nd = -9999.0
    dem = np.full((5, 10), nd)
    dem[1:4, 1:9] = 20.0
    dem[2, 2:7] = 10.0
    dem[2, 7] = -2.0
    pit_row, pit_col = 2, 1
    dem[pit_row, pit_col] = -1.0

    result = least_cost_breach_from_pit(dem, pit_row, pit_col, nodata=nd)
    assert result.success
    assert len(result.path_cells) >= 5

    corrected = dem.astype(np.float64).copy()
    for cell, amount in result.excavation.items():
        corrected[cell] -= amount

    profile = [corrected[cell] for cell in result.path_cells]
    for i in range(len(profile) - 1):
        assert profile[i] > profile[i + 1], (
            f"channel not strictly decreasing at step {i}: {profile[i]} -> {profile[i+1]} "
            f"(cells {result.path_cells[i]} -> {result.path_cells[i+1]}) - not D8-resolvable here"
        )
    print("test_breach_channel_is_d8_resolvable_along_its_own_length: PASS")


def test_long_channel_gradient_survives_float32_at_realistic_elevation():
    """Regression test for a real bug found from Chris reporting "same
    result" a fourth time, after three previous verified fixes - see
    flat_resolution.py's own test of the identical issue for the fill
    case, and float32_safe_min_epsilon's own docstring for the full
    account. A breach channel is MORE exposed to this than a 2D fill
    region: a channel's own step count IS its length directly, with
    none of the square-root-of-area benefit a compact 2D flat gets,
    so a long channel at a realistic elevation is exactly where this
    bug was found to bite hardest (confirmed separately: a 300-cell
    channel at 1500 m collapsed to just 83 distinct float32 values
    before this fix, with many adjacent cells landing on the identical
    rounded value - not a total collapse, but enough partial collapse
    to break D8 resolution at those specific points, matching
    "disconnected within flowpaths" exactly). Checked here at the
    scale that originally reproduced it: every cell along a 300-cell
    channel at 1500 m must have a DISTINCT float32 value, in strictly
    decreasing order along the channel."""
    elev, n_cells = 1500.0, 300
    path = [(0, i) for i in range(n_cells + 2)]
    dem64 = np.full((1, n_cells + 2), elev + 20.0)
    pit_elev = elev
    escape_elev = elev - 10.0
    dem64[path[0]] = pit_elev
    dem64[path[-1]] = escape_elev

    excavation = _excavation_along_path(
        dem64, path, pit_elev, escape_elevation=escape_elev
    )
    profile_f32 = [np.float32(dem64[c] - excavation.get(c, 0.0)) for c in path]

    assert len(set(profile_f32)) == len(
        profile_f32
    ), "two or more cells collapsed to the same float32 value"
    for i in range(len(profile_f32) - 1):
        assert (
            profile_f32[i] > profile_f32[i + 1]
        ), f"not strictly decreasing at step {i}"
    print("test_long_channel_gradient_survives_float32_at_realistic_elevation: PASS")


if __name__ == "__main__":
    tests = [
        test_no_barrier_immediate_escape,
        test_single_cell_barrier_breach_cost,
        test_cheaper_route_is_chosen,
        test_max_breach_length_constraint,
        test_max_breach_depth_constraint,
        test_nodata_not_traversed,
        test_never_raises_anything_and_gradient_stays_within_bound,
        test_breach_channel_is_d8_resolvable_along_its_own_length,
        test_long_channel_gradient_survives_float32_at_realistic_elevation,
        test_allowed_mask_forces_the_route_even_when_a_cheaper_one_exists,
        test_allowed_mask_with_no_valid_route_fails_cleanly,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
