"""
Tests for core/priority_flood.py.

Every hand-computed case's expected result is derived by manually
tracing the algorithm's own defining rule (a cell's filled elevation
is the greatest of its own elevation and whatever elevation first
reaches it from an already-resolved neighbour, flooding inward from
the domain's edges in increasing elevation order) - not copied from
any other tool's output, consistent with this whole plugin's
independent-implementation goal.

Run: python3 tests/test_priority_flood.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.priority_flood import (
    priority_flood_fill,
    priority_flood_fill_verbose,
)


def assert_no_remaining_depressions(filled: np.ndarray, nodata: float | None = None):
    """General validity check, reusable across any filled grid: no
    valid interior cell may be strictly lower than every one of its
    valid neighbours (that would be an unfilled pit) - every cell must
    have at least one neighbour at an equal or lower filled elevation,
    giving it a non-ascending step outward. Border cells are exempt
    (they're the domain's own edges, entitled to be local minima)."""
    rows, cols = filled.shape
    is_nodata = (
        (filled == nodata) if nodata is not None else np.zeros_like(filled, dtype=bool)
    )
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if is_nodata[r, c]:
                continue
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not is_nodata[nr, nc]:
                        neighbors.append(filled[nr, nc])
            if neighbors:
                assert (
                    min(neighbors) <= filled[r, c]
                ), f"cell ({r},{c}) is a remaining depression: {filled[r, c]} < all neighbours {neighbors}"


def test_flat_dem_unchanged():
    dem = np.full((5, 5), 5.0)
    filled = priority_flood_fill(dem)
    assert np.array_equal(filled, dem)
    print("test_flat_dem_unchanged: PASS")


def test_simple_single_cell_pit():
    """3x3: every border cell is itself the domain edge (seeded
    directly), so the pit must fill to the lowest of its 8
    neighbours."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    expected = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    filled = priority_flood_fill(dem)
    assert np.array_equal(filled, expected), f"got\n{filled}"
    print("test_simple_single_cell_pit: PASS")


def test_pit_fills_to_surrounding_ring_not_past_it():
    """5x5: a pit enclosed by a ring at elevation 5, with a gap in the
    OUTER border at elevation 3. The ring itself sits above the gap's
    elevation, so the ring cells need no raising - but the pit must
    still rise to the ring's own height (5) before it can reach the
    lower gap beyond it. This specifically tests that filling respects
    the nearest enclosing barrier, not just the lowest possible exit
    anywhere in the domain."""
    dem = np.array(
        [
            [9.0, 9.0, 3.0, 9.0, 9.0],
            [9.0, 5.0, 5.0, 5.0, 9.0],
            [9.0, 5.0, 1.0, 5.0, 9.0],
            [9.0, 5.0, 5.0, 5.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
        ]
    )
    expected = np.array(
        [
            [9.0, 9.0, 3.0, 9.0, 9.0],
            [9.0, 5.0, 5.0, 5.0, 9.0],
            [9.0, 5.0, 5.0, 5.0, 9.0],
            [9.0, 5.0, 5.0, 5.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
        ]
    )
    filled = priority_flood_fill(dem)
    assert np.array_equal(filled, expected), f"got\n{filled}"
    assert_no_remaining_depressions(filled)
    print("test_pit_fills_to_surrounding_ring_not_past_it: PASS")


def test_nodata_edge_treated_as_drainage_outlet():
    """A NoData column (simulating a clipped DEM tile edge) must let
    adjacent cells drain there, exactly like a real array border -
    the interior pit (not itself adjacent to NoData) must still fill
    correctly through the intervening ring."""
    nd = -9999.0
    dem = np.array(
        [
            [9.0, 9.0, 9.0, 9.0, nd],
            [9.0, 5.0, 5.0, 5.0, nd],
            [9.0, 5.0, 1.0, 5.0, nd],
            [9.0, 5.0, 5.0, 5.0, nd],
            [9.0, 9.0, 9.0, 9.0, nd],
        ]
    )
    filled = priority_flood_fill(dem, nodata=nd)
    assert filled[2, 2] == 5.0, f"pit should fill to 5.0, got {filled[2, 2]}"
    assert np.all(filled[:, 4] == nd), "NoData column must remain untouched"
    assert_no_remaining_depressions(filled, nodata=nd)
    print("test_nodata_edge_treated_as_drainage_outlet: PASS")


def test_never_lowers_a_cell():
    """Fundamental invariant, checked on a random grid: filling can
    only raise cells, never lower them."""
    rng = np.random.default_rng(42)
    dem = rng.uniform(0, 100, size=(25, 25))
    filled = priority_flood_fill(dem)
    assert np.all(filled >= dem - 1e-12)
    print("test_never_lowers_a_cell: PASS")


def test_no_remaining_depressions_on_random_terrain():
    """The defining correctness property, checked structurally across
    several random seeds rather than a single hand-picked grid."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        dem = rng.uniform(0, 100, size=(30, 30))
        filled = priority_flood_fill(dem)
        assert_no_remaining_depressions(filled)
    print("test_no_remaining_depressions_on_random_terrain: PASS")


def test_nested_depressions_fill_correctly():
    """A small, deep sub-pit nested inside a larger, shallower basin -
    the sub-pit must fill first to the inner rim, and (since the inner
    rim is itself below the outer rim) the whole inner region then
    fills further to the outer rim's spill elevation."""
    dem = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 6.0, 6.0, 6.0, 6.0, 6.0, 10.0],
            [10.0, 6.0, 3.0, 3.0, 3.0, 6.0, 10.0],
            [10.0, 6.0, 3.0, 0.5, 3.0, 6.0, 10.0],
            [10.0, 6.0, 3.0, 3.0, 3.0, 6.0, 10.0],
            [10.0, 6.0, 6.0, 6.0, 6.0, 6.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    filled = priority_flood_fill(dem)
    # the deepest cell must fill all the way to the OUTER rim (10),
    # since the inner rim (6) is itself enclosed by the outer one with
    # no lower exit anywhere in this fully-enclosed test grid
    assert filled[3, 3] == 10.0, f"expected 10.0, got {filled[3, 3]}"
    assert np.array_equal(
        filled, np.full((7, 7), 10.0)
    ), f"whole enclosed region should fill to 10.0:\n{filled}"
    assert_no_remaining_depressions(filled)
    print("test_nested_depressions_fill_correctly: PASS")


def test_every_valid_cell_gets_visited_regardless_of_nodata_shape():
    """A direct consequence of the seeding design (a valid cell
    adjacent to NoData is itself treated as a drainage point, the
    standard convention for a clipped DEM's boundary): any connected
    region of valid cells necessarily has its own boundary touching
    either NoData or the array edge, so it always gets seeded and
    flooded - there is no constructible 'landlocked, unreachable'
    case under this design. This test confirms that invariant
    directly (every valid cell ends up visited) across an irregular
    NoData shape, rather than asserting an unreachable scenario that
    the design specifically rules out.

    The pit sits at least two cells away from any NoData cell in every
    direction (including diagonally), so it is reached via genuine
    flood-filling through its own 5-elevation ring, not seeded
    directly by touching NoData itself - a first attempt at this test
    put the pit only one cell from NoData and it was (correctly)
    seeded directly by 8-connected adjacency, which wasn't the case
    being tested."""
    nd = -9999.0
    dem = np.array(
        [
            [nd, nd, nd, nd, nd, nd, nd, nd],
            [nd, 5.0, 5.0, 5.0, 5.0, 5.0, nd, nd],
            [nd, 5.0, 5.0, 5.0, 5.0, 5.0, nd, nd],
            [nd, 5.0, 5.0, 1.0, 5.0, 5.0, nd, nd],
            [nd, 5.0, 5.0, 5.0, 5.0, 5.0, nd, nd],
            [nd, 5.0, 5.0, 5.0, 5.0, 5.0, nd, nd],
            [nd, nd, nd, nd, nd, nd, nd, nd],
        ]
    )
    filled, visited = priority_flood_fill_verbose(dem, nodata=nd)
    valid_mask = dem != nd
    assert np.all(
        visited[valid_mask]
    ), "every valid cell must be reachable given this design's seeding rule"
    assert (
        filled[3, 3] == 5.0
    ), f"expected the pit (genuinely 2 cells from any NoData) to fill to 5.0, got {filled[3, 3]}"
    print("test_every_valid_cell_gets_visited_regardless_of_nodata_shape: PASS")


def test_single_row_dem():
    """Degenerate case: every cell is a border cell (a 1-row DEM has
    no interior at all) - must not crash, and nothing should change."""
    dem = np.array([[3.0, 1.0, 5.0, 2.0, 4.0]])
    filled = priority_flood_fill(dem)
    assert np.array_equal(
        filled, dem
    ), "every cell in a 1-row DEM is already a border/edge cell"
    print("test_single_row_dem: PASS")


if __name__ == "__main__":
    tests = [
        test_flat_dem_unchanged,
        test_simple_single_cell_pit,
        test_pit_fills_to_surrounding_ring_not_past_it,
        test_nodata_edge_treated_as_drainage_outlet,
        test_never_lowers_a_cell,
        test_no_remaining_depressions_on_random_terrain,
        test_nested_depressions_fill_correctly,
        test_every_valid_cell_gets_visited_regardless_of_nodata_shape,
        test_single_row_dem,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
