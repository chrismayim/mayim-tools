"""
Tests for core/impact_reduction.py.

Run: python3 tests/test_impact_reduction.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)
from mayim_tools.hydrology.dem_hydro_processing.core.impact_reduction import (
    apply_ira_to_all_leaves,
    resolve_leaf_with_least_impact,
)


def test_fill_impact_matches_hand_calculation():
    """3x3 single pit, surrounded directly by the array border at 5.0:
    filling raises exactly the one pit cell from 1.0 to 5.0 - total
    change = 4.0, 1 cell."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    result = resolve_leaf_with_least_impact(dem, h, leaf_id)
    assert result.fill_impact.num_cells == 1
    assert result.fill_impact.total_abs_change == 4.0
    print("test_fill_impact_matches_hand_calculation: PASS")


def test_fill_chosen_when_cheaper():
    """A wide, shallow basin (25 cells) enclosed by a rim that is
    itself a dead end (bordered by NoData beyond it, so it satisfies
    fill's "reaches the data edge" criterion but not breach's
    stricter "reaches a genuinely lower natural cell" criterion - see
    this module's own docstring on why these differ). The only
    genuine escape is far away past a tall, wide barrier, making
    breach's only valid path far more expensive than filling the
    basin up to its own dead-end rim."""
    nd = -9999.0
    size = 21
    cy, cx = 10, 6
    dem = np.full((size, size), nd)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    dem[dist <= 2] = 0.9
    dem[(dist > 2) & (dist <= 4)] = 1.0
    barrier_col_start = (
        cx + 5
    )  # contiguous with the rim's own rightmost extent - no accidental NoData gap
    dem[cy - 4 : cy + 5, barrier_col_start : barrier_col_start + 3] = 50.0
    dem[cy - 4 : cy + 5, barrier_col_start + 3] = 0.05

    h = build_depression_hierarchy(dem, nodata=nd)
    leaf_id = int(h.leaf_label[cy, cx])
    result = resolve_leaf_with_least_impact(dem, h, leaf_id, nodata=nd)
    assert result.method == "fill", (
        f"expected fill to be cheaper, got {result.method} "
        f"(fill={result.fill_impact.total_abs_change}, breach={result.breach_impact.total_abs_change if result.breach_impact else None})"
    )
    print("test_fill_chosen_when_cheaper: PASS")


def test_breach_chosen_when_cheaper():
    """Same wide, shallow basin and rim, but this time the genuine
    escape sits directly beyond a SHORT stretch of the rim's own
    northern edge - breaching only needs to cut through that short
    stretch (2 cells), while filling still needs to raise the entire
    25-cell basin."""
    nd = -9999.0
    size = 21
    cy, cx = 10, 10
    dem = np.full((size, size), nd)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    dem[dist <= 2] = 0.9
    dem[(dist > 2) & (dist <= 4)] = 1.0
    north_edge_row = cy - 4
    dem[north_edge_row - 1, cx - 1 : cx + 2] = (
        0.05  # short escape strip just past the rim's own edge
    )

    h = build_depression_hierarchy(dem, nodata=nd)
    leaf_id = int(h.leaf_label[cy, cx])
    result = resolve_leaf_with_least_impact(dem, h, leaf_id, nodata=nd)
    assert result.method == "breach", (
        f"expected breach to be cheaper, got {result.method} "
        f"(fill={result.fill_impact.total_abs_change}, breach={result.breach_impact.total_abs_change if result.breach_impact else None})"
    )
    print("test_breach_chosen_when_cheaper: PASS")


def test_breach_unavailable_falls_back_to_fill():
    """A pit fully enclosed with no possible escape within the given
    breach bounds - fill must be used regardless of its own cost,
    since it's the only option."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    # max_breach_length=0 makes breaching impossible from the very first step
    result = resolve_leaf_with_least_impact(dem, h, leaf_id, max_breach_length=0)
    assert result.method == "fill"
    assert result.breach_impact is None
    print("test_breach_unavailable_falls_back_to_fill: PASS")


def test_resolve_works_on_meta_depressions_now_but_still_raises_on_ocean():
    """A real change of behaviour, not a bug: resolve_leaf_with_least_impact
    now accepts meta-depressions as well as leaves - needed to resolve a
    whole fully-artificial chain or subtree as one unit rather than leaf by
    leaf (see this function's own updated docstring, and
    hierarchical_classification.group_into_resolution_units, for the real
    bug this fixes - resolving leaves individually to only their own direct
    parent left a staircase of residual depressions behind on a long
    artificial chain). An earlier version of this test asserted the OLD,
    leaf-only behaviour (a ValueError on a meta id) - replaced now that the
    capability is real and intentional. The ocean node is still rejected,
    since it has no spill_elevation of its own to resolve toward."""
    dem = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 8.0, 2.0, 8.0, 4.0, 8.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_a = int(h.leaf_label[2, 2])
    meta_id = h.nodes[leaf_a].parent
    result = resolve_leaf_with_least_impact(dem, h, meta_id)
    assert result.leaf_id == meta_id
    assert result.method in ("fill", "breach")
    try:
        resolve_leaf_with_least_impact(dem, h, 0)  # ocean
        assert False, "should have raised"
    except ValueError as e:
        assert "ocean" in str(e).lower()
    print("test_resolve_works_on_meta_depressions_now_but_still_raises_on_ocean: PASS")


def test_apply_ira_to_all_leaves_resolves_every_leaf():
    """Two independent pits under one shared basin - after applying
    IRA to all leaves, neither original pit cell should remain at its
    original (unresolved) low elevation."""
    dem = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 8.0, 2.0, 8.0, 4.0, 8.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    output, results = apply_ira_to_all_leaves(dem, h)
    assert (
        output[2, 2] > dem[2, 2]
    ), "pit A must have been resolved (raised or otherwise changed)"
    assert output[2, 4] > dem[2, 4], "pit B must have been resolved"
    assert len(results) == 2, f"expected 2 leaf results, got {len(results)}"
    print("test_apply_ira_to_all_leaves_resolves_every_leaf: PASS")


def test_apply_ira_leaves_are_disjoint_no_double_counting():
    """Confirms each leaf's changes are applied independently without
    one leaf's resolution overwriting another's - checked across
    random terrain by verifying every modified cell's new elevation
    is explained by exactly one leaf's own recorded elevation_changes."""
    rng = np.random.default_rng(11)
    dem = rng.uniform(0, 30, size=(20, 20))
    h = build_depression_hierarchy(dem)
    output, results = apply_ira_to_all_leaves(dem, h)
    seen_cells = set()
    for r in results:
        for cell in r.elevation_changes:
            assert (
                cell not in seen_cells
            ), f"cell {cell} modified by more than one leaf's result"
            seen_cells.add(cell)
            assert output[cell] == r.elevation_changes[cell]
    print("test_apply_ira_leaves_are_disjoint_no_double_counting: PASS")


def test_leaf_ids_subset_resolves_only_the_chosen_leaves():
    """Classification decides WHICH depressions to resolve; the
    unchosen ones must be left exactly as they were."""
    dem = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 8.0, 2.0, 8.0, 4.0, 8.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_a = int(h.leaf_label[2, 2])
    leaf_b = int(h.leaf_label[2, 4])
    output, results = apply_ira_to_all_leaves(dem, h, leaf_ids=[leaf_a])
    assert len(results) == 1 and results[0].leaf_id == leaf_a
    assert output[2, 2] > dem[2, 2], "the chosen leaf must be resolved"
    assert output[2, 4] == dem[2, 4], "the unchosen leaf must be left exactly as it was"
    none_out, none_results = apply_ira_to_all_leaves(dem, h, leaf_ids=[])
    assert none_results == [] and np.array_equal(none_out, dem)
    print("test_leaf_ids_subset_resolves_only_the_chosen_leaves: PASS")


if __name__ == "__main__":
    tests = [
        test_fill_impact_matches_hand_calculation,
        test_fill_chosen_when_cheaper,
        test_breach_chosen_when_cheaper,
        test_breach_unavailable_falls_back_to_fill,
        test_resolve_works_on_meta_depressions_now_but_still_raises_on_ocean,
        test_apply_ira_to_all_leaves_resolves_every_leaf,
        test_apply_ira_leaves_are_disjoint_no_double_counting,
        test_leaf_ids_subset_resolves_only_the_chosen_leaves,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
