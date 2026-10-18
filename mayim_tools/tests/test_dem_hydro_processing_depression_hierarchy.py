"""
Tests for core/depression_hierarchy.py.

Every hand-traced expected hierarchy below was worked out by tracing
the algorithm's own defining rules (flood inward from every local
minimum and the ocean simultaneously in increasing elevation order;
record a merge, and which elevation it happens at, whenever two
different regions' flood fronts first touch) - not copied from any
other tool's output.

Run: python3 tests/test_depression_hierarchy.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)


def test_single_pit_drains_directly_to_ocean():
    """3x3: a single local minimum, ringed directly by the array
    border. Its leaf must have the ocean as its immediate parent (no
    intermediate meta-node), spilling at the ring's own elevation."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    assert leaf_id != 0, "the pit must not be classified as ocean"
    leaf = h.nodes[leaf_id]
    assert (
        leaf.parent == 0
    ), f"expected the pit to drain directly to ocean, parent={leaf.parent}"
    assert (
        leaf.spill_elevation == 5.0
    ), f"expected spill at 5.0, got {leaf.spill_elevation}"
    assert leaf.children == []
    print("test_single_pit_drains_directly_to_ocean: PASS")


def test_two_pits_merge_with_each_other_before_draining_to_ocean():
    """Two separate local minima sharing one enclosing ring, itself
    enclosed by a higher outer border: the two pits must merge with
    EACH OTHER first (at the shared ring's elevation), and only that
    combined meta-depression subsequently drains to the ocean (at the
    outer border's higher elevation)."""
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
    leaf_a = int(h.leaf_label[2, 2])  # the elevation-2 pit
    leaf_b = int(h.leaf_label[2, 4])  # the elevation-4 pit
    assert leaf_a != leaf_b and leaf_a != 0 and leaf_b != 0

    assert (
        h.nodes[leaf_a].spill_elevation == 8.0
    ), f"got {h.nodes[leaf_a].spill_elevation}"
    assert (
        h.nodes[leaf_b].spill_elevation == 8.0
    ), f"got {h.nodes[leaf_b].spill_elevation}"
    assert (
        h.nodes[leaf_a].parent == h.nodes[leaf_b].parent
    ), "both pits must share the same meta-parent"
    meta_id = h.nodes[leaf_a].parent
    assert (
        meta_id != 0
    ), "the two pits must merge with EACH OTHER first, not go directly to ocean"
    meta = h.nodes[meta_id]
    assert set(meta.children) == {leaf_a, leaf_b}
    assert meta.parent == 0, "the combined meta-depression must then drain to ocean"
    assert (
        meta.spill_elevation == 10.0
    ), f"expected the meta-depression to spill to ocean at 10.0, got {meta.spill_elevation}"
    print("test_two_pits_merge_with_each_other_before_draining_to_ocean: PASS")


def test_flat_bottomed_pit_is_a_single_leaf():
    """A pit whose bottom is a flat, multi-cell plateau (all at the
    same minimum elevation) must be ONE leaf depression, not several -
    a flat floor is not multiple separate depressions."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 1.0, 1.0, 1.0, 5.0],
            [5.0, 1.0, 1.0, 1.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    flat_labels = {int(h.leaf_label[r, c]) for r in (1, 2) for c in (1, 2, 3)}
    assert (
        len(flat_labels) == 1
    ), f"expected one shared leaf for the flat floor, got labels {flat_labels}"
    leaf_id = next(iter(flat_labels))
    assert leaf_id != 0
    assert h.nodes[leaf_id].parent == 0
    assert h.nodes[leaf_id].spill_elevation == 5.0
    print("test_flat_bottomed_pit_is_a_single_leaf: PASS")


def test_cells_of_leaf_and_meta_node():
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
    meta_id = h.nodes[leaf_a].parent

    cells_a = h.cells_of(leaf_a)
    cells_b = h.cells_of(leaf_b)
    assert (
        cells_a[2, 2] and not cells_a[2, 4]
    ), "leaf A's cells must not include pit B's cell"
    assert cells_b[2, 4] and not cells_b[2, 2]
    assert not np.any(cells_a & cells_b), "leaf A and leaf B must not share any cells"

    cells_meta = h.cells_of(meta_id)
    assert np.array_equal(
        cells_meta, cells_a | cells_b
    ), "the meta-node's cells must be exactly the union of its two leaves"
    print("test_cells_of_leaf_and_meta_node: PASS")


def test_tree_well_formedness_on_random_terrain():
    """Structural invariants that must hold for ANY valid hierarchy,
    checked across several random DEMs rather than one hand-picked
    case: every node except the ocean has a parent; every non-leaf,
    non-ocean node has exactly two children; every leaf has zero
    children; the ocean is never anyone's child of a non-ocean node's
    child list by accident (it can only be node 0, the root)."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        dem = rng.uniform(0, 50, size=(20, 20))
        h = build_depression_hierarchy(dem)
        for node_id, node in h.nodes.items():
            if node.is_ocean:
                assert node.parent is None
                continue
            assert (
                node.parent is not None
            ), f"node {node_id} has no parent and is not the ocean"
            if node.children:
                assert (
                    len(node.children) == 2
                ), f"non-ocean, non-leaf node {node_id} has {len(node.children)} children"
            for child_id in node.children:
                assert h.nodes[child_id].parent == node_id
    print("test_tree_well_formedness_on_random_terrain: PASS")


def test_filled_dem_matches_priority_flood_fill():
    """The hierarchy construction runs its own flood in parallel with
    computing the filled DEM - that filled DEM must agree exactly with
    the standalone priority_flood_fill function, since both implement
    the same underlying rule. A real discrepancy here would mean the
    two implementations have diverged, which would be a genuine bug."""
    from mayim_tools.hydrology.dem_hydro_processing.core.priority_flood import (
        priority_flood_fill,
    )

    rng = np.random.default_rng(7)
    dem = rng.uniform(0, 50, size=(15, 15))
    h = build_depression_hierarchy(dem)
    standalone_filled = priority_flood_fill(dem)
    assert np.allclose(
        h.filled_dem, standalone_filled
    ), "hierarchy's own filled DEM disagrees with priority_flood_fill"
    print("test_filled_dem_matches_priority_flood_fill: PASS")


def test_nodata_hierarchy_construction():
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
    h = build_depression_hierarchy(dem, nodata=nd)
    leaf_id = int(h.leaf_label[2, 2])
    assert leaf_id != 0
    assert h.nodes[leaf_id].parent == 0
    assert (
        h.nodes[leaf_id].spill_elevation == 5.0
    ), f"got {h.nodes[leaf_id].spill_elevation}"
    print("test_nodata_hierarchy_construction: PASS")


def test_three_leaves_independent_branches_under_ocean():
    """Two pits that merge with each other, plus a THIRD, entirely
    separate pit elsewhere that drains directly to ocean on its own -
    confirms the ocean can hold multiple, independent branches at
    once, rather than being forced through successive binary merges
    that would misrepresent depressions that never actually interact."""
    dem = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 8.0, 2.0, 8.0, 4.0, 8.0, 10.0, 10.0, 6.0, 10.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_c = int(h.leaf_label[2, 8])  # the standalone pit at elevation 6
    assert leaf_c != 0
    assert (
        h.nodes[leaf_c].parent == 0
    ), "the standalone pit must drain directly to ocean"
    assert h.nodes[leaf_c].spill_elevation == 10.0
    assert len(h.nodes[leaf_c].children) == 0
    # and it must be a genuinely separate branch from the other two pits
    leaf_a = int(h.leaf_label[2, 2])
    assert leaf_c != leaf_a
    ocean_children = h.nodes[0].children
    assert leaf_c in ocean_children
    print("test_three_leaves_independent_branches_under_ocean: PASS")


def _valley_dem():
    """Valley with a 2-cell 10 m embankment; floor rises steeply upstream
    (so the pond only occupies floor cells below 10 m) and falls
    downstream. Footprint of the pond = row 7, cols 3-11 (9 cells)."""
    dem = np.full((15, 27), 20.0)
    for c in range(1, 27):
        if c <= 11:
            dem[7, c] = 3.0 + 0.8 * (11 - c)
        elif c >= 14:
            dem[7, c] = 2.6 - 0.05 * (c - 14)
    dem[7, 12:14] = 10.0
    return dem


def test_footprint_is_smaller_than_catchment_cells_of():
    """Regression test for a real bug: cells_of() is the leaf's
    CATCHMENT (every cell its flood claimed, including uphill cells
    above the spill elevation), NOT the depression. The size signal and
    stream relation both initially treated it as the footprint, so a
    9-cell pond counted as 36 cells."""
    dem = _valley_dem()
    h = build_depression_hierarchy(dem)
    leaf = int(h.leaf_label[7, 6])
    catchment = int(h.cells_of(leaf).sum())
    footprint = h.footprint_of(leaf, dem)
    assert (
        int(footprint.sum()) == 9
    ), f"expected the 9 pond cells, got {int(footprint.sum())}"
    assert catchment > 3 * int(
        footprint.sum()
    ), f"catchment {catchment} should dwarf footprint"
    assert sorted(set(np.argwhere(footprint)[:, 1].tolist())) == list(range(3, 12))
    assert not (
        footprint & ~h.cells_of(leaf)
    ).any(), "footprint must be a subset of the catchment"
    assert (dem[footprint] < h.nodes[leaf].spill_elevation).all()
    print("test_footprint_is_smaller_than_catchment_cells_of: PASS")


def test_cells_of_does_not_recurse_on_a_long_chain_hierarchy():
    """Regression test for a real crash report from real QGIS (not
    anticipated by any test until then): on real, noisy, basin-scale
    terrain, the hierarchy is typically a long CHAIN of sequential
    binary merges, not a balanced tree, so a recursive implementation
    of cells_of()'s own internal tree traversal could be O(number of
    leaves) stack frames deep. Every synthetic DEM in this test suite,
    before that crash, had at most a few hundred leaves - nowhere near
    enough to expose this. 4000 leaves here, deliberately chain-shaped
    (each pit merges with its immediate neighbour before the whole
    chain reaches ocean, confirmed below via a single ocean child
    covering the whole chain, not many independent small merges) -
    comfortably past the ~3000 that reproduced a RecursionError before
    the fix (see depression_hierarchy.py's own _descendant_leaf_ids
    docstring for the full account, including why this manifested as
    a hard Windows access-violation crash in QGIS itself rather than a
    catchable Python exception)."""
    n_pits = 4000
    nd = -9999.0
    width = n_pits * 2 + 3
    dem = np.full((5, width), nd)
    dem[1:4, 1 : width - 1] = 90.0
    for i in range(n_pits):
        col = 2 + i * 2
        dem[2, col] = 10.0 + i * 0.001
        if i < n_pits - 1:
            dem[2, col + 1] = 20.0 + i * 0.0001

    h = build_depression_hierarchy(dem, nodata=nd)
    n_leaves = sum(1 for nid, n in h.nodes.items() if not n.is_ocean and not n.children)
    assert n_leaves == n_pits
    ocean = h.nodes[0]
    assert (
        len(ocean.children) == 1
    ), "test setup expects one single chain, not several independent merges"

    top_level_id = ocean.children[0]
    mask = h.cells_of(top_level_id)  # must not raise RecursionError
    assert mask.sum() > 0
    print("test_cells_of_does_not_recurse_on_a_long_chain_hierarchy: PASS")


def test_footprint_of_ocean_is_empty():
    dem = _valley_dem()
    h = build_depression_hierarchy(dem)
    assert not h.footprint_of(0, dem).any()
    print("test_footprint_of_ocean_is_empty: PASS")


if __name__ == "__main__":
    tests = [
        test_single_pit_drains_directly_to_ocean,
        test_two_pits_merge_with_each_other_before_draining_to_ocean,
        test_flat_bottomed_pit_is_a_single_leaf,
        test_cells_of_leaf_and_meta_node,
        test_tree_well_formedness_on_random_terrain,
        test_filled_dem_matches_priority_flood_fill,
        test_nodata_hierarchy_construction,
        test_three_leaves_independent_branches_under_ocean,
        test_footprint_is_smaller_than_catchment_cells_of,
        test_footprint_of_ocean_is_empty,
        test_cells_of_does_not_recurse_on_a_long_chain_hierarchy,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
