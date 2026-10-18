"""
Tests for core/hierarchical_classification.py.

Run: python3 tests/test_hierarchical_classification.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
    LABEL_LIKELY_ARTIFICIAL,
    LABEL_LIKELY_NATURAL,
    LABEL_UNCERTAIN,
)
from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)
from mayim_tools.hydrology.dem_hydro_processing.core.hierarchical_classification import (
    classify_hierarchy,
)


def test_fragmented_noisy_lake_is_recognised_and_protected_at_meta_level():
    """THE regression test for the finding that motivated this whole
    module: a genuinely deep (5m), real depression with small floor
    noise fragments into many noise-scale leaves - none of which,
    individually, has anything like the lake's own true depth. This
    confirms the fix directly: the fragmentation is real (dozens of
    leaves), yet every one of them ends up protected by a Likely
    Natural meta-depression ancestor, and none are selected for
    resolution."""
    rng = np.random.default_rng(42)
    size = 25
    dem = np.full((size, size), 10.0)
    cy, cx = 12, 12
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    basin = dist <= 8
    dem[basin] = 5.0 + rng.uniform(-0.05, 0.05, size=basin.sum())

    h = build_depression_hierarchy(dem)
    basin_leaves = [
        nid
        for nid, n in h.nodes.items()
        if not n.is_ocean and not n.children and h.footprint_of(nid, dem)[basin].any()
    ]
    assert (
        len(basin_leaves) > 10
    ), f"expected the noisy floor to genuinely fragment into many leaves, got {len(basin_leaves)}"

    result = classify_hierarchy(dem, h, vertical_error_std=0.1, n_iterations=30, seed=1)

    assert (
        result.resolve_leaf_ids == []
    ), f"no basin-floor leaf should be resolved, got {result.resolve_leaf_ids}"
    for leaf_id in basin_leaves:
        assert (
            leaf_id in result.protected_by
        ), f"leaf {leaf_id} was not protected by any natural ancestor"
        protector_id = result.protected_by[leaf_id]
        assert result.all_classifications[protector_id].label == LABEL_LIKELY_NATURAL
    print("test_fragmented_noisy_lake_is_recognised_and_protected_at_meta_level: PASS")


def test_isolated_tiny_artifact_with_no_natural_ancestor_is_resolved():
    """A genuinely tiny, shallow, isolated pit - not part of any
    larger real feature - should still be resolved, exactly matching
    the existing leaf-only behaviour for this simple case."""
    dem = np.full((20, 20), 10.0)
    dem[10, 10] = 10.0 - 0.005  # far shallower than the assumed noise
    h = build_depression_hierarchy(dem)
    result = classify_hierarchy(
        dem, h, vertical_error_std=0.1, n_iterations=100, seed=2
    )
    leaf_id = int(h.leaf_label[10, 10])
    assert (
        leaf_id in result.resolve_leaf_ids
    ), f"expected the tiny isolated artifact to be resolved, got {result.resolve_leaf_ids}"
    assert leaf_id not in result.protected_by
    print("test_isolated_tiny_artifact_with_no_natural_ancestor_is_resolved: PASS")


def test_two_pit_natural_meta_protects_both_children():
    """The earlier, hand-verified two-pit hierarchy test, re-used here
    to directly confirm meta-level classification and protection in a
    small, fully traceable case (not just the large stochastic one
    above)."""
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
    assert meta_id == h.nodes[leaf_b].parent

    result = classify_hierarchy(dem, h, vertical_error_std=0.2, n_iterations=50, seed=3)
    assert meta_id in result.all_classifications
    assert result.all_classifications[meta_id].label == LABEL_LIKELY_NATURAL
    assert leaf_a in result.protected_by and result.protected_by[leaf_a] == meta_id
    assert leaf_b in result.protected_by and result.protected_by[leaf_b] == meta_id
    assert leaf_a not in result.resolve_leaf_ids
    assert leaf_b not in result.resolve_leaf_ids
    print("test_two_pit_natural_meta_protects_both_children: PASS")


def test_all_classifications_includes_leaves_and_metas():
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
    result = classify_hierarchy(dem, h, vertical_error_std=0.2, n_iterations=30, seed=4)
    assert leaf_a in result.all_classifications
    assert meta_id in result.all_classifications
    print("test_all_classifications_includes_leaves_and_metas: PASS")


def test_uncertain_meta_does_not_block_descent_into_children():
    """A meta-depression that is merely Uncertain (not confidently
    Natural) must not protect its children automatically - the walk
    should continue into them, potentially resolving an individually
    artificial leaf beneath an uncertain parent. Constructed with a
    threshold band wide enough to force the meta into Uncertain
    regardless of its own moderate confidence."""
    dem = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 9.5, 9.5, 9.5, 9.5, 9.5, 10.0],
            [10.0, 9.5, 9.0, 9.5, 8.9, 9.5, 10.0],
            [10.0, 9.5, 9.5, 9.5, 9.5, 9.5, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_a = int(h.leaf_label[2, 2])
    meta_id = h.nodes[leaf_a].parent
    # force the meta into Uncertain territory with a wide band, regardless of its own score
    result = classify_hierarchy(
        dem,
        h,
        vertical_error_std=2.0,
        n_iterations=30,
        seed=5,
        confidence_thresholds=(0.01, 0.99),
    )
    assert (
        result.all_classifications[meta_id].label == LABEL_UNCERTAIN
    ), f"test setup assumption failed: meta was {result.all_classifications[meta_id].label}, not Uncertain"
    assert (
        leaf_a not in result.protected_by
    ), "an Uncertain meta must not auto-protect its children"
    print("test_uncertain_meta_does_not_block_descent_into_children: PASS")


def test_reproducible_with_same_seed():
    rng = np.random.default_rng(7)
    dem = np.full((20, 20), 10.0) + rng.uniform(-0.02, 0.02, size=(20, 20))
    dem[5:10, 5:10] -= 3.0
    h = build_depression_hierarchy(dem)
    r1 = classify_hierarchy(dem, h, vertical_error_std=0.1, n_iterations=20, seed=99)
    r2 = classify_hierarchy(dem, h, vertical_error_std=0.1, n_iterations=20, seed=99)
    assert r1.resolve_leaf_ids == r2.resolve_leaf_ids
    assert r1.protected_by == r2.protected_by
    for node_id in r1.all_classifications:
        assert (
            r1.all_classifications[node_id].confidence
            == r2.all_classifications[node_id].confidence
        )
    print("test_reproducible_with_same_seed: PASS")


def test_zero_volume_leaves_excluded_from_classification():
    """A flat plateau 'leaf' (a local-minimum candidate group with no
    actual depression, per the hierarchy's own leaf-seeding rule) must
    not appear in all_classifications at all, and must not crash the
    walk."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    result = classify_hierarchy(dem, h, vertical_error_std=0.1, n_iterations=10, seed=6)
    assert result.all_classifications == {}
    assert result.resolve_leaf_ids == []
    print("test_zero_volume_leaves_excluded_from_classification: PASS")


def test_decide_resolution_from_labels_does_not_recurse_on_a_long_chain_hierarchy():
    """The matching regression test for hierarchical_classification's
    own _walk_and_decide - see
    depression_hierarchy.DepressionHierarchy._descendant_leaf_ids's own
    docstring for the full account of the real crash this whole class
    of bug was found from. This walk has the identical shape (descend
    one child at a time through a binary-merge hierarchy) as the
    function that actually crashed, so it carried the same risk even
    before being reported to crash itself."""
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
    leaf_ids = [nid for nid, n in h.nodes.items() if not n.is_ocean and not n.children]
    assert len(leaf_ids) == n_pits

    from mayim_tools.hydrology.dem_hydro_processing.core.hierarchical_classification import (
        decide_resolution_from_labels,
    )

    # every leaf labelled Likely Artificial - forces the walk to descend all
    # the way to every leaf down the full chain, exercising the worst case
    labels = {
        nid: LABEL_LIKELY_ARTIFICIAL for nid in h.nodes if not h.nodes[nid].is_ocean
    }
    result = decide_resolution_from_labels(h, labels)  # must not raise RecursionError
    assert len(result.resolve_leaf_ids) == n_pits
    print(
        "test_decide_resolution_from_labels_does_not_recurse_on_a_long_chain_hierarchy: PASS"
    )


def _chain_dem(n_pits, nd=-9999.0):
    """A 1D valley of pits with increasing depth, separated by internal
    rims lower than the outer wall - forces sequential chain merging.
    Shared by the tests below; also the exact construction used to
    find and confirm the real residual-depression bug (see
    group_into_resolution_units's own docstring)."""
    width = n_pits * 2 + 3
    dem = np.full((5, width), nd)
    dem[1:4, 1 : width - 1] = 90.0
    for i in range(n_pits):
        col = 2 + i * 2
        dem[2, col] = 10.0 + i * 1.0
        if i < n_pits - 1:
            dem[2, col + 1] = 20.0 + i * 0.5
    return dem


def test_group_into_resolution_units_fully_artificial_chain_leaves_zero_residuals():
    """THE regression test for the real bug this function fixes (see
    its own module docstring for the full account): a 29-leaf
    synthetic chain, every leaf classified artificial, left 28
    residual depressions behind when resolved leaf-by-leaf to only
    each one's own direct parent. Grouped resolution must leave none."""
    from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
        compute_depression_metrics,
    )
    from mayim_tools.hydrology.dem_hydro_processing.core.hierarchical_classification import (
        group_into_resolution_units,
    )
    from mayim_tools.hydrology.dem_hydro_processing.core.impact_reduction import (
        apply_ira_to_all_leaves,
    )

    nd = -9999.0
    dem = _chain_dem(29, nd=nd)
    h = build_depression_hierarchy(dem, nodata=nd)
    leaf_ids = [nid for nid, n in h.nodes.items() if not n.is_ocean and not n.children]

    units = group_into_resolution_units(h, leaf_ids)
    assert (
        len(units) == 1
    ), f"a fully-artificial chain should group into exactly one unit, got {len(units)}"

    corrected, results = apply_ira_to_all_leaves(dem, h, nodata=nd, leaf_ids=units)
    h2 = build_depression_hierarchy(corrected, nodata=nd)
    residual = [
        nid
        for nid, n in h2.nodes.items()
        if not n.is_ocean
        and not n.children
        and compute_depression_metrics(corrected, h2, nid).volume > 0
    ]
    assert residual == [], f"expected zero residual depressions, got {len(residual)}"
    print(
        "test_group_into_resolution_units_fully_artificial_chain_leaves_zero_residuals: PASS"
    )


def test_group_into_resolution_units_does_not_cross_an_unresolved_sibling():
    """A leaf NOT selected for resolution must never have its own
    cells pulled into a neighbouring group - confirmed directly by
    checking for footprint overlap, not just by trusting the grouping
    logic."""
    from mayim_tools.hydrology.dem_hydro_processing.core.hierarchical_classification import (
        group_into_resolution_units,
    )

    nd = -9999.0
    dem = _chain_dem(10, nd=nd)
    h = build_depression_hierarchy(dem, nodata=nd)
    leaf_ids = sorted(
        nid for nid, n in h.nodes.items() if not n.is_ocean and not n.children
    )
    excluded = leaf_ids[5]
    resolve_subset = [l for l in leaf_ids if l != excluded]

    units = group_into_resolution_units(h, resolve_subset)
    excluded_mask = h.cells_of(excluded)
    for u in units:
        overlap = (h.cells_of(u) & excluded_mask).sum()
        assert (
            overlap == 0
        ), f"unit {u} incorrectly overlaps the excluded leaf's own cells"
    print("test_group_into_resolution_units_does_not_cross_an_unresolved_sibling: PASS")


def test_KNOWN_LIMITATION_scattered_exclusion_mid_chain_can_still_leave_a_residual():
    """Documents a real, known, NOT YET FULLY SOLVED limitation rather
    than hiding it: because this chain topology means every ancestor
    transitively contains ALL earlier leaves (not just its own two
    direct children), a single excluded leaf part-way through a long
    chain 'poisons' every subsequent ancestor, so leaves AFTER the
    exclusion point cannot group with each other even though they are
    immediate physical neighbours - reverting to the original
    individual-resolution behaviour, and its residual-depression bug,
    for just those leaves. This test exists so a future fix for this
    specific gap has a concrete, reproducible case to check against -
    and so this limitation stays visible in the test suite rather than
    being forgotten. If this test ever starts failing because the
    residual count dropped to zero, that is GOOD NEWS: update this
    test to assert zero and remove this docstring's caveat, don't
    just delete the test."""
    from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
        compute_depression_metrics,
    )
    from mayim_tools.hydrology.dem_hydro_processing.core.hierarchical_classification import (
        group_into_resolution_units,
    )
    from mayim_tools.hydrology.dem_hydro_processing.core.impact_reduction import (
        apply_ira_to_all_leaves,
    )

    nd = -9999.0
    dem = _chain_dem(10, nd=nd)
    h = build_depression_hierarchy(dem, nodata=nd)
    leaf_ids = sorted(
        nid for nid, n in h.nodes.items() if not n.is_ocean and not n.children
    )
    excluded = leaf_ids[5]
    resolve_subset = [l for l in leaf_ids if l != excluded]

    units = group_into_resolution_units(h, resolve_subset)
    corrected, results = apply_ira_to_all_leaves(dem, h, nodata=nd, leaf_ids=units)
    h2 = build_depression_hierarchy(corrected, nodata=nd)
    residual = [
        nid
        for nid, n in h2.nodes.items()
        if not n.is_ocean
        and not n.children
        and compute_depression_metrics(corrected, h2, nid).volume > 0
    ]
    assert len(residual) == 1, (
        f"expected the known, documented 1-residual limitation for this exact scattered-exclusion "
        f"case, got {len(residual)} - if this is now 0, the limitation has been fixed: update this "
        f"test (see its own docstring)"
    )
    print(
        "test_KNOWN_LIMITATION_scattered_exclusion_mid_chain_can_still_leave_a_residual: PASS"
    )


if __name__ == "__main__":
    tests = [
        test_fragmented_noisy_lake_is_recognised_and_protected_at_meta_level,
        test_isolated_tiny_artifact_with_no_natural_ancestor_is_resolved,
        test_two_pit_natural_meta_protects_both_children,
        test_all_classifications_includes_leaves_and_metas,
        test_uncertain_meta_does_not_block_descent_into_children,
        test_reproducible_with_same_seed,
        test_zero_volume_leaves_excluded_from_classification,
        test_decide_resolution_from_labels_does_not_recurse_on_a_long_chain_hierarchy,
        test_group_into_resolution_units_fully_artificial_chain_leaves_zero_residuals,
        test_group_into_resolution_units_does_not_cross_an_unresolved_sibling,
        test_KNOWN_LIMITATION_scattered_exclusion_mid_chain_can_still_leave_a_residual,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
