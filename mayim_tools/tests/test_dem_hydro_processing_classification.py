"""
Tests for core/classification.py.

Run: python3 tests/test_classification.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
    DEM_SOURCE_VERTICAL_ERROR_M,
    LABEL_LIKELY_ARTIFICIAL,
    LABEL_LIKELY_NATURAL,
    LABEL_UNCERTAIN,
    combine_evidence,
    compute_depression_metrics,
)
from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)


def test_dem_source_table_has_expected_categories_and_ordering():
    expected_keys = {
        "survey",
        "lidar",
        "uav_photogrammetry",
        "copernicus_glo30",
        "srtm",
        "aster",
    }
    assert set(DEM_SOURCE_VERTICAL_ERROR_M) == expected_keys
    # accuracy should degrade in this order - survey best, aster worst
    assert (
        DEM_SOURCE_VERTICAL_ERROR_M["survey"]
        < DEM_SOURCE_VERTICAL_ERROR_M["lidar"]
        < DEM_SOURCE_VERTICAL_ERROR_M["copernicus_glo30"]
        < DEM_SOURCE_VERTICAL_ERROR_M["srtm"]
        < DEM_SOURCE_VERTICAL_ERROR_M["aster"]
    )
    print("test_dem_source_table_has_expected_categories_and_ordering: PASS")


def test_compute_depression_metrics_matches_hand_calculation():
    """3x3 single pit, depth 4.0, 1 cell - matches the same hand-
    calculated case used for impact_reduction.py's own fill-cost
    test."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    metrics = compute_depression_metrics(dem, h, leaf_id)
    assert metrics.num_cells == 1
    assert metrics.volume == 4.0
    assert metrics.max_depth == 4.0
    assert metrics.mean_depth == 4.0
    print("test_compute_depression_metrics_matches_hand_calculation: PASS")


def test_compute_depression_metrics_works_on_meta_depressions_now():
    """A real change of behaviour, not a bug: compute_depression_metrics
    now accepts meta-depressions as well as leaves - see this
    function's own updated docstring and core/hierarchical_classification.py
    for why (a real basin in noisy data has no single leaf that IS the
    basin; only the merged meta-depression has its true depth). An
    earlier version of this test asserted the OLD, leaf-only behaviour
    (a ValueError on a meta id) - replaced now that the capability is
    real and intentional, not something to keep forbidding."""
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
    metrics = compute_depression_metrics(dem, h, meta_id)
    # the meta covers both pits' full footprint up to its own spill (10.0):
    # 15 cells (the 5x5 block minus the 4 corners, per the hierarchy's own
    # earlier-verified structure), volume/depths measured from elevation 10.0
    assert metrics.num_cells == 15, f"got {metrics.num_cells}"
    assert (
        metrics.max_depth == 8.0
    ), f"got {metrics.max_depth}"  # 10.0 - 2.0, the deeper of the two original pits
    print("test_compute_depression_metrics_works_on_meta_depressions_now: PASS")


def test_compute_depression_metrics_still_raises_on_ocean():
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
    try:
        compute_depression_metrics(dem, h, 0)  # 0 is always the ocean node id
        assert False, "should have raised"
    except ValueError as e:
        assert "ocean" in str(e).lower()
    print("test_compute_depression_metrics_still_raises_on_ocean: PASS")


def test_high_persistence_large_depression_classified_natural():
    dem = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 1.0, 1.0, 1.0, 5.0],
            [5.0, 1.0, 1.0, 1.0, 5.0],
            [5.0, 1.0, 1.0, 1.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[2, 2])
    result = combine_evidence(
        dem, h, leaf_id, persistence_probability=0.95, vertical_error_std=0.1
    )
    assert (
        result.label == LABEL_LIKELY_NATURAL
    ), f"got {result.label}, confidence={result.confidence}"
    print("test_high_persistence_large_depression_classified_natural: PASS")


def test_low_persistence_tiny_depression_classified_artificial():
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.99, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    result = combine_evidence(
        dem, h, leaf_id, persistence_probability=0.05, vertical_error_std=0.1
    )
    assert (
        result.label == LABEL_LIKELY_ARTIFICIAL
    ), f"got {result.label}, confidence={result.confidence}"
    print("test_low_persistence_tiny_depression_classified_artificial: PASS")


def test_moderate_persistence_classified_uncertain():
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    result = combine_evidence(
        dem, h, leaf_id, persistence_probability=0.5, vertical_error_std=0.5
    )
    assert (
        result.label == LABEL_UNCERTAIN
    ), f"got {result.label}, confidence={result.confidence}"
    print("test_moderate_persistence_classified_uncertain: PASS")


def test_stream_crossing_true_pulls_toward_natural():
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    without = combine_evidence(
        dem, h, leaf_id, persistence_probability=0.5, vertical_error_std=0.5
    )
    with_stream = combine_evidence(
        dem,
        h,
        leaf_id,
        persistence_probability=0.5,
        vertical_error_std=0.5,
        stream_crossing=True,
    )
    assert (
        with_stream.confidence > without.confidence
    ), f"stream crossing should increase confidence: {with_stream.confidence} vs {without.confidence}"
    print("test_stream_crossing_true_pulls_toward_natural: PASS")


def test_stream_crossing_false_pulls_toward_artificial_mildly():
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    without = combine_evidence(
        dem, h, leaf_id, persistence_probability=0.5, vertical_error_std=0.5
    )
    no_stream = combine_evidence(
        dem,
        h,
        leaf_id,
        persistence_probability=0.5,
        vertical_error_std=0.5,
        stream_crossing=False,
    )
    assert no_stream.confidence < without.confidence
    # but the effect should be MILD relative to a confirmed crossing - absence of
    # evidence in a sparse dataset is weaker evidence than a confirmed crossing
    with_stream = combine_evidence(
        dem,
        h,
        leaf_id,
        persistence_probability=0.5,
        vertical_error_std=0.5,
        stream_crossing=True,
    )
    pull_down = without.confidence - no_stream.confidence
    pull_up = with_stream.confidence - without.confidence
    assert (
        pull_down < pull_up
    ), "absence-of-stream should pull less than a confirmed crossing pulls"
    print("test_stream_crossing_false_pulls_toward_artificial_mildly: PASS")


def test_confidence_always_in_valid_range():
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
        for stream in [None, True, False]:
            for verr in [0.01, 0.1, 1.0, 10.0]:
                result = combine_evidence(
                    dem,
                    h,
                    leaf_id,
                    persistence_probability=p,
                    vertical_error_std=verr,
                    stream_crossing=stream,
                )
                assert (
                    0.0 <= result.confidence <= 1.0
                ), f"p={p}, stream={stream}, verr={verr}: confidence={result.confidence} out of range"
    print("test_confidence_always_in_valid_range: PASS")


def test_size_adjustment_is_bounded():
    """The secondary signal must never be able to swing the score by
    more than its documented bound, regardless of how extreme the
    depression's own metrics are - confirming it can nudge but never
    dominate persistence."""
    # a huge, deep depression - the adjustment should still be capped
    dem = np.full((50, 50), 1000.0)
    dem[10:40, 10:40] = 0.001  # huge, extremely deep relative to a tiny assumed error
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[25, 25])
    result = combine_evidence(
        dem, h, leaf_id, persistence_probability=0.0, vertical_error_std=0.001
    )
    assert (
        result.size_adjustment <= 0.15 + 1e-9
    ), f"size adjustment {result.size_adjustment} exceeded documented bound"
    print("test_size_adjustment_is_bounded: PASS")


def test_custom_thresholds_respected():
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    # with very wide "uncertain" bounds, a moderate-confidence result should land there
    result = combine_evidence(
        dem,
        h,
        leaf_id,
        persistence_probability=0.5,
        vertical_error_std=0.5,
        confidence_thresholds=(0.05, 0.95),
    )
    assert result.label == LABEL_UNCERTAIN
    # with very narrow bounds, the same confidence should now land outside them
    result2 = combine_evidence(
        dem,
        h,
        leaf_id,
        persistence_probability=0.5,
        vertical_error_std=0.5,
        confidence_thresholds=(0.49, 0.51),
    )
    assert result2.label != LABEL_UNCERTAIN or abs(result2.confidence - 0.5) < 0.01
    print("test_custom_thresholds_respected: PASS")


def test_passes_through_pulls_toward_artificial_terminates_in_toward_natural():
    """The two stream relations must pull in OPPOSITE directions: a
    stream ending in a depression suggests a genuine waterbody; a
    stream flowing straight across a supposedly closed basin suggests
    an uncaptured culvert/bridge. An earlier version treated any
    stream contact as 'natural', which would have classified exactly
    the culvert case backwards."""
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    kw = dict(persistence_probability=0.7, vertical_error_std=0.3)
    none = combine_evidence(dem, h, leaf_id, stream_relation="none", **kw)
    ends = combine_evidence(dem, h, leaf_id, stream_relation="terminates_in", **kw)
    through = combine_evidence(dem, h, leaf_id, stream_relation="passes_through", **kw)
    unset = combine_evidence(dem, h, leaf_id, **kw)
    assert through.confidence < none.confidence < ends.confidence
    assert through.confidence < unset.confidence < ends.confidence
    assert through.stream_relation == "passes_through"
    print(
        "test_passes_through_pulls_toward_artificial_terminates_in_toward_natural: PASS"
    )


def test_legacy_boolean_stream_crossing_still_means_terminates_in():
    dem = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[1, 1])
    kw = dict(persistence_probability=0.5, vertical_error_std=0.5)
    a = combine_evidence(dem, h, leaf_id, stream_crossing=True, **kw)
    b = combine_evidence(dem, h, leaf_id, stream_relation="terminates_in", **kw)
    assert a.confidence == b.confidence
    print("test_legacy_boolean_stream_crossing_still_means_terminates_in: PASS")


def test_metrics_count_the_footprint_not_the_catchment():
    """num_cells must be the cells actually below the spill elevation.
    On this terrain the leaf's catchment (cells_of) is several times
    larger than the 9-cell pond."""
    dem = np.full((15, 27), 20.0)
    for c in range(1, 27):
        if c <= 11:
            dem[7, c] = 3.0 + 0.8 * (11 - c)
        elif c >= 14:
            dem[7, c] = 2.6 - 0.05 * (c - 14)
    dem[7, 12:14] = 10.0
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[7, 6])
    m = compute_depression_metrics(dem, h, leaf_id)
    assert m.num_cells == 9, m
    assert (
        m.mean_depth > 0.5 * m.max_depth * 0.5
    )  # mean taken over pond cells only, not diluted by zero-depth catchment cells
    print("test_metrics_count_the_footprint_not_the_catchment: PASS")


if __name__ == "__main__":
    tests = [
        test_dem_source_table_has_expected_categories_and_ordering,
        test_compute_depression_metrics_matches_hand_calculation,
        test_compute_depression_metrics_works_on_meta_depressions_now,
        test_compute_depression_metrics_still_raises_on_ocean,
        test_high_persistence_large_depression_classified_natural,
        test_low_persistence_tiny_depression_classified_artificial,
        test_moderate_persistence_classified_uncertain,
        test_stream_crossing_true_pulls_toward_natural,
        test_stream_crossing_false_pulls_toward_artificial_mildly,
        test_confidence_always_in_valid_range,
        test_size_adjustment_is_bounded,
        test_custom_thresholds_respected,
        test_passes_through_pulls_toward_artificial_terminates_in_toward_natural,
        test_legacy_boolean_stream_crossing_still_means_terminates_in,
        test_metrics_count_the_footprint_not_the_catchment,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
