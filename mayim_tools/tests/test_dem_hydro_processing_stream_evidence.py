"""
Tests for core/stream_evidence.py.

Coordinate/rasterization values below were verified directly against
the actual implementation before being written into assertions (not
hand-predicted blind) - deliberately avoiding exact .5 pixel-boundary
values, which hit Python's round()-to-even ("banker's rounding")
behaviour and would make the expected cell ambiguous for reasons
unrelated to the rasterizer's own correctness.

Run: python3 tests/test_stream_evidence.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)
from mayim_tools.hydrology.dem_hydro_processing.core.stream_evidence import (
    RasterTransform,
    compute_stream_crossing,
    compute_stream_relation,
    rasterize_polylines,
)


def test_world_to_pixel_simple_transform():
    t = RasterTransform(origin_x=0.0, origin_y=100.0, pixel_width=1.0, pixel_height=1.0)
    row, col = t.world_to_pixel(5.5, 94.5)
    assert abs(row - 5.5) < 1e-9 and abs(col - 5.5) < 1e-9
    print("test_world_to_pixel_simple_transform: PASS")


def test_world_to_pixel_non_unit_cell_size():
    t = RasterTransform(
        origin_x=100.0, origin_y=500.0, pixel_width=2.0, pixel_height=2.0
    )
    row, col = t.world_to_pixel(110.0, 480.0)
    # col = (110-100)/2 = 5.0, row = (500-480)/2 = 10.0
    assert abs(row - 10.0) < 1e-9 and abs(col - 5.0) < 1e-9
    print("test_world_to_pixel_non_unit_cell_size: PASS")


def test_rasterize_horizontal_line():
    t = RasterTransform(origin_x=0.0, origin_y=100.0, pixel_width=1.0, pixel_height=1.0)
    mask = rasterize_polylines([[(0.4, 94.6), (9.4, 94.6)]], t, (10, 10))
    expected = {(5, c) for c in range(10)}
    actual = {tuple(cell) for cell in np.argwhere(mask)}
    assert actual == expected, f"got {actual}"
    print("test_rasterize_horizontal_line: PASS")


def test_rasterize_vertical_line():
    t = RasterTransform(origin_x=0.0, origin_y=100.0, pixel_width=1.0, pixel_height=1.0)
    mask = rasterize_polylines([[(3.6, 99.4), (3.6, 90.4)]], t, (10, 10))
    expected = {(r, 4) for r in range(1, 10)}
    actual = {tuple(cell) for cell in np.argwhere(mask)}
    assert actual == expected, f"got {actual}"
    print("test_rasterize_vertical_line: PASS")


def test_rasterize_out_of_bounds_vertices_clipped_not_crashed():
    """A line running from well inside the raster to far outside it
    must not crash, and must only mark the cells that are actually
    within bounds - the ordinary case of a stream continuing past a
    DEM's own extent."""
    t = RasterTransform(origin_x=0.0, origin_y=100.0, pixel_width=1.0, pixel_height=1.0)
    mask = rasterize_polylines([[(3.6, 94.6), (300.6, 94.6)]], t, (10, 10))
    actual = {tuple(cell) for cell in np.argwhere(mask)}
    assert actual, "expected at least the in-bounds portion to be rasterized"
    assert all(
        0 <= c < 10 for _, c in actual
    ), "no out-of-bounds cell should have been marked"
    print("test_rasterize_out_of_bounds_vertices_clipped_not_crashed: PASS")


def test_rasterize_multiple_polylines_combined():
    t = RasterTransform(origin_x=0.0, origin_y=100.0, pixel_width=1.0, pixel_height=1.0)
    mask = rasterize_polylines(
        [
            [(0.4, 94.6), (2.4, 94.6)],
            [(0.4, 90.6), (2.4, 90.6)],
        ],  # two separate short lines
        t,
        (10, 10),
    )
    assert mask[5, 0] and mask[5, 1] and mask[5, 2]
    assert mask[9, 0] and mask[9, 1] and mask[9, 2]
    assert mask.sum() == 6
    print("test_rasterize_multiple_polylines_combined: PASS")


def test_rasterize_skips_degenerate_single_vertex_polyline():
    t = RasterTransform(origin_x=0.0, origin_y=100.0, pixel_width=1.0, pixel_height=1.0)
    mask = rasterize_polylines(
        [[(3.0, 94.0)]], t, (10, 10)
    )  # single vertex - not a real line
    assert not mask.any()
    print("test_rasterize_skips_degenerate_single_vertex_polyline: PASS")


def test_stream_crossing_true_when_stream_intersects_depression():
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
    stream_mask = np.zeros_like(dem, dtype=bool)
    stream_mask[2, 2] = True  # directly on the depression's own cell
    assert compute_stream_crossing(h, leaf_id, stream_mask, buffer_cells=0)
    print("test_stream_crossing_true_when_stream_intersects_depression: PASS")


def test_stream_crossing_false_when_stream_far_away():
    dem = np.full((15, 15), 5.0)
    dem[6:9, 6:9] = 1.0
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[7, 7])
    stream_mask = np.zeros_like(dem, dtype=bool)
    stream_mask[0, 0] = (
        True  # genuinely far from the depression (rows/cols 6-8), well beyond any reasonable buffer
    )
    assert not compute_stream_crossing(h, leaf_id, stream_mask, buffer_cells=1)
    print("test_stream_crossing_false_when_stream_far_away: PASS")


def test_stream_crossing_buffer_catches_nearby_stream():
    """A stream just OUTSIDE the depression's own cells, but within
    the buffer distance, should still count as a crossing - a mapped
    centreline is a simplification of the real channel's position."""
    dem = np.array(
        [
            [9.0, 9.0, 9.0, 9.0, 9.0],
            [9.0, 5.0, 5.0, 5.0, 9.0],
            [9.0, 5.0, 1.0, 5.0, 9.0],
            [9.0, 5.0, 5.0, 5.0, 9.0],
            [9.0, 9.0, 9.0, 9.0, 9.0],
        ]
    )
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[2, 2])
    stream_mask = np.zeros_like(dem, dtype=bool)
    stream_mask[0, 0] = (
        True  # just outside the depression's own extent (which spans rows/cols 1-3)
    )
    without_buffer = compute_stream_crossing(h, leaf_id, stream_mask, buffer_cells=0)
    with_buffer = compute_stream_crossing(h, leaf_id, stream_mask, buffer_cells=2)
    assert not without_buffer, "the stream cell itself is not part of the depression"
    assert with_buffer, "a 2-cell buffer should reach the nearby stream cell"
    print("test_stream_crossing_buffer_catches_nearby_stream: PASS")


def test_stream_crossing_raises_on_non_leaf():
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
    stream_mask = np.zeros_like(dem, dtype=bool)
    try:
        compute_stream_crossing(h, meta_id, stream_mask)
        assert False, "should have raised"
    except ValueError as e:
        assert "LEAF" in str(e)
    print("test_stream_crossing_raises_on_non_leaf: PASS")


def test_end_to_end_stream_evidence_feeds_classification():
    """Confirms the full pipeline actually connects: build the
    hierarchy, rasterize a stream, compute crossing evidence, and feed
    it into classification.py's own combine_evidence() - the whole
    point of this module existing. A stream crossing a moderate-
    persistence depression should noticeably raise its confidence
    relative to the same depression with no stream evidence."""
    from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
        combine_evidence,
    )

    dem = np.full((15, 15), 5.0)
    dem[6:9, 6:9] = 4.5  # a shallow, moderate depression
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[7, 7])

    t = RasterTransform(origin_x=0.0, origin_y=15.0, pixel_width=1.0, pixel_height=1.0)
    # a stream running straight through the depression's own location
    stream_mask = rasterize_polylines([[(0.5, 7.6), (14.5, 7.6)]], t, dem.shape)
    crossing = compute_stream_crossing(h, leaf_id, stream_mask, buffer_cells=1)
    assert crossing, "the stream should be detected as crossing this depression"

    without_stream = combine_evidence(
        dem, h, leaf_id, persistence_probability=0.5, vertical_error_std=0.3
    )
    with_stream = combine_evidence(
        dem,
        h,
        leaf_id,
        persistence_probability=0.5,
        vertical_error_std=0.3,
        stream_crossing=crossing,
    )
    assert with_stream.confidence > without_stream.confidence
    print("test_end_to_end_stream_evidence_feeds_classification: PASS")


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


def _stream(shape, col_from, col_to, row=7):
    m = np.zeros(shape, dtype=bool)
    m[row, col_from : col_to + 1] = True
    return m


def test_relation_none_when_no_stream_near():
    dem = _valley_dem()
    h = build_depression_hierarchy(dem)
    leaf = int(h.leaf_label[7, 6])
    stream = np.zeros(dem.shape, dtype=bool)
    stream[1, 20:25] = True
    assert compute_stream_relation(h, leaf, stream, dem=dem) == "none"
    print("test_relation_none_when_no_stream_near: PASS")


def test_relation_terminates_in_when_stream_ends_in_depression():
    dem = _valley_dem()
    h = build_depression_hierarchy(dem)
    leaf = int(h.leaf_label[7, 6])
    stream = _stream(dem.shape, 1, 8)  # arrives from upstream and stops inside the pond
    assert compute_stream_relation(h, leaf, stream, dem=dem) == "terminates_in"
    print("test_relation_terminates_in_when_stream_ends_in_depression: PASS")


def test_relation_passes_through_when_stream_continues_on_both_sides():
    dem = _valley_dem()
    h = build_depression_hierarchy(dem)
    leaf = int(h.leaf_label[7, 6])
    stream = _stream(dem.shape, 1, 26)  # straight across the pond and the embankment
    assert compute_stream_relation(h, leaf, stream, dem=dem) == "passes_through"
    print("test_relation_passes_through_when_stream_continues_on_both_sides: PASS")


def test_hillside_stream_above_spill_is_not_inside_the_depression():
    """Regression test for the catchment/footprint bug: a stream on the
    wall ABOVE the spill elevation lies inside the leaf's catchment
    (cells_of) but is not in the depression, and must not count as
    being in it."""
    dem = _valley_dem()
    h = build_depression_hierarchy(dem)
    leaf = int(h.leaf_label[7, 6])
    stream = np.zeros(dem.shape, dtype=bool)
    stream[6, 3:12] = True  # one row up: a 20 m wall cell row, adjacent to the valley
    assert h.cells_of(leaf)[
        6, 5
    ], "precondition: that wall cell belongs to the leaf's catchment"
    assert compute_stream_relation(h, leaf, stream, buffer_cells=0, dem=dem) == "none"
    print("test_hillside_stream_above_spill_is_not_inside_the_depression: PASS")


if __name__ == "__main__":
    tests = [
        test_world_to_pixel_simple_transform,
        test_world_to_pixel_non_unit_cell_size,
        test_rasterize_horizontal_line,
        test_rasterize_vertical_line,
        test_rasterize_out_of_bounds_vertices_clipped_not_crashed,
        test_rasterize_multiple_polylines_combined,
        test_rasterize_skips_degenerate_single_vertex_polyline,
        test_stream_crossing_true_when_stream_intersects_depression,
        test_stream_crossing_false_when_stream_far_away,
        test_stream_crossing_buffer_catches_nearby_stream,
        test_stream_crossing_raises_on_non_leaf,
        test_end_to_end_stream_evidence_feeds_classification,
        test_relation_none_when_no_stream_near,
        test_relation_terminates_in_when_stream_ends_in_depression,
        test_relation_passes_through_when_stream_continues_on_both_sides,
        test_hillside_stream_above_spill_is_not_inside_the_depression,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
