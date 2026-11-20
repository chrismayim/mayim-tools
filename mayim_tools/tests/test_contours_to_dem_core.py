"""Contours to DEM: heights along stream and ridge lines from their contour
crossings, the contour interval and the flat-stretch check (no QGIS)."""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.hydrology.contours_to_dem import core

INTERVAL = 2.0


LEVELS = np.arange(96, 130, 2.0)


def _valley_contours(levels=LEVELS):
    """V-shaped valley along +y: z = 100 + 0.01 y + 0.1 |x|."""
    out = []
    ys = np.linspace(-200, 1200, 281)
    for z in levels:
        d = (z - 100 - 0.01 * ys) / 0.1
        ok = d >= 0
        if ok.sum() < 2:
            continue
        # open end - apex - open end
        xy = np.vstack([np.c_[-d[ok], ys[ok]], np.c_[d[ok], ys[ok]][::-1]])
        out.append((xy, float(z)))
    return out


def test_segment_crossings_and_shared_vertex():
    line = [(0, 0), (10, 0), (20, 0)]
    assert core.segment_crossings(line, [(5, -1), (5, 1)]) == [(5.0, 5.0, 0.0)]
    # crossing exactly at the line's middle vertex is found once
    assert len(core.segment_crossings(line, [(10, -1), (10, 1)])) == 1
    assert core.segment_crossings(line, [(30, -1), (30, 1)]) == []


def test_contour_interval():
    assert core.contour_interval([100, 102, 104, 104, 110, 112]) == 2.0
    with pytest.raises(core.ContourError):
        core.contour_interval([100, 100])


def test_stream_heights_follow_the_valley_floor():
    contours = _valley_contours()
    line = core.line_heights("s", [(0, 1000), (0, 0)], contours, INTERVAL, 5.0)
    assert line.warnings == []
    assert [c.z for c in line.crossings] == [110, 108, 106, 104, 102, 100]
    truth = 100 + 0.01 * line.xyz[:, 1]
    np.testing.assert_allclose(line.xyz[:, 2], truth, atol=1e-9)
    # every crossing is a vertex of the line, at the contour's level
    for c in line.crossings:
        k = np.argmin(np.abs(line.chainage - c.chainage))
        assert line.chainage[k] == pytest.approx(c.chainage)
        assert line.xyz[k, 2] == pytest.approx(c.z)


def test_ends_carry_the_gradient_within_one_interval():
    contours = _valley_contours()
    # runs 600 m below the 100 m contour: the gradient would give 94 m,
    # but it crosses no further contour, so it stays above 98 m
    line = core.line_heights("s", [(0, 800), (0, -600)], contours, INTERVAL, 10.0)
    assert line.xyz[-1, 2] == pytest.approx(100 - 0.999 * INTERVAL)
    assert line.xyz[-1, 2] > 98.0
    # just past the last crossing the gradient carries on
    k = np.searchsorted(line.chainage, line.crossings[-1].chainage + 50)
    assert line.xyz[k, 2] == pytest.approx(99.5, abs=0.01)


def test_line_drawn_upstream_is_reversed_for_the_report():
    contours = _valley_contours()
    up = core.line_heights("s", [(0, 0), (0, 1000)], contours, INTERVAL, 5.0)
    assert up.xyz[0, 2] < up.xyz[-1, 2]
    down = core.oriented_downstream(up)
    assert down.xyz[0, 2] > down.xyz[-1, 2]
    assert down.crossings[0].z == 110 and down.crossings[0].chainage < 50
    np.testing.assert_allclose(down.chainage, np.sort(down.chainage))


def test_warnings():
    contours = _valley_contours()
    none = core.line_heights("a", [(500, 0), (500, 5)], contours, INTERVAL, 1.0)
    assert not none.usable and "crosses no contour" in none.warnings[0]
    one = core.line_heights("b", [(0, 150), (0, 250)], contours, INTERVAL, 1.0)
    assert one.usable and "one contour only" in one.warnings[0]
    assert np.all(one.xyz[:, 2] == 102.0)
    # a stream that climbs out over the valley side goes up and down
    zig = core.line_heights(
        "c", [(-40, 300), (0, 300), (0, 600), (60, 600)], contours, INTERVAL, 5.0
    )
    assert any("both up and down" in w for w in zig.warnings)
    # a gap: the 104 m contour is missing
    gap = [c for c in contours if c[1] != 104]
    skip = core.line_heights("d", [(0, 1000), (0, 0)], gap, INTERVAL, 5.0)
    assert any("skips a contour" in w for w in skip.warnings)


def test_ridge_lines_are_not_checked_for_direction():
    contours = _valley_contours()
    ridge = core.line_heights(
        "r", [(-40, 300), (0, 300), (0, 600), (60, 600)], contours, INTERVAL, 5.0,
        kind="ridge",
    )  # fmt: skip
    assert not any("both up and down" in w for w in ridge.warnings)


def test_flat_stretches():
    ch = np.arange(0.0, 101.0)
    stair = np.where(ch < 60, 110.0, 108.0)
    fc = core.flat_stretches(ch, stair, INTERVAL)
    assert fc.longest_flat_m == pytest.approx(59.0)
    assert fc.flat_share == pytest.approx(0.99)
    slope = 110.0 - 0.02 * ch
    assert core.flat_stretches(ch, slope, INTERVAL).longest_flat_m == 0.0
    # descending chainage and missing cells
    z = slope.copy()
    z[10:20] = np.nan
    fc = core.flat_stretches(ch[::-1], z[::-1], INTERVAL)
    assert fc.longest_flat_m == 0.0 and fc.sampled_m == pytest.approx(89.0)


def test_profile_samples():
    contours = _valley_contours()
    line = core.line_heights("s", [(0, 1000), (0, 0)], contours, INTERVAL, 5.0)
    ch, x, y = core.profile_samples(line, 1.0)
    assert ch[0] == 0 and ch[-1] == pytest.approx(1000)
    np.testing.assert_allclose(x, 0, atol=1e-9)
    np.testing.assert_allclose(y, 1000 - ch)
