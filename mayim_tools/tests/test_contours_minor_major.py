"""Tests of the minor / major contours for report layout maps (report
standard 3.2)."""

import numpy as np

from mayim_tools.hydrology._common.contours import (
    MAJOR_FACTOR,
    minor_major_contours,
)

GT = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)


def plane(relief_per_cell: float):
    y, x = np.mgrid[0:100, 0:100].astype(float)
    return 100.0 + relief_per_cell * x


def test_major_every_fifth_minor_and_no_duplicates():
    minor, major, warnings = minor_major_contours(plane(0.2), GT, 1.0)
    assert not warnings
    assert MAJOR_FACTOR == 5
    assert sorted({lvl for lvl, _ in major}) == [105.0, 110.0, 115.0]
    minor_levels = {lvl for lvl, _ in minor}
    assert 101.0 in minor_levels and 104.0 in minor_levels
    assert not minor_levels & {105.0, 110.0, 115.0}


def test_zero_interval_gives_no_contours():
    assert minor_major_contours(plane(0.2), GT, 0.0) == ([], [], [])


def test_dense_minor_falls_back_to_major_only():
    minor, major, warnings = minor_major_contours(plane(4.0), GT, 1.0)
    assert minor == [] and major
    assert "only the 5 m major contours" in warnings[0]


def test_too_dense_for_major_gives_none():
    minor, major, warnings = minor_major_contours(plane(40.0), GT, 1.0)
    assert minor == [] and major == []
    assert "larger minor contour interval" in warnings[0]
