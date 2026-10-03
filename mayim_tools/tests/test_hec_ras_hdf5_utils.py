"""Unit tests for core/hdf5_utils.py against a small synthetic HDF5
fixture built in-test with h5py - no QGIS, no real HEC-RAS project
needed. The fixture reproduces the Polyline Info/Parts/Points and
Attributes convention as documented (and confirmed for Reference Lines)
in claude/hec_ras_automation_plugin_research.md.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras._common.hdf5_utils import (  # noqa: E402
    read_attribute_table,
    read_point_features,
    read_polyline_features,
)


@pytest.fixture
def polyline_fixture(tmp_path):
    """Two features: a single-part 3-point line ('Line A') and a
    two-part line ('Line B', simulating a split/disjoint reference
    line)."""
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Reference Lines")

        attrs_dtype = np.dtype([("Name", "S32"), ("SHG_ID", "i4")])
        attrs = np.array(
            [("Line A", 1), ("Line B", 2)],
            dtype=attrs_dtype,
        )
        group.create_dataset("Attributes", data=attrs)

        # Line A: 1 part, 3 points (part index 0)
        # Line B: 2 parts, points split across them (part indices 1, 2)
        info = np.array(
            [
                [0, 3, 0, 1],  # point_start, point_count, part_start, part_count
                [3, 5, 1, 2],
            ],
            dtype="i8",
        )
        group.create_dataset("Polyline Info", data=info)

        # A part's own point_start is RELATIVE to its feature's Polyline
        # Info point_start, not an absolute Polyline Points index - see
        # the 2026-09-23 bugfix note in hdf5_utils.read_polyline_features.
        # Line A's point_start is 0, so its part's relative/absolute
        # start coincide; Line B's point_start is 3, so its two parts'
        # relative starts (0, 2) are offsets from that 3, landing on
        # absolute points 3 and 5 respectively.
        parts = np.array(
            [
                [0, 3],  # Line A's only part: relative 0 -> absolute points 0..3
                [0, 2],  # Line B's first part: relative 0 -> absolute points 3..5
                [2, 3],  # Line B's second part: relative 2 -> absolute points 5..8
            ],
            dtype="i8",
        )
        group.create_dataset("Polyline Parts", data=parts)

        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],  # Line A
                [5.0, 5.0],
                [6.0, 5.0],  # Line B part 1
                [10.0, 10.0],
                [11.0, 10.0],
                [11.0, 11.0],  # Line B part 2
            ],
            dtype="f8",
        )
        group.create_dataset("Polyline Points", data=points)

    return path


@pytest.fixture
def point_fixture(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Reference Points")

        attrs_dtype = np.dtype([("Name", "S32")])
        attrs = np.array([("Point A",), ("Point B",)], dtype=attrs_dtype)
        group.create_dataset("Attributes", data=attrs)

        points = np.array([[2.0, 3.0], [4.0, 5.0]], dtype="f8")
        group.create_dataset("Points", data=points)

    return path


def test_read_attribute_table_decodes_bytes_to_str(polyline_fixture):
    with h5py.File(polyline_fixture, "r") as h5file:
        rows = read_attribute_table(h5file, "Geometry/Reference Lines/Attributes")

    assert rows == [
        {"Name": "Line A", "SHG_ID": 1},
        {"Name": "Line B", "SHG_ID": 2},
    ]


def test_read_attribute_table_missing_dataset_returns_empty(polyline_fixture):
    with h5py.File(polyline_fixture, "r") as h5file:
        rows = read_attribute_table(h5file, "Geometry/Does Not Exist")
    assert rows == []


def test_read_polyline_features_single_part(polyline_fixture):
    with h5py.File(polyline_fixture, "r") as h5file:
        features = read_polyline_features(h5file, "Geometry/Reference Lines")

    assert len(features) == 2

    line_a = features[0]
    assert line_a.name == "Line A"
    assert line_a.parts == [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]


def test_read_polyline_features_multi_part(polyline_fixture):
    with h5py.File(polyline_fixture, "r") as h5file:
        features = read_polyline_features(h5file, "Geometry/Reference Lines")

    line_b = features[1]
    assert line_b.name == "Line B"
    assert len(line_b.parts) == 2
    assert line_b.parts[0] == [(5.0, 5.0), (6.0, 5.0)]
    assert line_b.parts[1] == [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0)]


def test_read_polyline_features_missing_group_raises(polyline_fixture):
    with h5py.File(polyline_fixture, "r") as h5file:
        with pytest.raises(KeyError):
            read_polyline_features(h5file, "Geometry/Does Not Exist")


def test_read_polyline_features_bad_index_range_raises(tmp_path):
    """A Polyline Parts range that runs past Polyline Points should
    raise, not silently slice - this is the guard rail for catching a
    wrong column-order assumption against a real file."""
    path = tmp_path / "bad.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Reference Lines")
        group.create_dataset(
            "Attributes",
            data=np.array([("Bad",)], dtype=np.dtype([("Name", "S8")])),
        )
        group.create_dataset("Polyline Info", data=np.array([[0, 5, 0, 1]], dtype="i8"))
        group.create_dataset("Polyline Parts", data=np.array([[0, 5]], dtype="i8"))
        # Only 2 points, but the part above claims 5 starting at 0.
        group.create_dataset(
            "Polyline Points", data=np.array([[0.0, 0.0], [1.0, 1.0]], dtype="f8")
        )

    with h5py.File(path, "r") as h5file:
        with pytest.raises(ValueError):
            read_polyline_features(h5file, "Geometry/Reference Lines")


def test_read_point_features(point_fixture):
    with h5py.File(point_fixture, "r") as h5file:
        features = read_point_features(h5file, "Geometry/Reference Points")

    assert len(features) == 2
    assert features[0].name == "Point A"
    assert features[0].xy == (2.0, 3.0)
    assert features[1].name == "Point B"
    assert features[1].xy == (4.0, 5.0)
