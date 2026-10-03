"""Unit tests for geometry/reference_lines_points/core.py against a
synthetic HDF5 fixture - no QGIS, no real HEC-RAS project needed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.reference_lines_points.core import (  # noqa: E402
    read_reference_lines,
    read_reference_points,
)


@pytest.fixture
def geometry_hdf_with_reference_lines(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Reference Lines")
        attrs = np.array([("RL_Downstream",)], dtype=np.dtype([("Name", "S32")]))
        group.create_dataset("Attributes", data=attrs)
        group.create_dataset("Polyline Info", data=np.array([[0, 2, 0, 1]], dtype="i8"))
        group.create_dataset("Polyline Parts", data=np.array([[0, 2]], dtype="i8"))
        group.create_dataset(
            "Polyline Points",
            data=np.array([[100.0, 200.0], [101.0, 201.0]], dtype="f8"),
        )
    return path


@pytest.fixture
def geometry_hdf_with_reference_points(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Reference Points")
        attrs_dtype = np.dtype(
            [("Name", "S32"), ("SA/2D", "S16"), ("Cell Index", "i4")]
        )
        attrs = np.array([("Reference Point 1", "Perimeter 1", 888)], dtype=attrs_dtype)
        group.create_dataset("Attributes", data=attrs)
        group.create_dataset("Points", data=np.array([[300.0, 400.0]], dtype="f8"))
    return path


@pytest.fixture
def geometry_hdf_without_reference_lines(tmp_path):
    """A geometry file that simply has no Reference Lines/Points
    defined - the normal case for most models, must not error."""
    path = tmp_path / "geometry_no_refs.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    return path


def test_read_reference_lines(geometry_hdf_with_reference_lines):
    lines = read_reference_lines(str(geometry_hdf_with_reference_lines))
    assert len(lines) == 1
    assert lines[0].name == "RL_Downstream"
    assert lines[0].parts == [[(100.0, 200.0), (101.0, 201.0)]]


def test_read_reference_lines_absent_group_returns_empty(
    geometry_hdf_without_reference_lines,
):
    lines = read_reference_lines(str(geometry_hdf_without_reference_lines))
    assert lines == []


def test_read_reference_points(geometry_hdf_with_reference_points):
    points = read_reference_points(str(geometry_hdf_with_reference_points))
    assert len(points) == 1
    assert points[0].name == "Reference Point 1"
    assert points[0].xy == (300.0, 400.0)
    assert points[0].attributes["Cell Index"] == 888


def test_read_reference_points_absent_group_returns_empty(
    geometry_hdf_without_reference_lines,
):
    points = read_reference_points(str(geometry_hdf_without_reference_lines))
    assert points == []
