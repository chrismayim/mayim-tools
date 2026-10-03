"""Unit tests for geometry/initial_condition_points/core.py against a
synthetic HDF5 fixture - no QGIS, no real HEC-RAS project needed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.initial_condition_points.core import (  # noqa: E402
    read_initial_condition_points,
)


@pytest.fixture
def geometry_hdf_with_ic_point(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/IC Points")
        attrs_dtype = np.dtype(
            [("Name", "S32"), ("SA/2D", "S16"), ("Cell Index", "i4")]
        )
        attrs = np.array([("IC Points 1", "Perimeter 1", 878)], dtype=attrs_dtype)
        group.create_dataset("Attributes", data=attrs)
        group.create_dataset("Points", data=np.array([[10.0, 20.0]], dtype="f8"))
    return path


def test_read_initial_condition_points(geometry_hdf_with_ic_point):
    points = read_initial_condition_points(str(geometry_hdf_with_ic_point))
    assert len(points) == 1
    assert points[0].name == "IC Points 1"
    assert points[0].xy == (10.0, 20.0)
    assert points[0].attributes["Cell Index"] == 878


def test_read_initial_condition_points_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_ic.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_initial_condition_points(str(path)) == []
