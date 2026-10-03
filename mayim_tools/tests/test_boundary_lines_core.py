"""Unit tests for geometry/boundary_lines/core.py against a synthetic
HDF5 fixture - no QGIS, no real HEC-RAS project needed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.boundary_lines.core import (  # noqa: E402
    read_boundary_condition_lines,
)


@pytest.fixture
def geometry_hdf_with_bc_lines(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Boundary Condition Lines")
        attrs_dtype = np.dtype(
            [("Name", "S32"), ("SA-2D", "S16"), ("Type", "S8"), ("Length", "f4")]
        )
        attrs = np.array(
            [
                ("Outflow", "Perimeter 1", "External", 148.08582),
                ("Inflow", "Perimeter 1", "Internal", 12.397231),
            ],
            dtype=attrs_dtype,
        )
        group.create_dataset("Attributes", data=attrs)
        group.create_dataset(
            "Polyline Info", data=np.array([[0, 2, 0, 1], [2, 2, 1, 1]], dtype="i4")
        )
        # A part's own point_start is RELATIVE to its feature's Polyline
        # Info point_start (see hdf5_utils.read_polyline_features's
        # 2026-09-23 bugfix note) - both parts here start at relative 0.
        group.create_dataset(
            "Polyline Parts", data=np.array([[0, 2], [0, 2]], dtype="i4")
        )
        group.create_dataset(
            "Polyline Points",
            data=np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]], dtype="f8"),
        )
    return path


def test_read_boundary_condition_lines(geometry_hdf_with_bc_lines):
    lines = read_boundary_condition_lines(str(geometry_hdf_with_bc_lines))
    assert len(lines) == 2
    assert lines[0].name == "Outflow"
    assert lines[0].attributes["Type"] == "External"
    assert lines[1].name == "Inflow"
    assert lines[1].attributes["Type"] == "Internal"


def test_read_boundary_condition_lines_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_bc.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_boundary_condition_lines(str(path)) == []
