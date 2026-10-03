"""Unit tests for geometry/reference_areas/core.py against a synthetic
HDF5 fixture - no QGIS, no real HEC-RAS project needed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.reference_areas.core import (  # noqa: E402
    read_reference_areas,
)


@pytest.fixture
def geometry_hdf_with_reference_area(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Reference Areas")
        attrs_dtype = np.dtype([("Name", "S32"), ("SA-2D", "S16")])
        attrs = np.array([("Reference Area 1", "")], dtype=attrs_dtype)
        group.create_dataset("Attributes", data=attrs)
        group.create_dataset("Polygon Info", data=np.array([[0, 5, 0, 1]], dtype="i4"))
        group.create_dataset("Polygon Parts", data=np.array([[0, 5]], dtype="i4"))
        group.create_dataset(
            "Polygon Points",
            data=np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
                dtype="f8",
            ),
        )
    return path


def test_read_reference_areas(geometry_hdf_with_reference_area):
    areas = read_reference_areas(str(geometry_hdf_with_reference_area))
    assert len(areas) == 1
    assert areas[0].name == "Reference Area 1"
    assert len(areas[0].parts) == 1
    assert len(areas[0].parts[0]) == 5
    assert areas[0].parts[0][0] == areas[0].parts[0][-1]  # closed ring


def test_read_reference_areas_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_areas.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_reference_areas(str(path)) == []
