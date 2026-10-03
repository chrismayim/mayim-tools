"""Unit tests for geometry/mesh_2d/core.py against synthetic HDF5
fixtures - no QGIS, no real HEC-RAS project needed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.mesh_2d.core import (  # noqa: E402
    list_2d_flow_area_names,
    read_2d_flow_area_cells,
    read_2d_flow_area_perimeter,
    read_break_lines,
    read_refinement_regions,
)


@pytest.fixture
def geometry_hdf(tmp_path):
    """One flow area ('Perimeter 1') with 2 cells: a quad (4 valid
    facepoints, 4 padding slots) and a triangle (3 valid, 5 padding),
    plus one breakline."""
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        bl = f.create_group("Geometry/2D Flow Area Break Lines")
        bl.create_dataset(
            "Attributes", data=np.array([("BL1",)], dtype=np.dtype([("Name", "S8")]))
        )
        bl.create_dataset("Polyline Info", data=np.array([[0, 2, 0, 1]], dtype="i4"))
        bl.create_dataset("Polyline Parts", data=np.array([[0, 2]], dtype="i4"))
        bl.create_dataset(
            "Polyline Points", data=np.array([[0.0, 0.0], [1.0, 1.0]], dtype="f8")
        )

        flow_areas_group = f.create_group("Geometry/2D Flow Areas")
        # A sibling DATASET directly under Geometry/2D Flow Areas, same
        # as the real schema (Attributes/Cell Info/Cell Points/Polygon
        # Info/Parts/Points) - list_2d_flow_area_names must filter
        # these out and only return actual per-area Group children.
        flow_areas_group.create_dataset(
            "Attributes", data=np.array([(1,)], dtype=np.dtype([("Cell Count", "i4")]))
        )

        area = flow_areas_group.create_group("Perimeter 1")
        area.create_dataset(
            "Perimeter",
            data=np.array(
                [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
                dtype="f8",
            ),
        )
        area.create_dataset(
            "Cells Center Coordinate",
            data=np.array([[1.0, 1.0], [5.0, 5.0]], dtype="f8"),
        )
        # Cell 0: quad using facepoints 0,1,2,3; Cell 1: triangle using 3,4,5
        area.create_dataset(
            "Cells FacePoint Indexes",
            data=np.array(
                [
                    [0, 1, 2, 3, -1, -1, -1, -1],
                    [3, 4, 5, -1, -1, -1, -1, -1],
                ],
                dtype="i4",
            ),
        )
        area.create_dataset(
            "FacePoints Coordinate",
            data=np.array(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [2.0, 2.0],
                    [0.0, 2.0],
                    [4.0, 4.0],
                    [4.0, 6.0],
                ],
                dtype="f8",
            ),
        )

        rr = f.create_group("Geometry/2D Flow Area Refinement Regions")
        rr_attrs_dtype = np.dtype([("Name", "S16")])
        rr.create_dataset(
            "Attributes", data=np.array([("Region 1",)], dtype=rr_attrs_dtype)
        )
        rr.create_dataset("Polygon Info", data=np.array([[0, 5, 0, 1]], dtype="i4"))
        rr.create_dataset("Polygon Parts", data=np.array([[0, 5]], dtype="i4"))
        rr.create_dataset(
            "Polygon Points",
            data=np.array(
                [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]],
                dtype="f8",
            ),
        )
    return path


def test_list_2d_flow_area_names(geometry_hdf):
    assert list_2d_flow_area_names(str(geometry_hdf)) == ["Perimeter 1"]


def test_read_break_lines(geometry_hdf):
    lines = read_break_lines(str(geometry_hdf))
    assert len(lines) == 1
    assert lines[0].name == "BL1"


def test_read_2d_flow_area_perimeter(geometry_hdf):
    ring = read_2d_flow_area_perimeter(str(geometry_hdf), "Perimeter 1")
    assert ring[0] == (0.0, 0.0)
    assert ring[-1] == (0.0, 0.0)
    assert len(ring) == 5


def test_read_2d_flow_area_perimeter_missing_area_raises(geometry_hdf):
    with pytest.raises(KeyError):
        read_2d_flow_area_perimeter(str(geometry_hdf), "Does Not Exist")


def test_read_2d_flow_area_cells_drops_padding_and_closes_ring(geometry_hdf):
    cells = read_2d_flow_area_cells(str(geometry_hdf), "Perimeter 1")
    assert len(cells) == 2

    quad = cells[0]
    assert quad.center == (1.0, 1.0)
    assert len(quad.polygon) == 5  # 4 distinct vertices + closing point
    assert quad.polygon[0] == quad.polygon[-1]
    assert quad.polygon[:4] == [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    triangle = cells[1]
    assert triangle.center == (5.0, 5.0)
    assert len(triangle.polygon) == 4  # 3 distinct vertices + closing point
    assert triangle.polygon[:3] == [(0.0, 2.0), (4.0, 4.0), (4.0, 6.0)]


def test_read_refinement_regions(geometry_hdf):
    regions = read_refinement_regions(str(geometry_hdf))
    assert len(regions) == 1
    assert regions[0].name == "Region 1"
    assert len(regions[0].parts[0]) == 5
    assert regions[0].parts[0][0] == regions[0].parts[0][-1]


def test_read_refinement_regions_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_rr.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_refinement_regions(str(path)) == []
