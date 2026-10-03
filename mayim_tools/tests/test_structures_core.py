"""Unit tests for geometry/structures/core.py against a synthetic HDF5
fixture - no QGIS, no real HEC-RAS project needed.

The fixture's field names/values mirror the real M2627_model.g02.hdf
structure confirmed in core.py's module docstring (one Bridge
connection under Geometry/Structures, 'Centerline Info/Parts/Points'
naming instead of 'Polyline Info/Parts/Points').
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.structures.core import (  # noqa: E402
    read_bridge_1d_structures,
    read_bridge_2d_structures,
    read_culvert_barrels,
    read_culvert_structures,
    read_linear_routing_structures,
    read_structures,
    read_weir_structures,
)


@pytest.fixture
def geometry_hdf_with_structures(tmp_path):
    """Four structures covering every classification branch: a 2D
    bridge opening, a 1D (traditional-method) bridge, a plain weir/
    gate/culvert connection with a culvert group (Weir + Culvert), and
    a Linear Routing connection - field names/values mirror the real
    M2627_model.g02.hdf structures confirmed in core.py's docstring."""
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Structures")
        attrs_dtype = np.dtype(
            [
                ("Type", "S16"),
                ("Mode", "S24"),
                ("Connection", "S16"),
                ("Groupname", "S32"),
                ("US SA/2D", "S16"),
                ("DS SA/2D", "S16"),
                ("Culvert Groups", "i4"),
                ("Gate Groups", "i4"),
            ]
        )
        attrs = np.array(
            [
                (
                    "Connection",
                    "Bridge Opening (2D)",
                    "Bridge",
                    "Perimeter 1, Bridge 2D",
                    "Perimeter 1",
                    "Perimeter 1",
                    0,
                    0,
                ),
                (
                    "Connection",
                    "Energy",
                    "Bridge",
                    "Perimeter 1, Bridge 1D",
                    "Perimeter 1",
                    "Perimeter 1",
                    0,
                    0,
                ),
                (
                    "Connection",
                    "Weir/Gate/Culverts",
                    "SA2D Conn 1",
                    "Perimeter 1, SA2D Conn 1",
                    "Perimeter 1",
                    "Perimeter 1",
                    1,
                    0,
                ),
                (
                    "Connection",
                    "Linear Routing",
                    "SA2D Conn 2",
                    "Perimeter 1, SA2D Conn 2",
                    "Perimeter 1",
                    "Perimeter 1",
                    0,
                    0,
                ),
            ],
            dtype=attrs_dtype,
        )
        group.create_dataset("Attributes", data=attrs)
        group.create_dataset(
            "Centerline Info",
            data=np.array(
                [[0, 2, 0, 1], [2, 2, 1, 1], [4, 2, 2, 1], [6, 2, 3, 1]], dtype="i4"
            ),
        )
        # A part's own point_start is RELATIVE to its feature's own
        # Centerline Info point_start (see hdf5_utils.
        # read_polyline_features's 2026-09-23 bugfix note) - each of
        # these 4 structures has exactly one part starting at its own
        # relative 0.
        group.create_dataset(
            "Centerline Parts",
            data=np.array([[0, 2], [0, 2], [0, 2], [0, 2]], dtype="i4"),
        )
        group.create_dataset(
            "Centerline Points",
            data=np.array(
                [
                    [-54564.55156675, -2861260.6521151],
                    [-54535.45838348, -2861277.5449312],
                    [-54600.0, -2861300.0],
                    [-54580.0, -2861310.0],
                    [-54700.0, -2861400.0],
                    [-54680.0, -2861410.0],
                    [-54800.0, -2861500.0],
                    [-54780.0, -2861510.0],
                ],
                dtype="f8",
            ),
        )
    return path


def test_read_structures(geometry_hdf_with_structures):
    structures = read_structures(str(geometry_hdf_with_structures))
    assert len(structures) == 4

    bridge_2d = structures[0]
    assert bridge_2d.attributes["Connection"] == "Bridge"
    assert bridge_2d.attributes["Mode"] == "Bridge Opening (2D)"
    assert bridge_2d.attributes["US SA/2D"] == "Perimeter 1"
    assert bridge_2d.attributes["DS SA/2D"] == "Perimeter 1"
    assert len(bridge_2d.parts) == 1
    assert len(bridge_2d.parts[0]) == 2


def test_read_structures_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_structures.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_structures(str(path)) == []


def test_read_bridge_2d_structures(geometry_hdf_with_structures):
    bridges = read_bridge_2d_structures(str(geometry_hdf_with_structures))
    assert len(bridges) == 1
    assert bridges[0].attributes["Groupname"] == "Perimeter 1, Bridge 2D"


def test_read_bridge_1d_structures(geometry_hdf_with_structures):
    bridges = read_bridge_1d_structures(str(geometry_hdf_with_structures))
    assert len(bridges) == 1
    assert bridges[0].attributes["Groupname"] == "Perimeter 1, Bridge 1D"
    assert bridges[0].attributes["Mode"] == "Energy"


def test_read_weir_structures_includes_culvert_bearing_connection(
    geometry_hdf_with_structures,
):
    weirs = read_weir_structures(str(geometry_hdf_with_structures))
    assert len(weirs) == 1
    assert weirs[0].attributes["Connection"] == "SA2D Conn 1"
    assert weirs[0].attributes["Culvert Groups"] == 1


def test_read_culvert_structures_is_subset_of_weir(geometry_hdf_with_structures):
    culverts = read_culvert_structures(str(geometry_hdf_with_structures))
    assert len(culverts) == 1
    assert culverts[0].attributes["Connection"] == "SA2D Conn 1"


def test_read_linear_routing_structures(geometry_hdf_with_structures):
    routing = read_linear_routing_structures(str(geometry_hdf_with_structures))
    assert len(routing) == 1
    assert routing[0].attributes["Connection"] == "SA2D Conn 2"
    assert routing[0].attributes["Mode"] == "Linear Routing"


@pytest.fixture
def geometry_hdf_with_culvert_barrels(tmp_path):
    """A single weir/culvert connection whose culvert group has 2
    barrels, each with its own centreline - added 2026-09-23 per
    Chris's correction that the 'culvert' output should be each
    barrel's centreline, not the parent connection's. The 2nd barrel's
    point_start is non-zero, deliberately exercising the
    2026-09-23 relative-vs-absolute Polyline/Centerline Parts indexing
    fix (see hdf5_utils.read_polyline_features)."""
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Structures")
        conn_attrs_dtype = np.dtype(
            [("Connection", "S16"), ("Mode", "S24"), ("Culvert Groups", "i4")]
        )
        group.create_dataset(
            "Attributes",
            data=np.array(
                [("SA2D Conn 1", "Weir/Gate/Culverts", 1)], dtype=conn_attrs_dtype
            ),
        )
        group.create_dataset(
            "Centerline Info", data=np.array([[0, 2, 0, 1]], dtype="i4")
        )
        group.create_dataset("Centerline Parts", data=np.array([[0, 2]], dtype="i4"))
        group.create_dataset(
            "Centerline Points",
            data=np.array([[0.0, 0.0], [1.0, 0.0]], dtype="f8"),
        )

        barrels = group.create_group("Culvert Groups/Barrels")
        barrel_attrs_dtype = np.dtype(
            [
                ("Structure ID", "i4"),
                ("Culvert Group ID", "i4"),
                ("Name", "S16"),
            ]
        )
        barrels.create_dataset(
            "Attributes",
            data=np.array(
                [(1, 0, "Barrel #1"), (1, 0, "Barrel #2")], dtype=barrel_attrs_dtype
            ),
        )
        barrels.create_dataset(
            "Centerline Info",
            data=np.array([[0, 2, 0, 1], [2, 2, 1, 1]], dtype="i4"),
        )
        # Both parts start at their own feature's relative 0 - matches
        # the real HEC-RAS convention confirmed in hdf5_utils.py.
        barrels.create_dataset(
            "Centerline Parts", data=np.array([[0, 2], [0, 2]], dtype="i4")
        )
        barrels.create_dataset(
            "Centerline Points",
            data=np.array(
                [[10.0, 10.0], [11.0, 10.0], [20.0, 20.0], [21.0, 20.0]], dtype="f8"
            ),
        )
    return path


def test_read_culvert_barrels(geometry_hdf_with_culvert_barrels):
    barrels = read_culvert_barrels(str(geometry_hdf_with_culvert_barrels))
    assert len(barrels) == 2
    assert barrels[0].name == "Barrel #1"
    assert barrels[0].parts == [[(10.0, 10.0), (11.0, 10.0)]]
    assert barrels[1].name == "Barrel #2"
    assert barrels[1].parts == [[(20.0, 20.0), (21.0, 20.0)]]
    # The barrels' own centreline must not be the parent connection's.
    connection_centreline = read_structures(str(geometry_hdf_with_culvert_barrels))[
        0
    ].parts
    assert barrels[0].parts != connection_centreline
    assert barrels[1].parts != connection_centreline


def test_read_culvert_barrels_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_barrels.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_culvert_barrels(str(path)) == []
