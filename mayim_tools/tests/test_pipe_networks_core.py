"""Unit tests for geometry/pipe_networks/core.py against a synthetic
HDF5 fixture - no QGIS, no real HEC-RAS project needed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.pipe_networks.core import (  # noqa: E402
    read_pipe_conduits,
    read_pipe_nodes,
)


@pytest.fixture
def geometry_hdf_with_pipe_network(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        nodes = f.create_group("Geometry/Pipe Nodes")
        node_attrs_dtype = np.dtype([("Name", "S32"), ("Node Type", "S16")])
        node_attrs = np.array(
            [("Node 1", "Junction"), ("Node 2", "Outfall")], dtype=node_attrs_dtype
        )
        nodes.create_dataset("Attributes", data=node_attrs)
        nodes.create_dataset(
            "Points", data=np.array([[0.0, 0.0], [10.0, 0.0]], dtype="f8")
        )

        conduits = f.create_group("Geometry/Pipe Conduits")
        conduit_attrs_dtype = np.dtype(
            [("Name", "S32"), ("US Node", "S32"), ("DS Node", "S32")]
        )
        conduit_attrs = np.array(
            [("Conduit 1", "Node 1", "Node 2")], dtype=conduit_attrs_dtype
        )
        conduits.create_dataset("Attributes", data=conduit_attrs)
        conduits.create_dataset(
            "Polyline Info", data=np.array([[0, 2, 0, 1]], dtype="i4")
        )
        conduits.create_dataset("Polyline Parts", data=np.array([[0, 2]], dtype="i4"))
        conduits.create_dataset(
            "Polyline Points", data=np.array([[0.0, 0.0], [10.0, 0.0]], dtype="f8")
        )

        # Network-level metadata only - no geometry, not expected to be read.
        networks = f.create_group("Geometry/Pipe Networks")
        networks.create_dataset(
            "Attributes", data=np.array([("Base",)], dtype=np.dtype([("Name", "S16")]))
        )
    return path


def test_read_pipe_nodes(geometry_hdf_with_pipe_network):
    nodes = read_pipe_nodes(str(geometry_hdf_with_pipe_network))
    assert len(nodes) == 2
    assert nodes[0].name == "Node 1"
    assert nodes[1].attributes["Node Type"] == "Outfall"


def test_read_pipe_conduits(geometry_hdf_with_pipe_network):
    conduits = read_pipe_conduits(str(geometry_hdf_with_pipe_network))
    assert len(conduits) == 1
    assert conduits[0].name == "Conduit 1"
    assert len(conduits[0].parts[0]) == 2


def test_read_pipe_network_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_pipes.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_pipe_nodes(str(path)) == []
    assert read_pipe_conduits(str(path)) == []
