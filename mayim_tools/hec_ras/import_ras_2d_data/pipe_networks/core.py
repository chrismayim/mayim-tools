"""Pipe Network (storm drain) geometry reader: nodes and conduits.

Confirmed 2026-09-23 against a real project, once Chris added a pipe
network to the test project:

- 'Geometry/Pipe Nodes' uses the same Attributes + plain (N, 2) Points
  convention as Reference Points/IC Points/Pump Stations (see
  core/hdf5_utils.read_point_features). Per-node inlet detail lives
  under sibling 'Side Inlets'/'Top Inlets' subgroups, not read here -
  only the node's own location and summary attributes (name, system,
  node type/status, invert/terrain elevation, connection counts).
- 'Geometry/Pipe Conduits' uses the 'Polyline Info/Parts/Points'
  convention (see core/hdf5_utils.read_polyline_features) - a single-
  part line per conduit, same as breaklines/reference lines.
- 'Geometry/Pipe Networks' holds only network-level metadata (just a
  'Name' field, e.g. 'Base') with no geometry of its own - there is
  nothing to read as a spatial layer from this group, so no reader is
  provided for it.

Zero QGIS dependency - pure h5py/Python, independently testable (see
tests/test_pipe_networks_core.py).
"""

from __future__ import annotations

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import (
    PointFeature,
    PolylineFeature,
    read_point_features,
    read_polyline_features,
)

# Confirmed 2026-09-23 against a real project's geometry HDF5.
PIPE_NODES_GROUP = "Geometry/Pipe Nodes"
PIPE_CONDUITS_GROUP = "Geometry/Pipe Conduits"


def read_pipe_nodes(geometry_hdf5_path: str) -> list[PointFeature]:
    """Read every pipe network node (manhole/junction/outfall/etc.)
    defined in a HEC-RAS geometry HDF5 file.

    Returns an empty list if the project has no pipe network defined,
    rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if PIPE_NODES_GROUP not in h5file:
            return []
        return read_point_features(h5file, PIPE_NODES_GROUP)


def read_pipe_conduits(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every pipe network conduit defined in a HEC-RAS geometry
    HDF5 file.

    Returns an empty list if the project has no pipe network defined,
    rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if PIPE_CONDUITS_GROUP not in h5file:
            return []
        return read_polyline_features(h5file, PIPE_CONDUITS_GROUP)
