"""Initial Condition (IC) Point geometry reader.

Confirmed 2026-09-23 against a real project, once Chris added an IC
Point to the test project: 'Geometry/IC Points' uses the same
Attributes + plain (N, 2) Points convention as Reference Points, Pipe
Nodes, and Pump Stations (see core/hdf5_utils.read_point_features).
Attributes: 'Name', 'SA/2D', 'Cell Index'.

Zero QGIS dependency - pure h5py/Python, independently testable (see
tests/test_initial_condition_points_core.py).
"""

from __future__ import annotations

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import PointFeature, read_point_features

# Confirmed 2026-09-23 against a real project's geometry HDF5.
IC_POINTS_GROUP = "Geometry/IC Points"


def read_initial_condition_points(geometry_hdf5_path: str) -> list[PointFeature]:
    """Read every Initial Condition Point defined in a HEC-RAS geometry
    HDF5 file.

    Returns an empty list if the project has no IC points defined,
    rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if IC_POINTS_GROUP not in h5file:
            return []
        return read_point_features(h5file, IC_POINTS_GROUP)
