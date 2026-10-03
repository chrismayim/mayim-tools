"""Pump Station geometry reader.

Confirmed 2026-09-23 against a real project, once Chris added a pump
station to the test project: 'Geometry/Pump Stations' uses the same
Attributes + plain (N, 2) Points convention as Reference Points, IC
Points, and Pipe Nodes (see core/hdf5_utils.read_point_features). The
Attributes row carries each pump station's inlet/outlet/reference
River-Reach-RS or SA/2D references, plus a 'Pump Groups' count - the
per-pump-curve detail under the sibling 'Pump Groups' subgroup is not
read here, only the station's own location and summary attributes.

Zero QGIS dependency - pure h5py/Python, independently testable (see
tests/test_pump_stations_core.py).
"""

from __future__ import annotations

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import PointFeature, read_point_features

# Confirmed 2026-09-23 against a real project's geometry HDF5.
PUMP_STATIONS_GROUP = "Geometry/Pump Stations"


def read_pump_stations(geometry_hdf5_path: str) -> list[PointFeature]:
    """Read every Pump Station defined in a HEC-RAS geometry HDF5 file.

    Returns an empty list if the project has no pump stations defined,
    rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if PUMP_STATIONS_GROUP not in h5file:
            return []
        return read_point_features(h5file, PUMP_STATIONS_GROUP)
