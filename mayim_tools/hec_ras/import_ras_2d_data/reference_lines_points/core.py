"""Reference Line / Reference Point geometry reader.

This is the one geometry reader with no equivalent tool in
ras-commander-qgis at all (confirmed 2026-09-22 - see
claude/hec_ras_automation_plugin_research.md). Both Reference Lines and
Reference Points are now confirmed against a real project (2026-09-23:
3 Reference Points appeared once Chris added them to the test project,
read back correctly with zero code changes needed).

Zero QGIS dependency - pure h5py/Python, independently testable (see
tests/test_reference_lines_points_core.py). The QGIS-facing wrapper is
geometry/load_geometry_algorithm.py, the single consolidated tool.
"""

from __future__ import annotations

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import (
    PointFeature,
    PolylineFeature,
    read_point_features,
    read_polyline_features,
)

# Confirmed 2026-09-22 against a real project's geometry HDF5.
REFERENCE_LINES_GROUP = "Geometry/Reference Lines"

# Confirmed 2026-09-23 against a real project's geometry HDF5 (3
# Reference Points, Attributes fields Name/SA-2D/Cell Index/USXSID/
# DSXSID/US Fraction, plain (N, 2) Points dataset).
REFERENCE_POINTS_GROUP = "Geometry/Reference Points"


def read_reference_lines(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every Reference Line defined in a HEC-RAS geometry HDF5 file.

    Parameters
    ----------
    geometry_hdf5_path:
        Path to the HEC-RAS geometry HDF5 file (the companion .hdf next
        to a .gXX geometry file, or a plan HDF5 that embeds the same
        Geometry/ group).

    Returns
    -------
    list[PolylineFeature]
        Empty list if the project has no Reference Lines defined at all
        (a normal, non-error case - not every model has them).
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if REFERENCE_LINES_GROUP not in h5file:
            return []
        return read_polyline_features(h5file, REFERENCE_LINES_GROUP)


def read_reference_points(geometry_hdf5_path: str) -> list[PointFeature]:
    """Read every Reference Point defined in a HEC-RAS geometry HDF5
    file. Confirmed 2026-09-23 (see REFERENCE_POINTS_GROUP above).

    Returns an empty list if the group isn't present, same reasoning as
    read_reference_lines.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if REFERENCE_POINTS_GROUP not in h5file:
            return []
        return read_point_features(h5file, REFERENCE_POINTS_GROUP)
