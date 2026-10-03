"""2D Boundary Condition Line geometry reader (inflow/outflow lines).

Confirmed 2026-09-22 against a real project (M2627_model.g02.hdf): this
group uses the same 'Polyline Info/Parts/Points' + 'Attributes'
convention as Reference Lines, so it's a direct reuse of
core/hdf5_utils.read_polyline_features - no new decoding logic needed.

The Attributes table's 'Type' field is RAS's own External/Internal
classification (whether the BC line sits on the 2D area's outer
perimeter or is interior to it) - NOT an inflow/outflow direction flag.
Whether a boundary condition is inflow or outflow is set by the user's
naming and the unsteady flow file, not stored here; this reader exposes
the raw Attributes table (including 'Type' and 'SA-2D') and leaves any
inflow/outflow interpretation to the caller/UI.

Zero QGIS dependency - pure h5py/Python, independently testable (see
tests/test_boundary_lines_core.py).
"""

from __future__ import annotations

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import (
    PolylineFeature,
    read_polyline_features,
)

# Confirmed 2026-09-22 against a real project's geometry HDF5.
BOUNDARY_CONDITION_LINES_GROUP = "Geometry/Boundary Condition Lines"


def read_boundary_condition_lines(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every 2D boundary condition line defined in a HEC-RAS
    geometry HDF5 file.

    Returns an empty list if the project has no boundary condition
    lines defined at all, rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if BOUNDARY_CONDITION_LINES_GROUP not in h5file:
            return []
        return read_polyline_features(h5file, BOUNDARY_CONDITION_LINES_GROUP)
