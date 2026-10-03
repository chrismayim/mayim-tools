"""Reference Area geometry reader.

Confirmed 2026-09-23 against a real project, once Chris added a
Reference Area to the test project: 'Geometry/Reference Areas' uses the
'Polygon Info/Parts/Points' convention (see
core/hdf5_utils.read_polygon_features) - a closed ring per area, same
[point_start, point_count, part_start, part_count] encoding as
Polyline, just named Polygon. Attributes: 'Name', 'SA-2D'. Reused
directly, no new decoding logic needed.

Zero QGIS dependency - pure h5py/Python, independently testable (see
tests/test_reference_areas_core.py).
"""

from __future__ import annotations

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import (
    PolylineFeature,
    read_polygon_features,
)

# Confirmed 2026-09-23 against a real project's geometry HDF5.
REFERENCE_AREAS_GROUP = "Geometry/Reference Areas"


def read_reference_areas(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every Reference Area defined in a HEC-RAS geometry HDF5
    file, as a closed-ring polygon feature per area.

    Returns an empty list if the project has no reference areas
    defined, rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if REFERENCE_AREAS_GROUP not in h5file:
            return []
        return read_polygon_features(h5file, REFERENCE_AREAS_GROUP)
