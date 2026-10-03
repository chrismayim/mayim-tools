"""2D mesh geometry reader: breaklines, flow-area perimeter, cell
centres/polygons, and refinement regions.

CONFIRMED 2026-09-22 against a real project (M2627_model.g02.hdf):

- Breaklines ('Geometry/2D Flow Area Break Lines') use the exact same
  'Polyline Info/Parts/Points' + 'Attributes' convention as Reference
  Lines and Boundary Condition Lines - reused directly via
  read_polyline_features, same as geometry/boundary_lines/core.py.

- The 2D Flow Area itself ('Geometry/2D Flow Areas/Perimeter 1' in this
  project - the group name is the flow area's own name, so it varies
  per project, hence list_2d_flow_area_names()) uses a DIFFERENT,
  richer convention, NOT the Polyline triple:
    - 'Perimeter' (N, 2) float64 - the flow area's own outer boundary,
      a single closed ring of X,Y points.
    - 'Cells Center Coordinate' (N_cells, 2) float64 - one X,Y per
      cell.
    - 'Cells FacePoint Indexes' (N_cells, 8) int32 - up to 8 facepoint
      indices per cell (into FacePoints Coordinate), building each
      cell's polygon boundary. CONFIRMED padding value: -1 marks an
      unused slot for a cell with fewer than 8 sides (1453 of 1481
      cells in the real project have at least one -1; indices used are
      0-based directly into FacePoints Coordinate, max index observed
      equals len(FacePoints Coordinate) - 1).
    - 'FacePoints Coordinate' (N_facepoints, 2) float64 - mesh vertex
      coordinates referenced by the above.

CONFIRMED 2026-09-23, once Chris added a refinement region to the test
project: 'Geometry/2D Flow Area Refinement Regions' uses the 'Polygon
Info/Parts/Points' convention (see core/hdf5_utils.read_polygon_features)
- a closed ring per region, same [start,count,start,count] encoding as
Polyline, just named Polygon. Reused directly, no new decoding logic.

Zero QGIS dependency - pure h5py/Python, independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import (
    PolylineFeature,
    read_polygon_features,
    read_polyline_features,
)

# Confirmed 2026-09-22 against a real project's geometry HDF5.
BREAK_LINES_GROUP = "Geometry/2D Flow Area Break Lines"
FLOW_AREAS_GROUP = "Geometry/2D Flow Areas"

# Confirmed 2026-09-23 against a real project's geometry HDF5.
REFINEMENT_REGIONS_GROUP = "Geometry/2D Flow Area Refinement Regions"

# Confirmed 2026-09-22: the sentinel value marking an unused FacePoint
# slot for cells with fewer than 8 sides.
_FACEPOINT_PADDING = -1


@dataclass
class CellFeature:
    """One 2D mesh cell: its centre point and its polygon boundary (a
    closed ring - first point repeated as the last - built from
    'Cells FacePoint Indexes' with padding slots dropped)."""

    center: tuple[float, float]
    polygon: list[tuple[float, float]]


def read_break_lines(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every 2D Flow Area breakline defined in a HEC-RAS geometry
    HDF5 file.

    Returns an empty list if the project has no breaklines defined,
    rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if BREAK_LINES_GROUP not in h5file:
            return []
        return read_polyline_features(h5file, BREAK_LINES_GROUP)


def read_refinement_regions(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every 2D Flow Area Refinement Region defined in a HEC-RAS
    geometry HDF5 file, as a closed-ring polygon feature per region.

    Returns an empty list if the project has no refinement regions
    defined, rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if REFINEMENT_REGIONS_GROUP not in h5file:
            return []
        return read_polygon_features(h5file, REFINEMENT_REGIONS_GROUP)


def list_2d_flow_area_names(geometry_hdf5_path: str) -> list[str]:
    """List the 2D Flow Area group names under 'Geometry/2D Flow
    Areas/' (e.g. 'Perimeter 1') - each is a separate flow area with
    its own Perimeter/Cells/FacePoints datasets, and the group name is
    user-defined per project, not a fixed constant.

    CONFIRMED 2026-09-22: 'Geometry/2D Flow Areas' also holds several
    DATASETS directly (Attributes, Cell Info, Cell Points, Polygon
    Info/Parts/Points) describing all flow areas' summary metadata and
    an alternate boundary encoding - these are NOT per-area subgroups
    and must be filtered out; only actual h5py.Group children (one per
    flow area, named after it) are real flow areas.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if FLOW_AREAS_GROUP not in h5file:
            return []
        areas_group = h5file[FLOW_AREAS_GROUP]
        return [
            key
            for key in areas_group.keys()
            if isinstance(areas_group[key], h5py.Group)
        ]


def read_2d_flow_area_perimeter(
    geometry_hdf5_path: str, flow_area_name: str
) -> list[tuple[float, float]]:
    """Read one 2D Flow Area's own outer perimeter ring."""
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        group_path = f"{FLOW_AREAS_GROUP}/{flow_area_name}"
        if group_path not in h5file:
            raise KeyError(f"2D Flow Area '{flow_area_name}' not found.")
        perimeter = h5file[group_path]["Perimeter"][()]
        return [(float(x), float(y)) for x, y in perimeter]


def read_2d_flow_area_cells(
    geometry_hdf5_path: str, flow_area_name: str
) -> list[CellFeature]:
    """Read every cell of one 2D Flow Area as a CellFeature (centre
    point + reconstructed polygon boundary).

    A cell whose FacePoint Indexes row is all padding (-1) - not
    expected in practice, but guarded against - gets an empty polygon
    rather than raising, so one malformed cell doesn't take down the
    whole read.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        group_path = f"{FLOW_AREAS_GROUP}/{flow_area_name}"
        if group_path not in h5file:
            raise KeyError(f"2D Flow Area '{flow_area_name}' not found.")
        area = h5file[group_path]

        centers = area["Cells Center Coordinate"][()]
        facepoint_indexes = area["Cells FacePoint Indexes"][()]
        facepoint_coords = area["FacePoints Coordinate"][()]

        cells: list[CellFeature] = []
        for center_row, index_row in zip(centers, facepoint_indexes, strict=True):
            valid_indexes = [i for i in index_row if i != _FACEPOINT_PADDING]
            ring = [
                (float(x), float(y))
                for x, y in (facepoint_coords[i] for i in valid_indexes)
            ]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])  # close the polygon ring
            cells.append(
                CellFeature(
                    center=(float(center_row[0]), float(center_row[1])), polygon=ring
                )
            )
        return cells
