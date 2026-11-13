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

from mayim_tools.hec_ras._common.h5io import is_group, open_h5
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


# Confirmed 2026-11-13 against project 2625038 (HEC-RAS 7.0.2): the
# Manning's n overrides by land cover class (one empty row when there are
# none).
LAND_COVER_N_TABLE = "Geometry/Land Cover (Manning's n)/Calibration Table"

# Per-cell datasets in each flow area group (confirmed 2026-11-13).
_CELL_MANNINGS_N = "Cells Center Manning's n"
_CELL_MIN_ELEV = "Cells Minimum Elevation"
_CELL_AREA = "Cells Surface Area"


@dataclass
class CellFeature:
    """One 2D mesh cell: its centre point and its polygon boundary (a
    closed ring - first point repeated as the last - built from
    'Cells FacePoint Indexes' with padding slots dropped), with the
    cell's Manning's n, lowest terrain level and plan area when the
    geometry has been preprocessed (None otherwise)."""

    center: tuple[float, float]
    polygon: list[tuple[float, float]]
    mannings_n: float | None = None
    min_elevation: float | None = None
    area_m2: float | None = None


def _finite(value):
    value = float(value)
    return None if value != value else value  # NaN -> None


def read_2d_flow_area_settings(geometry_hdf5_path: str) -> dict[str, dict]:
    """Per flow area name: its settings from 'Geometry/2D Flow Areas/
    Attributes' (confirmed 2026-11-13, HEC-RAS 7.0.2): default Manning's n
    ('Mann'), mesh spacing and real cell count. {} when absent."""
    from mayim_tools.hec_ras._common.hdf5_utils import read_attribute_table

    with open_h5(geometry_hdf5_path) as h5file:
        rows = read_attribute_table(h5file, f"{FLOW_AREAS_GROUP}/Attributes")
    out = {}
    for row in rows:
        name = row.get("Name")
        if not name:
            continue
        out[name] = {
            "mannings_n": _finite(row["Mann"]) if "Mann" in row else None,
            "spacing_dx": _finite(row["Spacing dx"]) if "Spacing dx" in row else None,
            "spacing_dy": _finite(row["Spacing dy"]) if "Spacing dy" in row else None,
            "cell_count": int(row["Cell Count"]) if "Cell Count" in row else None,
        }
    return out


def read_mannings_n_overrides(geometry_hdf5_path: str) -> list[tuple[str, float]]:
    """[(land cover class, Manning's n)] from the geometry's Manning's n
    overrides table; empty rows (no overrides) are left out."""
    from mayim_tools.hec_ras._common.hdf5_utils import read_attribute_table

    with open_h5(geometry_hdf5_path) as h5file:
        rows = read_attribute_table(h5file, LAND_COVER_N_TABLE)
    out = []
    for row in rows:
        values = list(row.values())
        if len(values) < 2 or not str(values[0]).strip():
            continue
        n = _finite(values[1])
        if n is not None:
            out.append((str(values[0]).strip(), n))
    return out


def read_break_lines(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every 2D Flow Area breakline defined in a HEC-RAS geometry
    HDF5 file.

    Returns an empty list if the project has no breaklines defined,
    rather than raising.
    """
    with open_h5(geometry_hdf5_path) as h5file:
        if BREAK_LINES_GROUP not in h5file:
            return []
        return read_polyline_features(h5file, BREAK_LINES_GROUP)


def read_refinement_regions(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every 2D Flow Area Refinement Region defined in a HEC-RAS
    geometry HDF5 file, as a closed-ring polygon feature per region.

    Returns an empty list if the project has no refinement regions
    defined, rather than raising.
    """
    with open_h5(geometry_hdf5_path) as h5file:
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
    with open_h5(geometry_hdf5_path) as h5file:
        if FLOW_AREAS_GROUP not in h5file:
            return []
        areas_group = h5file[FLOW_AREAS_GROUP]
        return [key for key in areas_group.keys() if is_group(areas_group[key])]


def read_2d_flow_area_perimeter(
    geometry_hdf5_path: str, flow_area_name: str
) -> list[tuple[float, float]]:
    """Read one 2D Flow Area's own outer perimeter ring."""
    with open_h5(geometry_hdf5_path) as h5file:
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
    with open_h5(geometry_hdf5_path) as h5file:
        group_path = f"{FLOW_AREAS_GROUP}/{flow_area_name}"
        if group_path not in h5file:
            raise KeyError(f"2D Flow Area '{flow_area_name}' not found.")
        area = h5file[group_path]

        centers = area["Cells Center Coordinate"][()]
        facepoint_indexes = area["Cells FacePoint Indexes"][()]
        facepoint_coords = area["FacePoints Coordinate"][()]
        extra = {}
        for key, dataset in (
            ("mannings_n", _CELL_MANNINGS_N),
            ("min_elevation", _CELL_MIN_ELEV),
            ("area_m2", _CELL_AREA),
        ):
            if dataset in area:
                extra[key] = area[dataset][()]

    # HEC-RAS lists 'Cell Count' real cells first, then one virtual cell per
    # perimeter face (no area, no terrain); those are left out. Confirmed
    # 2026-11-13: 1106 real + 110 virtual cells in project 2625038.
    count = (
        read_2d_flow_area_settings(geometry_hdf5_path)
        .get(flow_area_name, {})
        .get("cell_count")
    )
    n_cells = len(centers) if not count else min(int(count), len(centers))

    cells: list[CellFeature] = []
    for c, (center_row, index_row) in enumerate(
        zip(centers[:n_cells], facepoint_indexes[:n_cells], strict=True)
    ):
        valid_indexes = [i for i in index_row if i != _FACEPOINT_PADDING]
        ring = [
            (float(x), float(y))
            for x, y in (facepoint_coords[i] for i in valid_indexes)
        ]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])  # close the polygon ring
        values = {k: (_finite(v[c]) if c < len(v) else None) for k, v in extra.items()}
        cells.append(
            CellFeature(
                center=(float(center_row[0]), float(center_row[1])),
                polygon=ring,
                **values,
            )
        )
    return cells
