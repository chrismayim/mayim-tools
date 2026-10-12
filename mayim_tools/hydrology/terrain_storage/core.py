"""
Core terrain-storage logic - pure NumPy, no QGIS dependency (GDAL is
imported only inside the file-I/O helpers at the bottom).

PURPOSE
Stage-storage (elevation-area-volume) curves for ponds, dams, detention
basins and natural depressions, read straight off a DEM inside a
user-drawn storage-area polygon.

METHOD
- Cell weights: each DEM cell gets the fraction of its area that lies
  inside the polygon (scanline even-odd fill of S x S sub-cells, so
  islands/holes and multipart polygons work). S = 1 is the simple
  "cell centre inside polygon" rule.
- Level pool over flat cells (prism method): at water level h a cell is
  wet when it can hold water below h, and it stores
  w * cell_area * (h - z). Plan area is the sum of w * cell_area over wet
  cells. This is exact for the DEM taken as flat-topped cells - no
  average-end-area or conic approximation.
- Connectivity: a priority flood (Dijkstra on the minimax path level,
  8-connected) from the lowest cell gives every cell the water level at
  which it first joins the pool ("entry level"). In CONNECTED mode a
  cell is wet when its entry level is below h, so separate hollows
  behind a ridge only fill once the pool overtops the ridge. In ALL CELLS
  mode (the HEC-RAS storage-area convention) every cell below h counts,
  connected or not; the volume held in disconnected hollows is reported.
- Spill level: the lowest water level at which the pool reaches the edge
  of the polygon (or the edge of the DEM) - the minimax path level from
  the floor to any edge cell. Drawing the polygon beyond the crest still
  finds the true crest; drawing it inside the crest cuts the side slopes
  and is flagged.
- Maximum stage: the spill level by default, or a user-specified level
  (e.g. for a proposed embankment) - the polygon edge then acts as a
  vertical wall, which is flagged when it lies above the spill level.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.d8_network import GridError
from mayim_tools.hydrology._common.depression import (  # noqa: F401 - re-exported
    _NEIGHBOURS,
    DEFAULT_SUPERSAMPLE,
    StageStorage,
    _index_edges,
    coverage_fraction,
    edge_cells,
    polygon_window,
    power_fit,
    priority_flood_levels,
    raster_info,
    read_window,
    window_geotransform,
)
from mayim_tools.hydrology._common.grid_polygons import (
    group_polygons,
    ring_to_map,
    simplify_ring,
    trace_rings,
)

from .bathymetry import DEFAULT_FLAT_TOL_M, WaterBody, estimate_bathymetry

VALID_MODES = ("connected", "all")
DEFAULT_DEPTH_NODATA = -9999.0
MAX_STAGES = 20000
DISCONNECTED_WARN_SHARE = 0.05
TIGHT_EDGE_TOL_M = 0.05
MIN_CONTOUR_CELLS = 3.0  # closed contours enclosing less are DEM noise


# Raised for invalid inputs (bad grid, bad interval, no storage, ...); the
# same class as the shared d8_network.GridError, so errors from the shared
# helpers in _common/depression.py are StorageErrors too.
StorageError = GridError


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class StorageArea:
    """One storage-area polygon feature, already in the DEM's CRS.

    ``polygons`` is [[outer, hole, ...], ...] with every ring a closed
    list of (x, y) map coordinates (one entry per multipart part)."""

    storage_id: str
    polygons: list[list[list[tuple[float, float]]]]
    src_fid: int = -1


@dataclass
class StorageResult:
    """Everything produced for one storage area."""

    storage_id: str
    src_fid: int
    mode: str
    floor_z: float
    floor_xy: tuple[float, float]
    spill_z: float | None
    spill_xy: tuple[float, float] | None
    top_stage: float
    top_source: str  # "spill" or "user"
    table: list[dict]
    ws_polygons: list[tuple[float, dict, list]]  # (stage, row, polygons_map)
    depth: np.ndarray  # float32 window, NaN where dry
    window: tuple[int, int, int, int]  # (row0, col0, rows, cols) in the DEM
    attributes: dict
    warnings: list[str] = field(default_factory=list)
    window_gt: tuple | None = None  # geotransform of ``depth``
    polygons: list = field(default_factory=list)  # storage polygon, map coords
    detail: dict = field(default_factory=dict)  # report/figure data
    contours: list = field(default_factory=list)  # (stage, row, [line, ...])
    bathymetry: object | None = None  # bathymetry.BathymetryResult when estimated
    range_tables: dict = field(default_factory=dict)  # "low"/"high" -> table


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def check_geotransform(geotransform) -> tuple[float, float]:
    """Returns positive (cell_size_x, cell_size_y); rejects rotated grids."""
    if abs(geotransform[2]) > 0 or abs(geotransform[4]) > 0:
        raise StorageError(
            "Rotated/sheared rasters are not supported - warp the DEM to a "
            "north-up grid first."
        )
    sx, sy = abs(float(geotransform[1])), abs(float(geotransform[5]))
    if sx <= 0 or sy <= 0:
        raise StorageError("Invalid raster cell size in geotransform.")
    return sx, sy


def nodata_mask(array: np.ndarray, nodata_value) -> np.ndarray:
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        mask = np.isnan(array)
    else:
        mask = np.zeros(array.shape, dtype=bool)
    if nodata_value is not None and not np.isnan(nodata_value):
        mask |= array == nodata_value
    return mask


# ---------------------------------------------------------------------------
# Polygon -> cell coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Priority flood: entry levels and spill
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoothed water-edge contours
# ---------------------------------------------------------------------------

# Marching-squares segments per case (corner bits tl=8, tr=4, br=2, bl=1 set
# when the corner is BELOW the level). Edges: 0 top, 1 right, 2 bottom,
# 3 left. Saddles (5, 10) are resolved with the block-centre mean.
_MS_SEGMENTS = {
    1: [(3, 2)],
    2: [(2, 1)],
    3: [(3, 1)],
    4: [(0, 1)],
    6: [(0, 2)],
    7: [(3, 0)],
    8: [(3, 0)],
    9: [(0, 2)],
    11: [(0, 1)],
    12: [(3, 1)],
    13: [(2, 1)],
    14: [(3, 2)],
}
_MS_SADDLE = {
    # (case, centre below level) -> segments
    (5, True): [(3, 0), (2, 1)],
    (5, False): [(0, 1), (3, 2)],
    (10, True): [(0, 1), (3, 2)],
    (10, False): [(3, 0), (2, 1)],
}


def marching_squares(values: np.ndarray, level: float) -> list[np.ndarray]:
    """Contour lines of ``values`` (sampled at cell centres) at ``level``,
    as arrays of (row, col) positions in fractional cell-index space
    (cell centre of row r, col c = (r, c)). Crossings are linearly
    interpolated between neighbouring cell centres. Values must be finite;
    lines that reach the grid edge are returned open, all others closed
    (first point repeated at the end)."""
    v = np.asarray(values, dtype=np.float64)
    rows, cols = v.shape
    if rows < 2 or cols < 2:
        return []
    below = v < level
    tl, tr = below[:-1, :-1], below[:-1, 1:]
    br, bl = below[1:, 1:], below[1:, :-1]
    case = tl * 8 + tr * 4 + br * 2 + bl * 1
    n_h = rows * (cols - 1)  # horizontal edges (between row neighbours)

    def edge_id(r, c, e):
        # 0 top: (r,c)-(r,c+1); 2 bottom: (r+1,c)-(r+1,c+1)
        # 3 left: (r,c)-(r+1,c); 1 right: (r,c+1)-(r+1,c+1)
        if e == 0:
            return r * (cols - 1) + c
        if e == 2:
            return (r + 1) * (cols - 1) + c
        if e == 3:
            return n_h + r * cols + c
        return n_h + r * cols + c + 1

    def edge_point(eid):
        if eid < n_h:
            r, c = divmod(eid, cols - 1)
            a, b = v[r, c], v[r, c + 1]
            t = (level - a) / (b - a)
            return (float(r), c + float(t))
        r, c = divmod(eid - n_h, cols)
        a, b = v[r, c], v[r + 1, c]
        t = (level - a) / (b - a)
        return (r + float(t), float(c))

    links: dict[int, list[int]] = {}
    for r, c in zip(*np.nonzero((case > 0) & (case < 15)), strict=True):
        k = int(case[r, c])
        if k in (5, 10):
            centre = 0.25 * (v[r, c] + v[r, c + 1] + v[r + 1, c] + v[r + 1, c + 1])
            segs = _MS_SADDLE[(k, bool(centre < level))]
        else:
            segs = _MS_SEGMENTS[k]
        for e0, e1 in segs:
            a, b = edge_id(r, c, e0), edge_id(r, c, e1)
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)

    lines = []
    visited: set[int] = set()
    # open chains first (start at degree-1 edges), then closed loops
    starts = [e for e, nb in links.items() if len(nb) == 1]
    starts += [e for e in links if e not in starts]
    for start in starts:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        prev, cur = None, start
        while True:
            nxt = [e for e in links[cur] if e != prev and e not in visited]
            if not nxt:
                if prev is not None and start in links[cur] and len(chain) > 2:
                    chain.append(start)  # closed loop
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            visited.add(cur)
        if len(chain) >= 2:
            lines.append(np.asarray([edge_point(e) for e in chain]))
    return lines


def chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting smoothing (1/4 - 3/4 rule). A closed line
    (first point == last) stays closed; an open line keeps its end
    points."""
    pts = np.asarray(points, dtype=np.float64)
    if iterations <= 0 or len(pts) < 3:
        return pts
    closed = bool(np.allclose(pts[0], pts[-1]))
    for _ in range(int(iterations)):
        if closed:
            ring = pts[:-1]
            nxt = np.roll(ring, -1, axis=0)
            q = 0.75 * ring + 0.25 * nxt
            r_ = 0.25 * ring + 0.75 * nxt
            new = np.empty((2 * len(ring), 2))
            new[0::2], new[1::2] = q, r_
            pts = np.vstack([new, new[:1]])
        else:
            a, b = pts[:-1], pts[1:]
            q = 0.75 * a + 0.25 * b
            r_ = 0.25 * a + 0.75 * b
            mid = np.empty((2 * len(a), 2))
            mid[0::2], mid[1::2] = q, r_
            pts = np.vstack([pts[:1], mid[1:-1], pts[-1:]])
    return pts


def index_to_map(points_rc: np.ndarray, geotransform) -> list[tuple[float, float]]:
    """(row, col) cell-centre index positions -> map (x, y)."""
    p = np.asarray(points_rc, dtype=np.float64)
    x = geotransform[0] + (p[:, 1] + 0.5) * geotransform[1]
    y = geotransform[3] + (p[:, 0] + 0.5) * geotransform[5]
    return list(zip(x.tolist(), y.tolist(), strict=True))


def contour_levels(
    floor: float,
    top: float,
    spill: float | None,
    interval: float | None,
    table_stages: Sequence[float] = (),
) -> list[tuple[float, str]]:
    """Water levels for the water-edge contours, as (stage, kind):
    floor + k x interval for round DEPTHS above the floor ("depth"), then
    the spill level ("spill") and the top stage ("top", only when it
    differs from the spill level, i.e. a specified maximum stage).
    ``interval`` None/0 uses the table stages instead of round depths."""
    eps = 1e-6
    levels: list[tuple[float, str]] = []
    if interval and interval > 0:
        if (top - floor) / interval > MAX_STAGES:
            raise StorageError(
                "Too many contours requested - use a larger contour interval."
            )
        k = 1
        while floor + k * interval < top - eps:
            levels.append((round(floor + k * interval, 9), "depth"))
            k += 1
    else:
        levels = [(h, "depth") for h in table_stages if floor + eps < h < top - eps]
    if spill is not None and floor + eps < spill <= top + eps:
        levels = [lv for lv in levels if abs(lv[0] - spill) > eps]
        levels.append((float(spill), "spill"))
    if not any(abs(h - top) <= eps for h, _ in levels):
        levels.append((float(top), "top"))
    return sorted(levels)


def water_edge_contours(
    entry: np.ndarray,
    sel: np.ndarray,
    level: float,
    geotransform,
    smoothing: int,
    min_cells: float = MIN_CONTOUR_CELLS,
) -> list[list[tuple[float, float]]]:
    """Smoothed outline(s) of the water surface at ``level``: contours of
    the cells' entry level (equal to the ground on the pool's banks, so the
    line is the interpolated water's edge and hollows not yet joined to
    the pool are left out). Cells outside the storage polygon or NoData are
    set high, so every line closes; clip to the polygon afterwards. Closed
    lines enclosing less than ``min_cells`` cells (single-cell islands and
    pits from DEM noise) are dropped."""
    finite = entry[sel & np.isfinite(entry)]
    if finite.size == 0:
        return []
    big = float(finite.max()) + max(1.0, float(np.ptp(finite)))
    field_ = np.where(sel & np.isfinite(entry), entry, big)
    field_ = np.pad(field_, 1, constant_values=big)
    lines = []
    for line in marching_squares(field_, level):
        if np.allclose(line[0], line[-1]) and len(line) > 3:
            enclosed = 0.5 * abs(
                float(np.sum(line[:-1, 0] * line[1:, 1] - line[1:, 0] * line[:-1, 1]))
            )
            if enclosed < min_cells:
                continue
        smooth = chaikin(line, smoothing) - 1.0  # undo padding
        if len(smooth) >= 2:
            lines.append(index_to_map(smooth, geotransform))
    return lines


# ---------------------------------------------------------------------------
# Stage-storage table
# ---------------------------------------------------------------------------


def stage_levels(floor: float, top: float, interval: float) -> list[float]:
    """Floor, then every multiple of ``interval`` strictly between floor
    and top, then top (round numbers for model input tables)."""
    if not interval > 0:
        raise StorageError("The stage interval must be greater than zero.")
    if top <= floor:
        return [float(floor)]
    if (top - floor) / interval > MAX_STAGES:
        raise StorageError(
            f"{(top - floor) / interval:.0f} stages requested (more than "
            f"{MAX_STAGES}) - use a larger stage interval."
        )
    levels = [float(floor)]
    k = math.floor(floor / interval + 1e-9) + 1
    eps = 1e-9 * max(1.0, abs(top))
    while True:
        v = round(k * interval, 9)
        if v >= top - eps:
            break
        if v > floor + eps:
            levels.append(v)
        k += 1
    levels.append(float(top))
    return levels


def build_table(stages: Sequence[float], storage: StageStorage, floor: float):
    rows = []
    prev_volume = 0.0
    for h in stages:
        area, volume, n_wet = storage.at(h)
        rows.append(
            {
                "stage_m": round(h, 4),
                "depth_m": round(h - floor, 4),
                "area_m2": round(area, 3),
                "area_ha": round(area / 1e4, 6),
                "volume_m3": round(volume, 3),
                "volume_ML": round(volume / 1e3, 6),
                "inc_vol_m3": round(volume - prev_volume, 3),
                "wet_cells": n_wet,
            }
        )
        prev_volume = volume
    return rows


def depth_distribution(depths, areas, n_bins: int = 10) -> list[dict]:
    """Water-surface area by depth class at one stage: ``n_bins`` equal
    classes from 0 to the maximum depth, as area (m2) and share (%)."""
    d = np.asarray(depths, dtype=np.float64).ravel()
    a = np.asarray(areas, dtype=np.float64).ravel()
    if d.size == 0 or a.sum() <= 0:
        return []
    top = float(d.max())
    if top <= 0:
        return []
    edges = np.linspace(0.0, top, n_bins + 1)
    counts, _ = np.histogram(d, bins=edges, weights=a)
    total = float(a.sum())
    return [
        {
            "from_m": float(edges[i]),
            "to_m": float(edges[i + 1]),
            "area_m2": float(counts[i]),
            "pct": 100.0 * float(counts[i]) / total,
        }
        for i in range(n_bins)
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_storage(
    dem_window: np.ndarray,
    dem_nodata,
    window_gt,
    window: tuple[int, int, int, int],
    area: StorageArea,
    interval: float,
    *,
    max_stage: float | None = None,
    mode: str = "connected",
    supersample: int = DEFAULT_SUPERSAMPLE,
    polygon_interval: float | None = None,
    contour_interval: float | None = None,
    contour_smoothing: int = 3,
    is_canceled: Callable[[], bool] | None = None,
) -> StorageResult:
    """Stage-storage curve, water-surface outlines and top-stage depth
    grid for one storage area over a DEM window (``window_gt`` is the
    geotransform of ``dem_window``; ``window`` is its (row0, col0, rows,
    cols) position in the full DEM, carried through for mosaicking).

    ``polygon_interval``: water-surface outlines and smoothed water-edge
    contours at this spacing (plus the top stage); None/0 = every table
    stage. ``contour_interval``: the smoothed water-edge
    contours are drawn at round depths above the floor (floor + k x
    interval), then at the spill level and the top stage; None = same as
    ``polygon_interval``, 0 = every table stage. ``contour_smoothing``:
    Chaikin iterations."""
    if mode not in VALID_MODES:
        raise StorageError(f"Unknown mode '{mode}' (expected {VALID_MODES}).")
    sx, sy = check_geotransform(window_gt)
    cell_area = sx * sy
    z = np.asarray(dem_window, dtype=np.float64)
    warnings: list[str] = []
    sid = area.storage_id

    cover = coverage_fraction(area.polygons, window_gt, z.shape, supersample)
    invalid = nodata_mask(z, dem_nodata)
    inside = cover > 0
    domain = inside & ~invalid
    if not domain.any():
        raise StorageError(
            f"Storage '{sid}': no valid DEM cells inside the polygon (check "
            "that the polygon overlaps the DEM and is not all NoData)."
        )
    n_nodata = int((inside & invalid).sum())
    if n_nodata:
        warnings.append(
            f"Storage '{sid}': {n_nodata} NoData cell(s) inside the polygon "
            f"({n_nodata * cell_area / 1e4:.4f} ha) are treated as solid "
            "ground (no storage, no flow path)."
        )

    # Polygon reaches beyond the DEM? (window clipped at the raster edge)
    outer_xy = np.asarray(
        [p for poly in area.polygons for p in poly[0]], dtype=np.float64
    )
    u = (outer_xy[:, 0] - window_gt[0]) / window_gt[1]
    v = (outer_xy[:, 1] - window_gt[3]) / window_gt[5]
    rows, cols = z.shape
    if (u < 0).any() or (v < 0).any() or (u > cols).any() or (v > rows).any():
        warnings.append(
            f"Storage '{sid}': the polygon extends beyond the DEM - storage "
            "is computed for the covered part only and the DEM edge is "
            "treated as an outflow edge."
        )

    zd = np.where(domain, z, np.inf)
    floor_idx = np.unravel_index(int(np.argmin(zd)), z.shape)
    floor_z = float(z[floor_idx])
    exterior = ~inside  # outside the polygon (NoData inside it is a wall)
    edge = edge_cells(domain, exterior)
    entry_connected, spill_cell = priority_flood_levels(
        z, domain, floor_idx, edge=edge, is_canceled=is_canceled
    )

    def centre(rc):
        return (
            window_gt[0] + (rc[1] + 0.5) * window_gt[1],
            window_gt[3] + (rc[0] + 0.5) * window_gt[5],
        )

    spill_z = spill_xy = None
    if spill_cell is not None:
        spill_z = float(entry_connected[spill_cell])
        # The crest that sets the level is the cell on the flood path whose
        # own elevation equals the spill level (the edge cell itself when the
        # polygon is drawn on the crest; further in when drawn beyond it).
        crest = np.argwhere(domain & (z == spill_z) & (entry_connected == spill_z))
        if len(crest):
            d2 = (crest[:, 0] - spill_cell[0]) ** 2 + (crest[:, 1] - spill_cell[1]) ** 2
            crest_cell = tuple(int(x) for x in crest[int(np.argmin(d2))])
        else:
            crest_cell = spill_cell
        spill_xy = centre(crest_cell)
        # polygon drawn inside the crest? ground just outside it is higher
        r, c = spill_cell
        outside = [
            z[r + dr, c + dc]
            for dr, dc in _NEIGHBOURS
            if 0 <= r + dr < rows
            and 0 <= c + dc < cols
            and exterior[r + dr, c + dc]
            and not invalid[r + dr, c + dc]
        ]
        if outside and min(outside) > spill_z + TIGHT_EDGE_TOL_M:
            warnings.append(
                f"Storage '{sid}': the spill level ({spill_z:.3f} m) is set by "
                "the polygon edge, not by a terrain crest - the ground just "
                f"outside the polygon there is higher ({min(outside):.3f} m). "
                "The polygon probably cuts through the side slopes; extend it "
                "beyond the crest/embankment to capture the full storage."
            )

    if max_stage is not None:
        top, top_source = float(max_stage), "user"
        if top < floor_z:
            raise StorageError(
                f"Storage '{sid}': maximum stage {top:.3f} m is below the "
                f"lowest ground in the polygon ({floor_z:.3f} m)."
            )
        if spill_z is not None and top > spill_z + 1e-6:
            warnings.append(
                f"Storage '{sid}': maximum stage {top:.3f} m is above the spill "
                f"level ({spill_z:.3f} m) - storage above the spill level "
                "assumes the polygon edge is a vertical wall (e.g. a proposed "
                "embankment)."
            )
    else:
        if spill_z is None:
            raise StorageError(
                f"Storage '{sid}': no spill level found (the pool never reaches "
                "the polygon edge) - set a maximum stage."
            )
        top, top_source = spill_z, "spill"

    if top <= floor_z + 1e-9:
        if top_source == "user":
            raise StorageError(
                f"Storage '{sid}': the maximum stage ({top:.3f} m) equals the "
                f"lowest ground in the polygon ({floor_z:.3f} m), so there is no "
                "storage to compute."
            )
        hint = ""
        if any("cuts through" in w for w in warnings):
            hint = (
                " The ground just outside the polygon is higher at that point, "
                "so the polygon probably stops on the side slope or at the "
                "waterline - extend it beyond the crest/embankment."
            )
        raise StorageError(
            f"Storage '{sid}': no depression found - the lowest ground "
            f"({floor_z:.3f} m) is on the polygon edge, so the polygon holds no "
            f"water before it spills.{hint} Draw the polygon around a closed "
            "depression, or set a maximum stage (e.g. a proposed embankment "
            "crest) to compute storage behind the polygon edge."
        )

    if mode == "connected":
        entry = entry_connected
    else:
        entry = np.where(domain, z, np.inf)
    sel = domain & np.isfinite(entry)
    storage = StageStorage(z[sel], cover[sel], entry[sel], cell_area)

    stages = stage_levels(floor_z, top, interval)
    table = build_table(stages, storage, floor_z)

    if mode == "all" and top > floor_z:
        sel_c = domain & np.isfinite(entry_connected)
        connected = StageStorage(
            z[sel_c], cover[sel_c], entry_connected[sel_c], cell_area
        )
        v_all = table[-1]["volume_m3"]
        v_con = connected.at(top)[1]
        if v_all > 0 and (v_all - v_con) / v_all > DISCONNECTED_WARN_SHARE:
            warnings.append(
                f"Storage '{sid}': {100 * (v_all - v_con) / v_all:.1f}% of the "
                f"volume at {top:.3f} m ({v_all - v_con:.1f} m3) is in hollows "
                "not connected to the main pool at that level (counted in "
                "'All cells' mode)."
            )

    # Water-surface outlines
    if polygon_interval and polygon_interval > 0:
        ws_stages = [h for h in stage_levels(floor_z, top, polygon_interval)[1:]]
    else:
        ws_stages = list(stages[1:])
    ws_polygons, contours = [], []
    wet_entry = np.where(sel, entry, np.inf)
    for h in ws_stages:
        if is_canceled is not None and is_canceled():
            break
        mask = wet_entry < h
        if not mask.any():
            continue
        polys_idx = group_polygons(trace_rings(mask))
        polys_map = [
            [ring_to_map(simplify_ring(r), window_gt) for r in poly]
            for poly in polys_idx
        ]
        row = build_table([h], storage, floor_z)[0]
        ws_polygons.append((h, row, polys_map))

    # Smoothed water-edge contours at round DEPTHS above the floor (own
    # interval; default = polygon spacing), then the spill level and the top
    for h, kind in contour_levels(
        floor_z,
        top,
        spill_z,
        polygon_interval if contour_interval is None else contour_interval,
        stages,
    ):
        if is_canceled is not None and is_canceled():
            break
        lines = water_edge_contours(entry, sel, h, window_gt, contour_smoothing)
        if lines:
            row = build_table([h], storage, floor_z)[0]
            row["kind"] = kind
            contours.append((h, row, lines))

    depth = np.where(sel & (entry < top), top - z, np.nan).astype(np.float32)

    top_row = table[-1]
    wet_top = sel & (entry < top)
    mean_depth = (
        top_row["volume_m3"] / top_row["area_m2"] if top_row["area_m2"] > 0 else None
    )
    poly_m2 = float(cover.sum()) * cell_area
    detail = {
        "depth_hist": depth_distribution(
            (top - z)[wet_top], cover[wet_top] * cell_area
        ),
        "fit_volume": power_fit(
            [r["depth_m"] for r in table], [r["volume_m3"] for r in table]
        ),
        "fit_area": power_fit(
            [r["depth_m"] for r in table], [r["area_m2"] for r in table]
        ),
    }
    attributes = {
        "storage_id": sid,
        "src_fid": area.src_fid,
        "mode": mode,
        "z_floor": round(floor_z, 3),
        "z_spill": None if spill_z is None else round(spill_z, 3),
        "z_top": round(top, 3),
        "top_src": top_source,
        "max_depth": round(top - floor_z, 3),
        "area_m2": top_row["area_m2"],
        "area_ha": top_row["area_ha"],
        "vol_m3": top_row["volume_m3"],
        "vol_ML": top_row["volume_ML"],
        "mean_dep": None if mean_depth is None else round(mean_depth, 3),
        "poly_ha": round(poly_m2 / 1e4, 6),
        "wet_pct": round(100.0 * top_row["area_m2"] / poly_m2, 2) if poly_m2 else None,
        "n_stages": len(table),
        "cell_m": round(math.sqrt(cell_area), 4),
        "x_floor": centre(floor_idx)[0],
        "y_floor": centre(floor_idx)[1],
        "x_spill": None if spill_xy is None else spill_xy[0],
        "y_spill": None if spill_xy is None else spill_xy[1],
    }
    return StorageResult(
        storage_id=sid,
        src_fid=area.src_fid,
        mode=mode,
        floor_z=floor_z,
        floor_xy=centre(floor_idx),
        spill_z=spill_z,
        spill_xy=spill_xy,
        top_stage=top,
        top_source=top_source,
        table=table,
        ws_polygons=ws_polygons,
        depth=depth,
        window=tuple(int(x) for x in window),
        attributes=attributes,
        warnings=warnings,
        window_gt=tuple(window_gt),
        polygons=area.polygons,
        detail=detail,
        contours=contours,
    )


def compute_storage_with_bed(
    dem_window: np.ndarray,
    dem_nodata,
    window_gt,
    window: tuple[int, int, int, int],
    area: StorageArea,
    interval: float,
    *,
    centrelines: Sequence = (),
    known_max_depth: float | None = None,
    flat_tol: float = DEFAULT_FLAT_TOL_M,
    water_level: float | None = None,
    buffer_m: float | None = None,
    reach_m: float | None = None,
    hydrolakes: dict | None = None,
    **storage_kwargs,
) -> StorageResult:
    """Stage-storage for a DEM whose surface is a WATER SURFACE: the bed below
    the surveyed water is estimated first (bathymetry.estimate_bathymetry,
    low / central / high), written into the DEM, and the curve is computed
    from the central bed. The low and high beds give volume_low_m3 /
    volume_high_m3 at every stage of the central table, and the result
    carries the BathymetryResult for the outputs and the report."""
    body = WaterBody(
        area.storage_id, area.polygons, area.src_fid, list(centrelines), known_max_depth
    )
    bathy = estimate_bathymetry(
        dem_window,
        dem_nodata,
        window_gt,
        window,
        body,
        flat_tol=flat_tol,
        water_level=water_level,
        buffer_m=buffer_m,
        reach_m=reach_m,
        supersample=storage_kwargs.get("supersample", DEFAULT_SUPERSAMPLE),
        hydrolakes=hydrolakes,
        is_canceled=storage_kwargs.get("is_canceled"),
    )
    z = np.asarray(dem_window, dtype=np.float64)

    def burned(scenario):
        bed = bathy.beds[scenario]
        return np.where(np.isfinite(bed), bed, z)

    result = compute_storage(
        burned("central"),
        dem_nodata,
        window_gt,
        window,
        area,
        interval,
        **storage_kwargs,
    )
    light = dict(storage_kwargs)
    light["polygon_interval"] = 1e9  # no outlines needed for the range runs
    light["contour_interval"] = 1e9
    stages = np.array([row["stage_m"] for row in result.table], dtype=float)
    for scenario in ("low", "high"):
        try:
            other = compute_storage(
                burned(scenario),
                dem_nodata,
                window_gt,
                window,
                area,
                interval,
                **light,
            )
        except StorageError:
            continue
        result.range_tables[scenario] = other.table
        h = np.array([row["stage_m"] for row in other.table], dtype=float)
        v = np.array([row["volume_m3"] for row in other.table], dtype=float)
        a = np.array([row["area_m2"] for row in other.table], dtype=float)
        vol = np.interp(stages, h, v, left=0.0)
        are = np.interp(stages, h, a, left=0.0)
        for row, vi, ai in zip(result.table, vol, are, strict=True):
            row[f"volume_{scenario}_m3"] = round(float(vi), 3)
            row[f"area_{scenario}_m2"] = round(float(ai), 3)
    result.bathymetry = bathy
    result.warnings = list(bathy.warnings) + list(result.warnings)
    ba = bathy.attributes
    result.attributes.update(
        {
            "z_survey": ba["water_lvl"],
            "bed_basis": ba["basis"],
            "ws_ha": ba["water_ha"],
            "vws_lo": ba["vol_lo"],
            "vws_c": ba["vol_c"],
            "vws_hi": ba["vol_hi"],
            "vol_lo": result.table[-1].get("volume_low_m3"),
            "vol_hi": result.table[-1].get("volume_high_m3"),
        }
    )
    return result


def mosaic_depths(
    results: Sequence[StorageResult], nodata: float = DEFAULT_DEPTH_NODATA
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Merges every result's top-stage depth window into one grid covering
    them all (deepest value where windows overlap). Returns (array with
    ``nodata`` for dry cells, (row0, col0) of the array in the DEM), or
    None when there is nothing to write."""
    results = [r for r in results if r.depth.size]
    if not results:
        return None
    r0 = min(r.window[0] for r in results)
    c0 = min(r.window[1] for r in results)
    r1 = max(r.window[0] + r.window[2] for r in results)
    c1 = max(r.window[1] + r.window[3] for r in results)
    out = np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)
    for r in results:
        wr, wc, nr, nc = r.window
        view = out[wr - r0 : wr - r0 + nr, wc - c0 : wc - c0 + nc]
        np.fmax(view, r.depth, out=view)
    out = np.where(np.isnan(out), nodata, out).astype(np.float32)
    return out, (r0, c0)


# ---------------------------------------------------------------------------
# File I/O (GDAL - bundled with QGIS)
# ---------------------------------------------------------------------------


def write_raster(
    path: str, array: np.ndarray, geotransform, projection: str, nodata: float
) -> None:
    """Writes a float32 GeoTIFF (LZW compressed)."""
    from osgeo import gdal

    gdal.UseExceptions()
    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"]
    )
    out_ds.SetGeoTransform(tuple(geotransform))
    out_ds.SetProjection(projection)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(array.astype(np.float32))
    band.SetNoDataValue(float(nodata))
    band.FlushCache()
    out_ds = None
