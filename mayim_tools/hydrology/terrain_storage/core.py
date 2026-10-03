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

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.grid_polygons import (
    group_polygons,
    ring_to_map,
    simplify_ring,
    trace_rings,
)

VALID_MODES = ("connected", "all")
DEFAULT_SUPERSAMPLE = 8
DEFAULT_DEPTH_NODATA = -9999.0
MAX_STAGES = 20000
COVERAGE_ROW_CHUNK = 128
DISCONNECTED_WARN_SHARE = 0.05
TIGHT_EDGE_TOL_M = 0.05
MIN_CONTOUR_CELLS = 3.0  # closed contours enclosing less are DEM noise


class StorageError(ValueError):
    """Raised for invalid inputs (bad grid, bad interval, no storage, ...)."""


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


def window_geotransform(geotransform, row0: int, col0: int) -> tuple:
    gt = list(geotransform)
    gt[0] = geotransform[0] + col0 * geotransform[1]
    gt[3] = geotransform[3] + row0 * geotransform[5]
    return tuple(gt)


def polygon_window(
    polygons, geotransform, shape, pad: int = 1
) -> tuple[int, int, int, int] | None:
    """Cell window (row0, col0, rows, cols) covering the polygon's
    bounding box plus ``pad`` cells, clipped to the raster. None when the
    polygon lies wholly outside the raster."""
    xy = np.asarray([p for poly in polygons for ring in poly for p in ring], float)
    if xy.size == 0:
        return None
    cols_f = (xy[:, 0] - geotransform[0]) / geotransform[1]
    rows_f = (xy[:, 1] - geotransform[3]) / geotransform[5]
    r0 = math.floor(rows_f.min()) - pad
    r1 = math.ceil(rows_f.max()) + pad
    c0 = math.floor(cols_f.min()) - pad
    c1 = math.ceil(cols_f.max()) + pad
    rows, cols = shape
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, rows), min(c1, cols)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, c0, r1 - r0, c1 - c0


# ---------------------------------------------------------------------------
# Polygon -> cell coverage
# ---------------------------------------------------------------------------


def _index_edges(polygons, geotransform) -> np.ndarray:
    """All ring edges as (u0, v0, u1, v1) in fractional cell-index space
    (u = column, v = row, v increasing downwards)."""
    edges = []
    for poly in polygons:
        for ring in poly:
            a = np.asarray(ring, dtype=np.float64)
            if len(a) < 3:
                continue
            if a[0, 0] != a[-1, 0] or a[0, 1] != a[-1, 1]:
                a = np.vstack([a, a[:1]])
            u = (a[:, 0] - geotransform[0]) / geotransform[1]
            v = (a[:, 1] - geotransform[3]) / geotransform[5]
            edges.append(np.stack([u[:-1], v[:-1], u[1:], v[1:]], axis=1))
    if not edges:
        return np.zeros((0, 4))
    return np.concatenate(edges)


def coverage_fraction(
    polygons, geotransform, shape, supersample: int = DEFAULT_SUPERSAMPLE
) -> np.ndarray:
    """Fraction (0..1) of every cell of a ``shape`` grid lying inside the
    polygon(s), from an even-odd scanline fill of supersample x
    supersample sub-cell centres. ``supersample=1`` gives the cell-centre
    rule (0 or 1)."""
    s = int(supersample)
    if s < 1:
        raise StorageError("Supersample factor must be at least 1.")
    rows, cols = shape
    out = np.zeros((rows, cols), dtype=np.float64)
    edges = _index_edges(polygons, geotransform)
    edges = edges[edges[:, 1] != edges[:, 3]]  # horizontal edges never cross
    if edges.size == 0:
        return out
    u0, v0, u1, v1 = edges.T
    v_lo, v_hi = np.minimum(v0, v1), np.maximum(v0, v1)
    # sub-row k has centre v = (k + 0.5) / s; an edge crosses it when
    # v_lo <= v < v_hi (half-open, so shared vertices count once)
    k_lo = np.ceil(v_lo * s - 0.5).astype(np.int64)
    k_hi = np.ceil(v_hi * s - 0.5).astype(np.int64)  # exclusive
    n_sub_cols = cols * s

    for row_start in range(0, rows, COVERAGE_ROW_CHUNK):
        row_end = min(rows, row_start + COVERAGE_ROW_CHUNK)
        ks0, ks1 = row_start * s, row_end * s
        lo = np.maximum(k_lo, ks0)
        hi = np.minimum(k_hi, ks1)
        counts = np.maximum(hi - lo, 0)
        total = int(counts.sum())
        inside = np.zeros((ks1 - ks0, n_sub_cols + 1), dtype=np.int32)
        if total:
            edge_idx = np.repeat(np.arange(len(counts)), counts)
            offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            k = lo[edge_idx] + offsets
            vc = (k + 0.5) / s
            e0u, e0v, e1u, e1v = u0[edge_idx], v0[edge_idx], u1[edge_idx], v1[edge_idx]
            uc = e0u + (vc - e0v) * (e1u - e0u) / (e1v - e0v)
            order = np.lexsort((uc, k))
            k, uc = k[order], uc[order]
            # pair crossings along each sub-row: (1st, 2nd), (3rd, 4th), ...
            starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
            rank = np.arange(len(k)) - np.repeat(starts, np.diff(np.r_[starts, len(k)]))
            enter = rank % 2 == 0
            has_pair = np.r_[k[1:] == k[:-1], False]
            ok = enter & has_pair
            ka = k[ok] - ks0
            j_start = np.ceil(uc[ok] * s - 0.5).astype(np.int64)
            j_end = np.ceil(uc[np.flatnonzero(ok) + 1] * s - 0.5).astype(np.int64)
            j_start = np.clip(j_start, 0, n_sub_cols)
            j_end = np.clip(j_end, 0, n_sub_cols)
            keep = j_end > j_start
            np.add.at(inside, (ka[keep], j_start[keep]), 1)
            np.add.at(inside, (ka[keep], j_end[keep]), -1)
        sub = np.cumsum(inside[:, :n_sub_cols], axis=1) > 0
        block = sub.reshape(row_end - row_start, s, cols, s).mean(axis=(1, 3))
        out[row_start:row_end] = block
    return out


# ---------------------------------------------------------------------------
# Priority flood: entry levels and spill
# ---------------------------------------------------------------------------

_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def edge_cells(domain: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    """Domain cells with an 8-neighbour in ``exterior`` or off the grid."""
    ext = np.pad(np.asarray(exterior, dtype=bool), 1, constant_values=True)
    rows, cols = domain.shape
    touch = np.zeros(domain.shape, dtype=bool)
    for dr, dc in _NEIGHBOURS:
        touch |= ext[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    return domain & touch


def priority_flood_levels(
    z: np.ndarray,
    domain: np.ndarray,
    seed: tuple[int, int],
    edge: np.ndarray | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Minimax path level from ``seed`` to every 8-connected ``domain``
    cell: the lowest water level, starting at the seed, at which the
    cell joins the pool (max(z) along the best path). Unreachable cells
    are +inf.

    Also returns the first ``edge`` cell reached - the spill cell, since
    cells are settled in increasing level order (None if no edge cell
    is reachable)."""
    rows, cols = z.shape
    w = cols + 2
    zp = np.pad(np.asarray(z, dtype=np.float64), 1).ravel().tolist()
    dom = np.pad(np.asarray(domain, dtype=bool), 1).ravel()
    edge_flat = (
        np.pad(np.asarray(edge, dtype=bool), 1).ravel().tolist()
        if edge is not None
        else None
    )
    allowed = bytearray(dom.astype(np.uint8).tobytes())
    level = [math.inf] * len(zp)
    offsets = [dr * w + dc for dr, dc in _NEIGHBOURS]
    start = (seed[0] + 1) * w + seed[1] + 1
    if not allowed[start]:
        raise StorageError("Flood seed is not inside the storage area.")
    level[start] = zp[start]
    heap = [(zp[start], start)]
    done = bytearray(len(zp))
    spill_idx = None
    popped = 0
    while heap:
        lv, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = 1
        popped += 1
        if is_canceled is not None and popped % 65536 == 0 and is_canceled():
            break
        if spill_idx is None and edge_flat is not None and edge_flat[i]:
            spill_idx = i
        for off in offsets:
            j = i + off
            if allowed[j] and not done[j]:
                nl = zp[j] if zp[j] > lv else lv
                if nl < level[j]:
                    level[j] = nl
                    heapq.heappush(heap, (nl, j))
    out = np.asarray(level, dtype=np.float64).reshape(rows + 2, cols + 2)[1:-1, 1:-1]
    spill = None
    if spill_idx is not None:
        spill = (spill_idx // w - 1, spill_idx % w - 1)
    return out, spill


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


class StageStorage:
    """Area/volume at any water level for one set of cells, via cumulative
    sums over cells sorted by entry level (O(log N) per stage)."""

    def __init__(self, z, weight, entry, cell_area: float):
        z = np.asarray(z, dtype=np.float64).ravel()
        w = np.asarray(weight, dtype=np.float64).ravel()
        e = np.asarray(entry, dtype=np.float64).ravel()
        order = np.argsort(e, kind="stable")
        self._e = e[order]
        self._cw = np.concatenate([[0.0], np.cumsum(w[order])])
        self._cwz = np.concatenate([[0.0], np.cumsum(w[order] * z[order])])
        self._cn = np.arange(len(e) + 1)
        self.cell_area = float(cell_area)

    def at(self, stage: float) -> tuple[float, float, int]:
        """(plan area m2, volume m3, wet cells) at water level ``stage``."""
        k = int(np.searchsorted(self._e, stage, side="left"))
        area = self.cell_area * self._cw[k]
        volume = self.cell_area * (stage * self._cw[k] - self._cwz[k])
        return float(area), max(float(volume), 0.0), int(self._cn[k])


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


def power_fit(x: Sequence[float], y: Sequence[float]) -> dict | None:
    """Least-squares fit of y = a x^b on log-log axes over the points with
    x > 0 and y > 0 (e.g. volume or area against depth). Returns
    {"a", "b", "r2", "n"} or None with fewer than 3 usable points.

    For volume against depth, b is a shape indicator: about 1 for
    vertical sides, 2 for a wedge/trough and 3 for a cone/bowl."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ok = (xa > 0) & (ya > 0) & np.isfinite(xa) & np.isfinite(ya)
    if ok.sum() < 3:
        return None
    lx, ly = np.log(xa[ok]), np.log(ya[ok])
    if np.ptp(lx) == 0:
        return None
    b, ln_a = np.polyfit(lx, ly, 1)
    pred = ln_a + b * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"a": float(np.exp(ln_a)), "b": float(b), "r2": r2, "n": int(ok.sum())}


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


def raster_info(path: str):
    """(shape, geotransform, projection, nodata) without reading pixels."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    if ds is None:
        raise StorageError(f"Could not open raster: {path}")
    band = ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    info = (
        (ds.RasterYSize, ds.RasterXSize),
        ds.GetGeoTransform(),
        ds.GetProjection(),
        np.nan if nodata_value is None else nodata_value,
    )
    ds = None
    return info


def read_window(path: str, window: tuple[int, int, int, int]) -> np.ndarray:
    """Band 1 pixels for (row0, col0, rows, cols)."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    if ds is None:
        raise StorageError(f"Could not open raster: {path}")
    r0, c0, nr, nc = window
    array = ds.GetRasterBand(1).ReadAsArray(c0, r0, nc, nr)
    ds = None
    return array


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
