"""
Depression / storage-area building blocks shared by the Hydrological Tools
(DEM depression stage-storage, DEM bathymetry estimate) - pure NumPy, no
QGIS dependency (GDAL is imported only inside the raster I/O helpers).

- polygon_window / window_geotransform: DEM window around a polygon.
- coverage_fraction: fraction of each cell inside a polygon (scanline
  even-odd fill of S x S sub-cells; holes and multipart supported).
- edge_cells / priority_flood_levels: minimax ("fill") level from a seed
  cell to every 8-connected cell of a domain, and the first edge cell
  reached (the spill cell).
- StageStorage: area/volume at any water level (prism method).
- power_fit: y = a x^b least-squares fit on log-log axes.
- raster_info / read_window: GDAL metadata and windowed reads.

Moved verbatim from hydrology/terrain_storage/core.py (which re-exports
every name). Errors are raised as d8_network.GridError, of which
terrain_storage's StorageError is an alias.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence

import numpy as np

from mayim_tools.hydrology._common.d8_network import GridError

DEFAULT_SUPERSAMPLE = 8
COVERAGE_ROW_CHUNK = 128


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
        raise GridError("Supersample factor must be at least 1.")
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
        raise GridError("Flood seed is not inside the storage area.")
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


def raster_info(path: str):
    """(shape, geotransform, projection, nodata) without reading pixels."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    if ds is None:
        raise GridError(f"Could not open raster: {path}")
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
        raise GridError(f"Could not open raster: {path}")
    r0, c0, nr, nc = window
    array = ds.GetRasterBand(1).ReadAsArray(c0, r0, nc, nr)
    ds = None
    return array
