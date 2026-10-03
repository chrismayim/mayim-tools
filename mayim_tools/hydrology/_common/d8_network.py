"""
D8 flow-network building blocks shared by the Hydrological Tools - pure
NumPy, no QGIS dependency (GDAL is imported only inside the raster I/O
helpers at the bottom).

Pointer encodings (this suite's D8 Flow Direction tool / WhiteboxTools
default, or ESRI)::

    Default: NE=1, E=2, SE=4, S=8, SW=16, W=32, NW=64, N=128
    ESRI:    E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128

0 = pit / no downslope neighbour.

Moved verbatim from hydrology/catchment_delineation/core.py (which still
re-exports every name) so that Catchment Delineation and Stream Network
share one tested implementation of pointer decoding, the downstream
index, pointer-doubling terminals, topological flow accumulation,
outlet snapping, Strahler ordering and stream links.
"""

from __future__ import annotations

import math

import numpy as np

# Direction order: NE, E, SE, S, SW, W, NW, N - identical to this suite's
# d8_flow_direction / d8_flow_accumulation tools.
D_ROW = np.array([-1, 0, 1, 1, 1, 0, -1, -1], dtype=np.int64)
D_COL = np.array([1, 1, 1, 0, -1, -1, -1, 0], dtype=np.int64)

POINTER_VALS_DEFAULT = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int64)
POINTER_VALS_ESRI = np.array([128, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)

MAX_DOUBLING_ITERATIONS = 64


class GridError(ValueError):
    """Raised for invalid raster inputs (mismatched grids, rotation, etc.)."""


def check_geotransform(geotransform) -> tuple[float, float]:
    """Returns positive (cell_size_x, cell_size_y); rejects rotated grids."""
    if abs(geotransform[2]) > 0 or abs(geotransform[4]) > 0:
        raise GridError(
            "Rotated/sheared rasters are not supported - warp the DEM and "
            "pointer to a north-up grid first."
        )
    sx, sy = abs(float(geotransform[1])), abs(float(geotransform[5]))
    if sx <= 0 or sy <= 0:
        raise GridError("Invalid raster cell size in geotransform.")
    return sx, sy


def check_same_grid(shape_a, gt_a, shape_b, gt_b) -> None:
    """The pointer and DEM must share exactly the same grid."""
    if tuple(shape_a) != tuple(shape_b):
        raise GridError(
            f"D8 pointer ({shape_a[1]} x {shape_a[0]} cells) and DEM "
            f"({shape_b[1]} x {shape_b[0]} cells) are not on the same grid. "
            "Build the pointer from this DEM with the D8 Flow Direction tool."
        )
    tol = 1e-6 * max(abs(gt_a[1]), abs(gt_a[5]))
    for i in range(6):
        if abs(float(gt_a[i]) - float(gt_b[i])) > tol:
            raise GridError(
                "D8 pointer and DEM have different extents or cell sizes. "
                "Build the pointer from this DEM with the D8 Flow Direction tool."
            )


def world_to_cell(x: float, y: float, geotransform) -> tuple[int, int]:
    col = math.floor((x - geotransform[0]) / geotransform[1])
    row = math.floor((y - geotransform[3]) / geotransform[5])
    return int(row), int(col)


def cell_center(row, col, geotransform):
    x = geotransform[0] + (np.asarray(col) + 0.5) * geotransform[1]
    y = geotransform[3] + (np.asarray(row) + 0.5) * geotransform[5]
    return x, y


def nodata_mask(array: np.ndarray, nodata_value) -> np.ndarray:
    mask = np.isnan(array) if np.issubdtype(array.dtype, np.floating) else None
    if mask is None:
        mask = np.zeros(array.shape, dtype=bool)
    if nodata_value is not None and not np.isnan(nodata_value):
        mask |= array == nodata_value
    return mask


def pointer_to_direction_index(
    pointer: np.ndarray, pointer_nodata, esri_style: bool = False
) -> tuple[np.ndarray, int]:
    """Maps pointer codes to 0-7 (D_ROW/D_COL order), -1 (pit, code 0)
    or -2 (NoData/invalid). Returns (direction_index, n_invalid)."""
    pointer = np.asarray(pointer)
    pointer_vals = POINTER_VALS_ESRI if esri_style else POINTER_VALS_DEFAULT
    is_nodata = nodata_mask(pointer.astype(np.float64), pointer_nodata)

    direction_index = np.full(pointer.shape, -2, dtype=np.int8)
    recognised = np.zeros(pointer.shape, dtype=bool)
    for i, val in enumerate(pointer_vals):
        hit = (pointer == val) & ~is_nodata
        direction_index[hit] = i
        recognised |= hit
    pit = (pointer == 0) & ~is_nodata
    direction_index[pit] = -1
    recognised |= pit

    invalid = ~recognised & ~is_nodata
    return direction_index, int(invalid.sum())


def build_downstream_index(direction_index: np.ndarray) -> np.ndarray:
    """Flat downstream index per cell. A cell is its own downstream
    (a terminal) if it is a pit, NoData, or drains off the grid or into
    NoData."""
    rows, cols = direction_index.shape
    n = rows * cols
    idx = np.arange(n, dtype=np.int64)
    nxt = idx.copy()
    flat_dir = direction_index.ravel()
    src = idx[flat_dir >= 0]
    d = flat_dir[src].astype(np.int64)
    rr = src // cols + D_ROW[d]
    cc = src % cols + D_COL[d]
    inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    tgt = rr[inside] * cols + cc[inside]
    ok = flat_dir[tgt] != -2
    nxt[src[inside][ok]] = tgt[ok]
    return nxt


def step_lengths(direction_index: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Flat D8 step length (m) from each cell to its downstream cell."""
    diag = math.hypot(sx, sy)
    # NE, E, SE, S, SW, W, NW, N
    per_dir = np.array([diag, sx, diag, sy, diag, sx, diag, sy])
    flat_dir = direction_index.ravel()
    out = np.zeros(flat_dir.shape, dtype=np.float64)
    valid = flat_dir >= 0
    out[valid] = per_dir[flat_dir[valid]]
    return out


def resolve_terminals(
    nxt: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Pointer doubling: for every cell, the terminal it drains to and
    (optionally) the summed weights along the way.

    Terminals must satisfy nxt[t] == t and must carry weight 0 (the
    caller zeroes them). Returns (terminal, distance, unresolved) where
    ``unresolved`` flags cells that never reach a terminal - only
    possible when the pointer contains a flow loop."""
    jump = nxt.copy()
    dist = None
    if weights is not None:
        dist = np.asarray(weights, dtype=np.float64).copy()
        dist[jump == np.arange(jump.size)] = 0.0
    for _ in range(MAX_DOUBLING_ITERATIONS):
        new_jump = jump[jump]
        if dist is not None:
            dist = dist + dist[jump]
        if np.array_equal(new_jump, jump):
            break
        jump = new_jump
    unresolved = nxt[jump] != jump  # not a genuine terminal: a flow loop
    return jump, dist, unresolved


def flow_accumulation_cells(
    nxt: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, int]:
    """Upstream cell count (each cell counts itself, as in WhiteboxTools).
    ``valid`` is a flat bool array. Returns (accumulation, n_loop_cells)."""
    n = nxt.size
    idx = np.arange(n, dtype=np.int64)
    moves = nxt != idx
    _, depth, unresolved = resolve_terminals(nxt, moves.astype(np.float64))
    acc = np.where(valid, 1.0, np.nan)

    cells = np.flatnonzero(valid & moves & ~unresolved)
    if cells.size:
        order = cells[np.argsort(-depth[cells], kind="stable")]
        d_sorted = depth[order]
        splits = np.flatnonzero(np.diff(d_sorted)) + 1
        for group in np.split(order, splits):
            np.add.at(acc, nxt[group], acc[group])
    return acc, int((unresolved & valid).sum())


def snap_to_max_accumulation(
    row: int, col: int, acc: np.ndarray, radius_cells: int
) -> tuple[int, int] | None:
    """Cell with the highest accumulation within ``radius_cells`` (circular
    search) of (row, col). Ties go to the nearest cell. None if no valid
    cell is in range."""
    rows, cols = acc.shape
    radius_cells = max(int(radius_cells), 0)
    r0, r1 = max(row - radius_cells, 0), min(row + radius_cells + 1, rows)
    c0, c1 = max(col - radius_cells, 0), min(col + radius_cells + 1, cols)
    if r0 >= r1 or c0 >= c1:
        return None
    window = acc[r0:r1, c0:c1]
    rr, cc = np.mgrid[r0:r1, c0:c1]
    d2 = (rr - row) ** 2 + (cc - col) ** 2
    ok = (d2 <= radius_cells**2) & ~np.isnan(window)
    if not ok.any():
        return None
    best = np.nanmax(np.where(ok, window, -np.inf))
    cand = ok & (window == best)
    pick = np.argmin(np.where(cand, d2, np.iinfo(np.int64).max))
    return int(rr.ravel()[pick]), int(cc.ravel()[pick])


def strahler_order(local_next: np.ndarray, stream: np.ndarray) -> np.ndarray:
    """Strahler (1957) order of every stream cell (0 elsewhere), on a
    network given by ``local_next`` (flat downstream index, terminals point
    to themselves). A cell's order is the largest inflowing order, plus one
    where two or more inflows share that largest order; headwater cells
    are order 1. Cells are processed deepest-first: all inflows of a cell
    sit exactly one D8 step further from the terminal, so they are
    resolved together, before the cell itself."""
    n = local_next.size
    idx = np.arange(n, dtype=np.int64)
    moves = local_next != idx
    _, depth, _ = resolve_terminals(local_next, moves.astype(np.float64))
    order = np.zeros(n, dtype=np.int32)
    max_in = np.zeros(n, dtype=np.int32)
    n_max = np.zeros(n, dtype=np.int32)
    cells = np.flatnonzero(stream)
    if cells.size == 0:
        return order
    cells = cells[np.argsort(-depth[cells], kind="stable")]
    splits = np.flatnonzero(np.diff(depth[cells])) + 1
    for group in np.split(cells, splits):
        mi, nm = max_in[group], n_max[group]
        order[group] = np.where(mi == 0, 1, np.where(nm >= 2, mi + 1, mi))
        mv = group[moves[group]]
        down = local_next[mv]
        ok = stream[down]
        mv, down = mv[ok], down[ok]
        np.maximum.at(max_in, down, order[mv])
        np.add.at(n_max, down, (order[mv] == max_in[down]).astype(np.int32))
    order[~stream] = 0
    return order


def stream_order_summary(
    order: np.ndarray, max_in: np.ndarray | None, local_next, step_len
) -> list[dict]:
    """Per-order stream count (segments) and length (km)."""
    idx = np.arange(order.size)
    moves = local_next != idx
    rows = []
    for u in range(1, int(order.max()) + 1 if order.size else 1):
        cells = order == u
        starts = cells & (max_in < u) if max_in is not None else cells
        rows.append(
            {
                "order": u,
                "n_segments": int(np.sum(starts)),
                "length_km": float(np.sum(step_len[cells & moves])) / 1000.0,
            }
        )
    return rows


def segment_starts(local_next: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Largest inflowing stream order per cell (used to find where a
    segment of a given order begins)."""
    n = local_next.size
    idx = np.arange(n)
    src = np.flatnonzero((order > 0) & (local_next != idx))
    max_in = np.zeros(n, dtype=np.int32)
    down = local_next[src]
    ok = order[down] > 0
    np.maximum.at(max_in, down[ok], order[src][ok])
    return max_in


def stream_links(
    local_next: np.ndarray,
    stream: np.ndarray,
    order: np.ndarray,
    step_len: np.ndarray,
    break_cells: np.ndarray | None = None,
) -> list[dict]:
    """Splits the stream cells into links: runs of cells from a headwater
    or junction down to the next junction (or the outlet). Each link keeps
    one Strahler order. The junction cell is repeated as the link's last
    vertex so consecutive links join up. Cells flagged in ``break_cells``
    (e.g. user pour points) also start a new link. Returns dicts with
    1-based ``link_id``, ``ds_link`` (None at the outlet), ``order``,
    ``cells`` (local flat indices, upstream first), ``last_own`` (the
    link's most downstream own cell) and ``length_m``."""
    idx = np.arange(local_next.size)
    moves = local_next != idx
    src = np.flatnonzero(stream & moves)
    down = local_next[src]
    n_in = np.zeros(local_next.size, dtype=np.int32)
    np.add.at(n_in, down[stream[down]], 1)
    node = n_in != 1
    if break_cells is not None:
        node = node | np.asarray(break_cells, dtype=bool)
    starts = np.flatnonzero(stream & node)
    link_of = {int(s): i + 1 for i, s in enumerate(starts)}
    links = []
    for s in starts:
        cells, length, cur, ds = [int(s)], 0.0, int(s), None
        while True:
            nx = int(local_next[cur])
            if nx == cur or not stream[nx]:
                break
            length += float(step_len[cur])
            cells.append(nx)
            if node[nx]:
                ds = link_of[nx]
                break
            cur = nx
        links.append(
            {
                "link_id": link_of[int(s)],
                "ds_link": ds,
                "order": int(order[s]),
                "cells": cells,
                "last_own": cells[-2] if ds is not None else cells[-1],
                "length_m": length,
            }
        )
    return links


def bifurcation_ratio(summary: list[dict]) -> float | None:
    """Mean of N_u / N_(u+1) over consecutive orders (Strahler 1964)."""
    ratios = [
        a["n_segments"] / b["n_segments"]
        for a, b in zip(summary[:-1], summary[1:], strict=True)
        if a["n_segments"] > 0 and b["n_segments"] > 0
    ]
    return float(np.mean(ratios)) if ratios else None


def read_raster(path: str):
    """First band as an array, plus geotransform, projection and NoData."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    if ds is None:
        raise GridError(f"Could not open raster: {path}")
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = np.nan
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None
    return array, geotransform, projection, nodata_value


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
