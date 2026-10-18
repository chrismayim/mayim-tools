"""
Contour lines from a gridded field - pure NumPy, shared by the Hydrological
Tools (DEM depression stage-storage water-edge contours, Inundation
boundary).

- marching_squares: contour lines of values sampled at cell centres, with
  crossings interpolated linearly between cell centres; saddles resolved by
  the block-centre mean; lines that leave the grid are open, others closed.
- chaikin: corner-cutting smoothing that keeps closed lines closed and open
  lines' end points fixed.
- index_to_map: cell-index positions -> map coordinates.

Moved verbatim from hydrology/terrain_storage/core.py (which re-exports
the names).
"""

from __future__ import annotations

import numpy as np

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


def dem_contours(
    dem: np.ndarray,
    geotransform,
    interval: float,
    extent: tuple[float, float, float, float] | None = None,
    max_cells: int = 1200,
    max_levels: int = 2000,
) -> tuple[list[tuple[float, list[tuple[float, float]]]], list[str]]:
    """Contour lines of a DEM at every multiple of ``interval`` (m), for map
    display. Only the window covering ``extent`` (xmin, ymin, xmax, ymax;
    None = whole DEM) is contoured, block-averaged so that neither side
    exceeds ``max_cells`` cells. NoData (NaN) is left blank: lines are
    broken where they touch a NoData cell. Returns ([(level, [(x, y),
    ...]), ...], warnings)."""
    warnings: list[str] = []
    if interval is None or interval <= 0:
        return [], warnings
    z = np.asarray(dem, dtype=np.float64)
    gt = list(geotransform)
    rows, cols = z.shape
    if extent is not None:
        xmin, ymin, xmax, ymax = extent
        c0 = int(np.floor((xmin - gt[0]) / gt[1])) - 1
        c1 = int(np.ceil((xmax - gt[0]) / gt[1])) + 1
        r0 = int(np.floor((ymax - gt[3]) / gt[5])) - 1
        r1 = int(np.ceil((ymin - gt[3]) / gt[5])) + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, rows), min(c1, cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            return [], warnings
        z = z[r0:r1, c0:c1]
        gt[0] += c0 * gt[1]
        gt[3] += r0 * gt[5]
    step = max(1, int(np.ceil(max(z.shape) / max_cells)))
    if step > 1:
        h, w = (z.shape[0] // step) * step, (z.shape[1] // step) * step
        blocks = z[:h, :w].reshape(h // step, step, w // step, step)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(blocks).all(axis=(1, 3))
            z = np.where(valid, np.nanmean(blocks, axis=(1, 3)), np.nan)
        gt[1] *= step
        gt[5] *= step
    finite = np.isfinite(z)
    if finite.sum() < 4:
        return [], warnings
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    levels = np.arange(np.ceil(lo / interval), np.floor(hi / interval) + 1) * interval
    if levels.size > max_levels:
        warnings.append(
            f"Contours not drawn: a {interval:g} m interval gives {levels.size} "
            f"contour levels over the {hi - lo:.1f} m relief of the map "
            f"(more than {max_levels}). Use a larger contour interval."
        )
        return [], warnings
    filled = np.where(finite, z, lo - 10.0 * interval)  # NoData far below
    nodata = ~finite
    out = []
    for level in levels:
        for line in marching_squares(filled, float(level)):
            for piece in _split_at_nodata(line, nodata):
                if len(piece) >= 2:
                    out.append((round(float(level), 6), index_to_map(piece, gt)))
    return out, warnings


def _split_at_nodata(line_rc: np.ndarray, nodata: np.ndarray) -> list[np.ndarray]:
    """Breaks a contour (fractional row/col positions) wherever it runs
    between cells of which one is NoData."""
    if not nodata.any():
        return [line_rc]
    rows, cols = nodata.shape
    r = np.asarray(line_rc[:, 0])
    c = np.asarray(line_rc[:, 1])
    r_lo = np.clip(np.floor(r).astype(int), 0, rows - 1)
    r_hi = np.clip(np.ceil(r).astype(int), 0, rows - 1)
    c_lo = np.clip(np.floor(c).astype(int), 0, cols - 1)
    c_hi = np.clip(np.ceil(c).astype(int), 0, cols - 1)
    bad = (
        nodata[r_lo, c_lo]
        | nodata[r_hi, c_hi]
        | nodata[r_lo, c_hi]
        | nodata[r_hi, c_lo]
    )
    pieces, current = [], []
    for i, point in enumerate(line_rc):
        if bad[i]:
            if len(current) >= 2:
                pieces.append(np.asarray(current))
            current = []
        else:
            current.append(point)
    if len(current) >= 2:
        pieces.append(np.asarray(current))
    return pieces
