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
