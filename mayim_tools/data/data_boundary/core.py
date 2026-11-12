"""
Data Boundary (mask polygon) - geometry core (NumPy / SciPy; no QGIS).

Builds the polygon that encloses a set of data points (survey points, or
the vertices of contours and breaklines densified along the lines):

- concave_polygons(xy, max_edge): Delaunay triangulation of the points;
  triangles with an edge longer than ``max_edge`` are removed, and the
  boundary of the triangles left is the data boundary. Long outer edges
  span areas with no data (bays between contour ends, gaps around the
  site), so removing them makes the boundary hug the data. This is the
  edge-length criterion for a concave hull (the chi-shape of Duckham et
  al. 2008 removes boundary edges longest-first down to a length; an
  alpha shape, Edelsbrunner et al. 1983, uses the circumradius).
- convex_polygon(xy): the convex hull.

A polygon is (shell, [holes]); rings are (n, 2) arrays, closed (first
point repeated), shells counter-clockwise and holes clockwise.
"""

from __future__ import annotations

import numpy as np

TOOL_VERSION = "1.0.0"


def unique_points(xy, tolerance: float = 1e-6) -> np.ndarray:
    """The points with duplicates (within ``tolerance``) removed."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) == 0:
        return xy
    key = np.round(xy / tolerance).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return xy[np.sort(idx)]


def ring_area(ring) -> float:
    """Signed area of a ring (positive counter-clockwise), shoelace."""
    r = np.asarray(ring, dtype=float)
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def polygon_area(poly) -> float:
    """Area of (shell, holes)."""
    shell, holes = poly
    return abs(ring_area(shell)) - sum(abs(ring_area(h)) for h in holes)


def point_in_ring(pt, ring) -> bool:
    """Ray casting."""
    x, y = pt
    r = np.asarray(ring, dtype=float)
    x1, y1 = r[:-1, 0], r[:-1, 1]
    x2, y2 = r[1:, 0], r[1:, 1]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(crosses & (x < xi)) % 2)


def _check(xy):
    if len(xy) < 3:
        raise ValueError(
            "At least 3 separate data points are needed to enclose an area."
        )
    spread = xy - xy.mean(axis=0)
    if np.linalg.matrix_rank(spread, tol=1e-9 * max(1.0, np.abs(spread).max())) < 2:
        raise ValueError(
            "The data points lie on one straight line, so they enclose no "
            "area. Use the buffer method instead."
        )


def convex_polygon(xy):
    """[(shell, [])] of the convex hull."""
    from scipy.spatial import ConvexHull

    xy = unique_points(xy)
    _check(xy)
    hull = ConvexHull(xy)
    ring = xy[hull.vertices]  # counter-clockwise for 2-D input
    ring = np.vstack([ring, ring[:1]])
    if ring_area(ring) < 0:
        ring = ring[::-1]
    return [(ring, [])]


def _triangles(xy):
    from scipy.spatial import Delaunay

    tri = Delaunay(xy).simplices.astype(np.int64)
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    flip = cross < 0  # orient every triangle counter-clockwise
    tri[flip] = tri[flip][:, [0, 2, 1]]
    keep = np.abs(cross) > 1e-12 * max(1.0, float(np.abs(xy).max()) ** 2)
    return tri[keep]


def _edge_lengths(xy, tri):
    p = xy[tri]  # (m, 3, 2)
    d01 = np.hypot(*(p[:, 1] - p[:, 0]).T)
    d12 = np.hypot(*(p[:, 2] - p[:, 1]).T)
    d20 = np.hypot(*(p[:, 0] - p[:, 2]).T)
    return np.column_stack([d01, d12, d20])


def _boundary_rings(xy, tri):
    """Closed rings of the boundary of a set of counter-clockwise
    triangles: shells counter-clockwise, holes clockwise."""
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    fwd = {(int(a), int(b)) for a, b in edges}
    nxt: dict[int, list[int]] = {}
    for a, b in edges:
        a, b = int(a), int(b)
        if (b, a) not in fwd:  # used by one triangle only: on the boundary
            nxt.setdefault(a, []).append(b)
    rings = []
    while nxt:
        start = next(iter(nxt))
        ring = [start]
        cur = start
        while True:
            outs = nxt.get(cur)
            if not outs:
                break
            b = outs.pop()
            if not outs:
                del nxt[cur]
            cur = b
            if cur == start:
                break
            ring.append(cur)
        if len(ring) >= 3:
            pts = xy[ring + [ring[0]]]
            rings.append(pts)
    return rings


def _assemble(rings):
    """[(shell, [holes])]: each clockwise ring goes to the smallest
    counter-clockwise ring that contains it."""
    shells = [r for r in rings if ring_area(r) > 0]
    holes = [r for r in rings if ring_area(r) < 0]
    out = [(s, []) for s in sorted(shells, key=ring_area)]
    for h in holes:
        probe = h[0] + 1e-9 * (h[1] - h[0])
        mid = 0.5 * (h[0] + h[1])
        for s, hs in out:  # smallest shell first
            if point_in_ring(mid, s) or point_in_ring(probe, s):
                hs.append(h)
                break
    out.sort(key=lambda p: -polygon_area(p))
    return out


def concave_polygons(xy, max_edge: float):
    """[(shell, [holes])] of the Delaunay triangles whose edges are all at
    most ``max_edge`` long (largest polygon first). Raises ValueError
    when no triangle is left."""
    if max_edge <= 0:
        raise ValueError("The maximum edge length must be greater than 0.")
    xy = unique_points(xy)
    _check(xy)
    tri = _triangles(xy)
    keep = _edge_lengths(xy, tri).max(axis=1) <= max_edge
    if not keep.any():
        shortest = float(np.min(_edge_lengths(xy, tri).max(axis=1)))
        raise ValueError(
            f"No area is left with a maximum edge length of {max_edge:g} m: the "
            f"data points are further apart. Use at least {shortest:.1f} m "
            "(about 1.5 to 3 times the contour spacing or point spacing)."
        )
    return _assemble(_boundary_rings(xy, tri[keep]))
