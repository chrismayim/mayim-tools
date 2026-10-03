"""
Cell-boundary polygon helpers shared by the Hydrological Tools - pure
NumPy, no QGIS dependency.

Traces the cell outline of a boolean grid mask into closed rings (outer
rings anticlockwise on the map, holes clockwise), fills enclosed holes,
groups rings into OGC-valid polygons and converts them to map
coordinates / WKT.

Moved verbatim from hydrology/catchment_delineation/core.py (which still
re-exports every name for backward compatibility) so that Catchment
Delineation and Terrain Storage share one tested implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MAX_HOLE_FILL_PASSES = 20


def _ring_signed_area_idx(ring: Sequence[tuple[int, int]]) -> float:
    """Shoelace area in index space (x = col, y = row, y pointing down).
    Outer rings (anticlockwise on the map) come out NEGATIVE here."""
    a = np.asarray(ring, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def trace_rings(
    mask: np.ndarray, join_diagonals: bool = False
) -> list[list[tuple[int, int]]]:
    """Traces the cell boundaries of a bool mask into closed rings of
    corner vertices (x = col, y = row, both integer), with the mask kept
    on the left when viewed on the map (outer rings anticlockwise, hole
    rings clockwise).

    Where cells touch only at a corner, the default splits them (left
    turn), which gives rings that form a valid MultiPolygon.
    ``join_diagonals=True`` turns right instead, treating the mask as
    8-connected - used to find every enclosed hole."""
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    inner = m[1:-1, 1:-1]
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    # bottom edge, heading east
    r, c = np.nonzero(inner & ~m[2:, 1:-1])
    starts.append(np.stack([c, r + 1], 1))
    ends.append(np.stack([c + 1, r + 1], 1))
    # right edge, heading north
    r, c = np.nonzero(inner & ~m[1:-1, 2:])
    starts.append(np.stack([c + 1, r + 1], 1))
    ends.append(np.stack([c + 1, r], 1))
    # top edge, heading west
    r, c = np.nonzero(inner & ~m[:-2, 1:-1])
    starts.append(np.stack([c + 1, r], 1))
    ends.append(np.stack([c, r], 1))
    # left edge, heading south
    r, c = np.nonzero(inner & ~m[1:-1, :-2])
    starts.append(np.stack([c, r], 1))
    ends.append(np.stack([c, r + 1], 1))

    s = np.concatenate(starts).tolist()
    e = np.concatenate(ends).tolist()
    out_edges: dict[tuple[int, int], list[int]] = {}
    for i, v in enumerate(s):
        out_edges.setdefault(tuple(v), []).append(i)

    used = [False] * len(s)
    rings = []
    for first in range(len(s)):
        if used[first]:
            continue
        ring = [tuple(s[first])]
        cur = first
        while True:
            used[cur] = True
            end = tuple(e[cur])
            ring.append(end)
            options = out_edges[end]
            if len(options) == 1:
                nxt_edge = options[0]
            else:
                dx, dy = end[0] - s[cur][0], end[1] - s[cur][1]
                # left (split) or right (join) turn on the map, in index space
                turn = (-dy, dx) if join_diagonals else (dy, -dx)
                nxt_edge = next(
                    o for o in options if (e[o][0] - s[o][0], e[o][1] - s[o][1]) == turn
                )
            if nxt_edge == first:
                break
            cur = nxt_edge
        rings.append(ring)
    return rings


def _winding_fill(rings, shape) -> np.ndarray:
    """Cells enclosed by any of the rings (non-zero winding)."""
    rows, cols = shape
    toggle = np.zeros((rows, cols + 1), dtype=np.int32)
    for ring in rings:
        a = np.asarray(ring, dtype=np.int64)
        x0, y0, y1 = a[:-1, 0], a[:-1, 1], a[1:, 1]
        vertical = y0 != y1
        sign = np.where(y1[vertical] > y0[vertical], 1, -1)
        np.add.at(
            toggle, (np.minimum(y0, y1)[vertical], x0[vertical]), sign.astype(np.int32)
        )
    return np.cumsum(toggle, axis=1)[:, :cols] != 0


def fill_holes(
    mask: np.ndarray, keep_out: np.ndarray | None = None
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Fills interior holes: background cells that are not 4-connected to
    the outside, i.e. enclosed by the (8-connected) catchment - including
    holes closed off only by corner contacts. Cells flagged in
    ``keep_out`` (e.g. a nested upstream sub-catchment in incremental
    mode) are never filled and stay as real holes.

    Returns (filled_mask, rings_of_filled_mask): outer rings anticlockwise
    and any remaining hole rings clockwise on the map, diagonal contacts
    split - see group_polygons()."""
    filled = np.asarray(mask, dtype=bool).copy()
    allowed = None if keep_out is None else ~np.asarray(keep_out, dtype=bool)
    for _ in range(MAX_HOLE_FILL_PASSES):
        rings = trace_rings(filled, join_diagonals=True)
        holes = [r for r in rings if _ring_signed_area_idx(r) > 0]
        if not holes:
            break
        inside = _winding_fill(holes, filled.shape)
        if allowed is not None:
            inside &= allowed
        if not (inside & ~filled).any():
            break
        filled |= inside
    return filled, trace_rings(filled)


def _point_in_ring(px: float, py: float, ring) -> bool:
    """Even-odd ray casting."""
    a = np.asarray(ring, dtype=np.float64)
    x0, y0, x1, y1 = a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1]
    crosses = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.sum(crosses & (px < xc)) % 2)


def split_ring(ring) -> list[list[tuple[int, int]]]:
    """Splits a closed ring that passes through the same vertex more than
    once into simple closed loops (they touch only at those vertices)."""
    loops = []
    path: list[tuple[int, int]] = []
    position: dict[tuple[int, int], int] = {}
    for v in ring[:-1]:
        if v in position:
            i = position[v]
            loop = path[i:] + [v]
            loops.append(loop)
            for u in path[i + 1 :]:
                position.pop(u, None)
            path = path[: i + 1]
        else:
            position[v] = len(path)
            path.append(v)
    loops.append(path + [path[0]])
    return [lp for lp in loops if len(lp) >= 4]


def group_polygons(rings) -> list[list[list[tuple[int, int]]]]:
    """Groups traced rings into polygons: [outer, hole, hole, ...] each.
    Self-touching rings are first split into simple loops, so every ring
    is simple (OGC-valid). Every hole goes to the smallest outer ring that
    contains it."""
    rings = [loop for r in rings for loop in split_ring(r)]
    outers = [r for r in rings if _ring_signed_area_idx(r) < 0]
    holes = [r for r in rings if _ring_signed_area_idx(r) > 0]
    polygons = [[r] for r in outers]
    areas = [-_ring_signed_area_idx(r) for r in outers]
    for hole in holes:
        (x0, y0), (x1, y1) = hole[0], hole[1]
        dx, dy = x1 - x0, y1 - y0
        # a point just inside the hole (to the right of the edge on the map)
        px, py = 0.5 * (x0 + x1) - 0.25 * dy, 0.5 * (y0 + y1) + 0.25 * dx
        owners = [i for i, o in enumerate(outers) if _point_in_ring(px, py, o)]
        if owners:
            polygons[min(owners, key=lambda i: areas[i])].append(hole)
    return polygons


def simplify_ring(ring: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drops collinear vertices from a closed rectilinear ring."""
    pts = list(ring[:-1])
    n = len(pts)
    keep = []
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        d1 = (p1[0] - p0[0], p1[1] - p0[1])
        d2 = (p2[0] - p1[0], p2[1] - p1[1])
        if d1[0] * d2[1] - d1[1] * d2[0] != 0:
            keep.append(p1)
    keep.append(keep[0])
    return keep


def ring_to_map(ring, geotransform, row_off: int = 0, col_off: int = 0):
    return [
        (
            geotransform[0] + (col_off + vx) * geotransform[1],
            geotransform[3] + (row_off + vy) * geotransform[5],
        )
        for vx, vy in ring
    ]


def rings_to_multipolygon_wkt(polygons_map) -> str:
    """WKT for a list of polygons, each a list of rings (outer first)."""
    parts = []
    for polygon in polygons_map:
        rings = []
        for ring in polygon:
            rings.append("(" + ", ".join(f"{x!r} {y!r}" for x, y in ring) + ")")
        parts.append("(" + ", ".join(rings) + ")")
    return "MULTIPOLYGON (" + ", ".join(parts) + ")"


def ring_length(ring_map) -> float:
    a = np.asarray(ring_map, dtype=np.float64)
    return float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))
