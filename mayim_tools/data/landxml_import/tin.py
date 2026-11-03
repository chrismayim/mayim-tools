"""
TIN outputs for the Import LandXML Surface tool (NumPy only; no QGIS).

- tin_to_grid: the surface sampled onto a regular grid by linear
  interpolation within each triangle (the TIN's own surface, so the grid
  matches the TIN at every vertex; no smoothing). Cells whose centre is
  outside every visible triangle are NoData.
- write_2dm: the TIN as a 2DM mesh file (nodes with their elevation,
  triangles), which QGIS opens as a mesh layer.
"""

from __future__ import annotations

import math

import numpy as np

NODATA = -9999.0


def grid_geometry(extent, cell: float):
    """(geotransform, rows, cols) of a north-up grid snapped to multiples of
    ``cell`` that covers ``extent`` (xmin, ymin, xmax, ymax)."""
    if cell <= 0:
        raise ValueError("the cell size must be greater than 0")
    xmin, ymin, xmax, ymax = extent
    x0 = math.floor(xmin / cell) * cell
    y1 = math.ceil(ymax / cell) * cell
    cols = max(1, int(math.ceil((xmax - x0) / cell - 1e-9)))
    rows = max(1, int(math.ceil((y1 - ymin) / cell - 1e-9)))
    return (x0, cell, 0.0, y1, 0.0, -cell), rows, cols


def tin_to_grid(xyz, triangles, cell: float, max_cells: int = 400_000_000):
    """(array float32 with NODATA, geotransform). Raises ValueError when the
    grid would have more than ``max_cells`` cells."""
    xyz = np.asarray(xyz, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64)
    extent = (xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 0].max(), xyz[:, 1].max())
    gt, rows, cols = grid_geometry(extent, cell)
    if rows * cols > max_cells:
        raise ValueError(
            f"a {cell:g} cell size gives {rows} x {cols} cells (more than "
            f"{max_cells}); use a larger cell size"
        )
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    x0, y1 = gt[0], gt[3]
    p = xyz[tri]  # (m, 3, 3)
    for (ax, ay, az), (bx, by, bz), (cx, cy, cz) in p:
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue  # degenerate triangle
        c0 = int(max(0, math.floor((min(ax, bx, cx) - x0) / cell - 0.5)))
        c1 = int(min(cols - 1, math.ceil((max(ax, bx, cx) - x0) / cell - 0.5)))
        r0 = int(max(0, math.floor((y1 - max(ay, by, cy)) / cell - 0.5)))
        r1 = int(min(rows - 1, math.ceil((y1 - min(ay, by, cy)) / cell - 0.5)))
        if c1 < c0 or r1 < r0:
            continue
        xs = x0 + (np.arange(c0, c1 + 1) + 0.5) * cell
        ys = y1 - (np.arange(r0, r1 + 1) + 0.5) * cell
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
        w2 = 1.0 - w0 - w1
        eps = -1e-9
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        z = w0 * az + w1 * bz + w2 * cz
        block = out[r0 : r1 + 1, c0 : c1 + 1]
        block[inside] = z[inside]
    arr = np.where(np.isfinite(out), out, NODATA).astype(np.float32)
    return arr, gt


def write_2dm(path: str, xyz, triangles, name: str = "") -> None:
    """2DM mesh: ND (node: id x y z) and E3T (triangle: id n1 n2 n3 material)
    records, ids starting at 1."""
    xyz = np.asarray(xyz, dtype=float)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("MESH2D\n")
        if name:
            f.write(f'MESHNAME "{name}"\n')
        for i, (x, y, z) in enumerate(xyz, 1):
            f.write(f"ND {i} {x:.4f} {y:.4f} {z:.4f}\n")
        for j, (a, b, c) in enumerate(np.asarray(triangles), 1):
            f.write(f"E3T {j} {a + 1} {b + 1} {c + 1} 1\n")


def outline_rings(xyz, triangles, edges) -> list:
    """The TIN boundary edges ((k, 2) vertex pairs) chained into closed
    rings of (x, y) coordinates (outer boundary and holes)."""
    nxt: dict[int, list] = {}
    for a, b in edges:
        nxt.setdefault(int(a), []).append(int(b))
        nxt.setdefault(int(b), []).append(int(a))
    seen: set = set()
    rings = []
    for a, b in edges:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        ring = [int(a)]
        prev, cur = int(a), int(b)
        seen.add(key)
        while cur != ring[0]:
            ring.append(cur)
            options = [n for n in nxt[cur] if (min(cur, n), max(cur, n)) not in seen]
            if not options:
                break
            prev, cur = cur, options[0]
            seen.add((min(prev, cur), max(prev, cur)))
        ring.append(ring[0])
        if len(ring) >= 4:
            rings.append([(float(xyz[i, 0]), float(xyz[i, 1])) for i in ring])
    return rings
