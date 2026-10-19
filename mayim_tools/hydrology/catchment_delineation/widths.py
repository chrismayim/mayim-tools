"""
SWMM sub-catchment width estimates - pure NumPy (SciPy for the nearest-line
search), no QGIS dependency.

In SWMM a sub-catchment is a rectangular kinematic-wave plane of area A and
width W; the overland flow length is A / W. Three published estimates are
given side by side, because W cannot be measured and none is right for every
shape (the SWMM manual treats W as a calibration parameter):

- FLOW LENGTH (SWMM Reference Manual, Rossman & Huber 2016):
  W = A / L_o, with L_o the representative overland flow length - here the
  mean D8 flow length from every cell to the nearest stream cell (or to the
  outlet where no stream is reached), optionally capped at a maximum
  overland flow length (e.g. 30 m after NRCS 2010, 150 m after UDFCD).
- CHANNEL (SWMM rule of thumb): W = 2 L for a sub-catchment drained by a
  central channel of length L, and W = L for one bank of a channel (a
  sub-catchment split along its stream centreline drains from one side).
- GUO & URBONAS (2009): the watershed is converted to an equivalent
  kinematic-wave plane that conserves its area and the fall along its
  waterway. With the shape factor X = A / L^2 (L the waterway length) and
  the skew factor Z = A_m / A (A_m the larger of the areas on either side
  of the waterway, 0.5 <= Z <= 1):  W = L (1.5 - Z) (2.286 X - 0.286 X^2).
  The parabola peaks at X = 4, which is used as the upper limit of X (Guo
  & Urbonas recommend 4 to 6; UDFCD uses 4).

The waterway is the stream through a stream sub-catchment, and the longest
flow path otherwise. w_ratio = W(Guo & Urbonas) / W(flow length) flags
shapes far from a rectangle, where calibration should decide.
"""

from __future__ import annotations

import math

import numpy as np

from mayim_tools.hydrology._common.d8_network import cell_center

GU_X_MAX = 4.0  # Guo & Urbonas: upper limit of the shape factor


def guo_urbonas_width(area_m2: float, length_m: float, skew: float):
    """(W in m, shape factor X used) by Guo & Urbonas (2009); (None, None)
    without a waterway length."""
    if not length_m or length_m <= 0 or area_m2 <= 0:
        return None, None
    x = min(area_m2 / length_m**2, GU_X_MAX)
    z = min(max(skew, 0.5), 1.0)
    k = (1.5 - z) * (2.286 * x - 0.286 * x * x)
    return k * length_m, x


def skew_factor(grid, cells: np.ndarray, line_cells: np.ndarray) -> float:
    """Z = larger side area / total area of ``cells`` (flat) about the
    waterway ``line_cells`` (flat, in flow order). Each cell takes the side
    of the nearest line segment; cells on the line count half to each."""
    from scipy.spatial import cKDTree

    cells = np.asarray(cells, dtype=np.int64)
    line_cells = np.asarray(line_cells, dtype=np.int64)
    if cells.size == 0 or line_cells.size < 2:
        return 0.5
    gt, cols = grid.geotransform, grid.cols
    lx, ly = cell_center(line_cells // cols, line_cells % cols, gt)
    xy = np.column_stack([lx, ly]).astype(float)
    cell = min(grid.sx, grid.sy)
    samples, seg_of = [], []
    for i in range(len(xy) - 1):
        d = math.hypot(*(xy[i + 1] - xy[i]))
        n = max(1, math.ceil(d / (0.25 * cell)))
        t = np.arange(n)[:, None] / n
        samples.append(xy[i] + t * (xy[i + 1] - xy[i]))
        seg_of.append(np.full(n, i))
    samples.append(xy[-1:])
    seg_of.append([len(xy) - 2])
    samples = np.vstack(samples)
    seg_of = np.concatenate(seg_of).astype(int)
    px, py = cell_center(cells // cols, cells % cols, gt)
    _, near = cKDTree(samples).query(np.column_stack([px, py]))
    seg = seg_of[near]
    ax, ay = xy[seg, 0], xy[seg, 1]
    bx, by = xy[seg + 1, 0], xy[seg + 1, 1]
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    on = np.isin(cells, line_cells)
    left = np.sum((cross > 0) & ~on) + 0.5 * np.sum(on)
    right = np.sum((cross <= 0) & ~on) + 0.5 * np.sum(on)
    return float(max(left, right) / cells.size)


def width_estimates(
    area_m2: float,
    mean_overland_m: float | None,
    channel_length_m: float | None,
    skew: float,
    one_sided: bool = False,
    max_overland_m: float = 0.0,
) -> dict:
    """The three width estimates (m) and their inputs, rounded for output."""
    lo = mean_overland_m
    if lo is not None and max_overland_m and max_overland_m > 0:
        lo = min(lo, max_overland_m)
    w_flow = area_m2 / lo if lo and lo > 0 else None
    w_chan = None
    if channel_length_m and channel_length_m > 0:
        w_chan = channel_length_m * (1.0 if one_sided else 2.0)
    z = 1.0 if one_sided else skew
    w_gu, x = guo_urbonas_width(area_m2, channel_length_m or 0.0, z)

    def r(v, nd=2):
        return None if v is None else round(float(v), nd)

    return {
        "lo_mean_m": r(mean_overland_m, 2),
        "w_flow_m": r(w_flow, 1),
        "w_chan_m": r(w_chan, 1),
        "w_gu_m": r(w_gu, 1),
        "gu_x": r(x, 4),
        "gu_z": r(z, 3),
        "w_ratio": r(w_gu / w_flow if w_gu and w_flow else None, 3),
    }
