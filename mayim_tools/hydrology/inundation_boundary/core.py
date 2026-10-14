"""
Core inundation-boundary logic - pure NumPy, no QGIS dependency (GDAL is
imported only by the shared raster I/O helpers).

PURPOSE
The extent of water standing at a given elevation ("bathtub" inundation),
for one level or a range of levels, as smoothed boundary lines and
matching polygons, with the area, volume and depth at each level.

METHOD
- Inundated ground at level h:
  - ALL mode: every cell below h inside the bounding area (separate hollows
    included - a static bathtub);
  - CONNECTED mode: cells joined below h to a seed point (river, dam, outlet).
    A priority flood from each seed gives every cell the lowest level at
    which water from the seed reaches it (the minimax path level,
    8-connected); a cell is wet when that level is below h.
- Boundary: marching squares on that field at h (crossings interpolated
  between cell centres), rounded with Chaikin corner cutting. Cells
  outside the bounding polygon (and NoData) are set "just dry", so the
  water's edge runs half a cell outside the bounding polygon there and is
  cut exactly at the polygon edge when clipped.
- Polygons are assembled from the same closed lines by nesting (outer
  ring = water, next ring in = dry island, and so on), so line and polygon
  match exactly. Closed rings smaller than a few cells (DEM noise) are
  dropped.
- Area, volume (prism method, weighted by the share of each cell inside
  the bounding polygon) and maximum depth per level are counted on the
  cells; the depth raster is for the highest level.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.contours import (
    chaikin,
    index_to_map,
    marching_squares,
)
from mayim_tools.hydrology._common.d8_network import GridError, nodata_mask
from mayim_tools.hydrology._common.depression import (
    coverage_fraction,
    priority_flood_levels,
)

MODES = ("all", "connected")
MAX_LEVELS = 1000
DEFAULT_MIN_CELLS = 3.0
DEFAULT_SMOOTHING = 3
DRY_EPS = 1e-6
DEFAULT_NODATA = -9999.0


class InundationError(GridError):
    """Raised for invalid inputs (bad levels, no valid cells, ...)."""


@dataclass
class LevelResult:
    level: float
    lines: list  # closed rings in map coordinates [(x, y), ...]
    polygons: list  # [[outer, hole, ...], ...] in map coordinates
    area_m2: float
    volume_m3: float
    max_depth: float
    wet_cells: int


@dataclass
class InundationResult:
    levels: list[LevelResult]
    depth: np.ndarray  # float32 window, NaN where dry (highest level)
    depth_level: float
    window: tuple
    window_gt: tuple
    mode: str
    n_seeds: int
    warnings: list[str] = field(default_factory=list)
    z: np.ndarray | None = None  # DEM window (float64)
    fill: np.ndarray | None = None  # inundation level per cell
    seed_ground: list[float] = field(default_factory=list)

    def wet_at(self, level: float) -> np.ndarray:
        return self.fill < level

    def depth_at(self, level: float) -> np.ndarray:
        """Water depth (float32, NaN where dry) at ``level``."""
        return np.where(self.wet_at(level), level - self.z, np.nan).astype(np.float32)

    def water_surface_at(self, level: float) -> np.ndarray:
        """Water-surface elevation (float32, NaN where dry): ``level`` on
        every inundated cell."""
        return np.where(self.wet_at(level), level, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# Levels and seeds
# ---------------------------------------------------------------------------


def water_levels(
    start: float, end: float | None = None, interval: float | None = None
) -> list[float]:
    """One level, or start, start + interval, ... up to and including end
    (end is always included)."""
    if end is None or abs(end - start) < 1e-9:
        return [float(start)]
    if end < start:
        start, end = end, start
    if not interval or interval <= 0:
        raise InundationError("A level range needs an interval greater than zero.")
    n = (end - start) / interval
    if n > MAX_LEVELS:
        raise InundationError(
            f"{n:.0f} levels requested (more than {MAX_LEVELS}) - use a larger "
            "interval."
        )
    levels = []
    k = 0
    while True:
        v = round(start + k * interval, 9)
        if v >= end - 1e-9:
            break
        levels.append(v)
        k += 1
    levels.append(float(end))
    return levels


def seed_cells(points, geotransform, domain: np.ndarray):
    """Cells (row, col) of seed points (map coordinates) that fall on valid
    cells of the domain; returns (cells, n_outside)."""
    rows, cols = domain.shape
    cells, outside = [], 0
    for x, y in points:
        c = math.floor((x - geotransform[0]) / geotransform[1])
        r = math.floor((y - geotransform[3]) / geotransform[5])
        if 0 <= r < rows and 0 <= c < cols and domain[r, c]:
            if (r, c) not in cells:
                cells.append((r, c))
        else:
            outside += 1
    return cells, outside


def fill_field(
    z: np.ndarray,
    domain: np.ndarray,
    mode: str,
    seeds: Sequence[tuple[int, int]] = (),
    is_canceled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Level at which each cell is inundated: its elevation (ALL mode) or
    the lowest level at which water from any seed reaches it (CONNECTED);
    +inf outside the domain or where no seed reaches."""
    if mode not in MODES:
        raise InundationError(f"Unknown mode '{mode}' (expected {MODES}).")
    if mode == "all":
        return np.where(domain, z, np.inf)
    out = np.full(z.shape, np.inf)
    for seed in seeds:
        levels, _ = priority_flood_levels(z, domain, seed, is_canceled=is_canceled)
        np.minimum(out, levels, out=out)
    return out


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def _point_in_ring(px: float, py: float, ring: np.ndarray) -> bool:
    x0, y0 = ring[:-1, 0], ring[:-1, 1]
    x1, y1 = ring[1:, 0], ring[1:, 1]
    crosses = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.sum(crosses & (px < xc)) % 2)


def nest_rings(rings: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
    """Groups non-crossing closed rings into polygons by nesting depth: depth
    0 = water (outer ring), 1 = dry island (hole), 2 = water again, ...
    Returns [[outer, hole, ...], ...]."""
    n = len(rings)
    if n == 0:
        return []
    areas = [_ring_area(r) for r in rings]
    parent = [-1] * n
    depth = [0] * n
    order = sorted(range(n), key=lambda i: areas[i])
    for i in order:
        px, py = rings[i][0]
        containers = [
            j
            for j in range(n)
            if j != i and areas[j] > areas[i] and _point_in_ring(px, py, rings[j])
        ]
        depth[i] = len(containers)
        if containers:
            parent[i] = min(containers, key=lambda j: areas[j])
    polygons = {}
    for i in range(n):
        if depth[i] % 2 == 0:
            polygons[i] = [rings[i]]
    for i in range(n):
        if depth[i] % 2 == 1 and parent[i] in polygons:
            polygons[parent[i]].append(rings[i])
    return [polygons[k] for k in sorted(polygons, key=lambda k: -areas[k])]


def level_boundaries(
    fill: np.ndarray,
    level: float,
    geotransform,
    smoothing: int = DEFAULT_SMOOTHING,
    min_cells: float = DEFAULT_MIN_CELLS,
):
    """Smoothed water's-edge rings at ``level`` and the polygons built from
    them. Cells with an infinite fill level are set just dry (level +
    DRY_EPS), so every ring closes - outside a bounding polygon half a cell
    beyond its edge, to be clipped there."""
    f = np.where(np.isfinite(fill), fill, level + DRY_EPS)
    f = np.pad(f, 1, constant_values=level + DRY_EPS)
    rings_idx = []
    for line in marching_squares(f, level):
        if not np.allclose(line[0], line[-1]) or len(line) < 4:
            continue
        if _ring_area(line[:, ::-1]) < min_cells:
            continue
        smooth = chaikin(line, smoothing) - 1.0  # undo the padding
        rings_idx.append(smooth)
    rings = [np.asarray(index_to_map(r, geotransform)) for r in rings_idx]
    return rings, nest_rings(rings)


def level_statistics(z, fill, weights, level, cell_area):
    wet = fill < level
    w = weights[wet]
    dep = level - z[wet]
    return {
        "area_m2": float(w.sum() * cell_area),
        "volume_m3": float(np.sum(w * dep) * cell_area),
        "max_depth": float(dep.max()) if dep.size else 0.0,
        "wet_cells": int(wet.sum()),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_inundation(
    dem_window: np.ndarray,
    dem_nodata,
    window_gt,
    window: tuple,
    levels: Sequence[float],
    *,
    bounding: list | None = None,
    mode: str = "all",
    seeds: Sequence[tuple[float, float]] = (),
    supersample: int = 4,
    smoothing: int = DEFAULT_SMOOTHING,
    min_cells: float = DEFAULT_MIN_CELLS,
    is_canceled: Callable[[], bool] | None = None,
) -> InundationResult:
    """Inundation boundaries for each level over a DEM window.

    ``bounding``: [[outer, hole, ...], ...] polygons in map coordinates
    limiting the analysis (None = the whole window). ``seeds``: (x, y)
    points for CONNECTED mode (none = the lowest cell)."""
    if not levels:
        raise InundationError("No water level given.")
    z = np.asarray(dem_window, dtype=np.float64)
    sx, sy = abs(float(window_gt[1])), abs(float(window_gt[5]))
    cell_area = sx * sy
    warnings: list[str] = []
    valid = ~nodata_mask(z, dem_nodata)
    if bounding:
        weights = coverage_fraction(bounding, window_gt, z.shape, supersample)
    else:
        weights = np.ones(z.shape)
    domain = (weights > 0) & valid
    if not domain.any():
        raise InundationError(
            "No valid DEM cells inside the bounding polygon (check that it "
            "overlaps the DEM)."
        )
    weights = np.where(domain, weights, 0.0)
    n_seeds = 0
    seed_list: list = []
    seed_ground: list[float] = []
    if mode == "connected":
        seed_list, outside = seed_cells(seeds, window_gt, domain)
        if outside:
            warnings.append(
                f"{outside} seed point(s) lie outside the bounding area or on "
                "NoData and were ignored."
            )
        if not seed_list:
            zd = np.where(domain, z, np.inf)
            seed_list = [
                tuple(int(v) for v in np.unravel_index(int(np.argmin(zd)), z.shape))
            ]
            warnings.append(
                "Connected mode without a usable seed point: the lowest cell "
                f"({float(z[seed_list[0]]):.3f} m) is used as the seed."
            )
        n_seeds = len(seed_list)
        seed_ground = [float(z[c]) for c in seed_list]
    fill = fill_field(z, domain, mode, seed_list, is_canceled)

    zmin = float(z[domain].min())
    zmax = float(z[domain].max())
    results = []
    for h in sorted(levels):
        if is_canceled is not None and is_canceled():
            break
        rings, polygons = level_boundaries(fill, h, window_gt, smoothing, min_cells)
        stats = level_statistics(z, fill, weights, h, cell_area)
        results.append(LevelResult(h, rings, polygons, **stats))
        if stats["wet_cells"] == 0:
            if seed_ground and h <= min(seed_ground):
                warnings.append(
                    f"Level {h:.3f} m: nothing is inundated - the level is not "
                    "above the ground at the seed point(s) (lowest seed cell "
                    f"{min(seed_ground):.3f} m; lowest ground {zmin:.3f} m). "
                    "Water spreads from the seed, so place it on the lowest "
                    "ground of the water body (e.g. the channel bed)."
                )
            else:
                warnings.append(
                    f"Level {h:.3f} m: nothing is inundated (lowest ground "
                    f"{zmin:.3f} m)."
                )
    if max(levels) >= zmax:
        warnings.append(
            f"The highest level ({max(levels):.3f} m) is at or above the highest "
            f"ground in the bounding area ({zmax:.3f} m) - everything is under "
            "water there; widen the bounding polygon if that is unexpected."
        )
    top = max(levels)
    depth = np.where(fill < top, top - z, np.nan).astype(np.float32)
    return InundationResult(
        levels=results,
        depth=depth,
        depth_level=float(top),
        window=tuple(int(v) for v in window),
        window_gt=tuple(window_gt),
        mode=mode,
        n_seeds=n_seeds,
        warnings=warnings,
        z=z,
        fill=fill,
        seed_ground=seed_ground,
    )
