"""
Bathymetry estimate for DEM depression stage-storage - pure NumPy, no QGIS
dependency (GDAL only in write_bed_dem; scipy for the distance transform
when available, with a NumPy fallback). Used when the tool's "DEM surface
is a water surface" option is ticked; core.compute_storage_with_bed()
writes the estimated bed into the DEM before the stage-storage curve is
computed.

PROBLEM
LiDAR, photogrammetric and radar DEMs record the WATER SURFACE of pans,
ponds and reservoirs, not the bed. This module estimates the bed below
that surface from the surrounding terrain, as a LOW / CENTRAL / HIGH
range, so that the storage below the surveyed water level can be put
into a stage-storage curve with its uncertainty stated.

STEPS
1. Water surface: the flat area at the bottom of the depression - cells
   joined to the lowest cell (priority flood, 8-connected) below
   floor + flat tolerance. The water level is the median elevation of
   those cells, or a level the user gives.
2. Shoreline-slope method (Hollister et al., 2011): the median slope of
   the land within a buffer of the shoreline, (z - WL) / distance to the
   water, carried inwards: depth = slope x distance to shore. Usually an
   over-estimate for flat-bottomed pans (an upper bound).
3. Hypsometric method (after Heathcote et al., 2015): the water-surface
   area above the water, A(h) for WL < h <= spill, is fitted with
   A(h) = A_w ((h - z_b) / D)^n, D = WL - z_b, constrained to pass through
   the water area A_w at WL. The best D (bed below WL) and n (shape) are
   found by a grid search on D with n solved by least squares. Flagged
   when D sits on a search bound (poorly constrained).
4. Valley (thalweg) method, for a built dam with a stream centreline: the
   river bed upstream of the water (gradient fitted over a reach) and
   below the wall (bed profile fitted beyond the foot of the wall) give
   two estimates of the original bed at the wall; the bed under water is
   joined linearly along the centreline from where the river enters the
   water to the wall, and both banks slope down to it (V-shaped valley).
5. Range:
   - maximum depth supplied: central = it, low/high = 0.8 / 1.2 x;
   - valley method: central = mean of the two wall estimates, low/high =
     the shallower / deeper of them (at least +-15%);
   - otherwise: central = the shallower of methods 2 and 3, high = the
     deeper, low = half of central. With only method 2: central = half of
     it, high = it, low = a quarter.
6. Bed surface (no centreline): water cells are ranked by distance to
   shore and placed on the fitted hypsometry, z = z_b + D r^(1/n), where r
   is the share of the water area at least as far from shore - so the
   area-depth relation follows the fitted shape and the deepest point is
   farthest from the shore. With a centreline: the valley surface above.

Volumes below the water surface use the prism method with the polygon
cell weights, as in DEM depression stage-storage.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.d8_network import GridError, nodata_mask
from mayim_tools.hydrology._common.depression import (
    DEFAULT_SUPERSAMPLE,
    StageStorage,
    coverage_fraction,
    edge_cells,
    priority_flood_levels,
)

SCENARIOS = ("low", "central", "high")
DEFAULT_FLAT_TOL_M = 0.10
MIN_WATER_CELLS = 9
DEFAULT_SHAPE_N = 2.0
USER_DEPTH_RANGE = (0.8, 1.2)
MIN_VALLEY_SPREAD = 0.15
LOW_FACTOR = 0.5
HYPS_MIN_POINTS = 4
HYPS_GRID = 400
WALL_FOOT_SLOPE = 0.10  # bed gradient below which the wall face has ended
DEFAULT_NODATA = -9999.0


class BathymetryError(GridError):
    """Raised when a water body cannot be estimated (no flat water, etc.)."""


@dataclass
class WaterBody:
    """One storage-area polygon (map coordinates, DEM CRS), with optional
    stream centreline(s) and known maximum depth."""

    storage_id: str
    polygons: list
    src_fid: int = -1
    centrelines: list = field(default_factory=list)  # [[(x, y), ...], ...]
    max_depth: float | None = None


@dataclass
class BathymetryResult:
    storage_id: str
    src_fid: int
    water_level: float
    water_area_m2: float
    water_mask: np.ndarray
    window: tuple
    window_gt: tuple
    beds: dict  # scenario -> bed elevation array (window), NaN outside water
    scenarios: dict  # scenario -> {max_depth, mean_depth, volume_m3, ...}
    methods: dict  # method -> {max_depth, ... , status}
    basis: str  # "user", "valley", "slope+hypsometry", "slope"
    attributes: dict
    profile: list = field(default_factory=list)  # centreline profile rows
    warnings: list = field(default_factory=list)


def _opt_float(value):
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def distance_to_false(
    mask: np.ndarray, sx: float, sy: float, off_grid: bool = False
) -> np.ndarray:
    """Euclidean distance (map units) from every True cell centre to the
    nearest False cell centre; 0 on False cells. Cells off the grid take
    the value ``off_grid`` (False = they count as targets). scipy's exact
    transform when available, else a chunked NumPy search over the
    boundary cells."""
    m = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=off_grid)
    try:
        from scipy.ndimage import distance_transform_edt

        d = distance_transform_edt(m, sampling=(sy, sx))
        return np.asarray(d[1:-1, 1:-1], dtype=np.float64)
    except ImportError:
        pass
    out = np.zeros(m.shape, dtype=np.float64)
    true_rc = np.argwhere(m)
    if true_rc.size == 0:
        return out[1:-1, 1:-1]
    # False cells that touch a True cell are the only candidates
    t = m
    touch = np.zeros_like(t)
    rows, cols = t.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                shifted = np.zeros_like(t)
                shifted[
                    max(dr, 0) : rows + min(dr, 0), max(dc, 0) : cols + min(dc, 0)
                ] = t[
                    max(-dr, 0) : rows + min(-dr, 0), max(-dc, 0) : cols + min(-dc, 0)
                ]
                touch |= shifted
    false_rc = np.argwhere(~t & touch).astype(np.float64)
    fy, fx = false_rc[:, 0] * sy, false_rc[:, 1] * sx
    chunk = max(1, 4_000_000 // max(len(false_rc), 1))
    for i in range(0, len(true_rc), chunk):
        rc = true_rc[i : i + chunk]
        dy = rc[:, 0, None] * sy - fy[None, :]
        dx = rc[:, 1, None] * sx - fx[None, :]
        out[rc[:, 0], rc[:, 1]] = np.sqrt((dx * dx + dy * dy).min(axis=1))
    return out[1:-1, 1:-1]


def area_rank(distance: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """For each cell, the share of the total weight held by cells at least
    as far from shore (ties share the same value). In (0, 1]."""
    d = np.asarray(distance, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    order = np.argsort(-d, kind="stable")
    cum = np.cumsum(w[order])
    total = cum[-1] if cum.size else 1.0
    ds = d[order]
    # last index of each tie group
    last = np.r_[ds[1:] != ds[:-1], True]
    idx = np.flatnonzero(last)
    group_end = idx[np.searchsorted(idx, np.arange(len(ds)))]
    r = np.empty_like(d)
    r[order] = cum[group_end] / total
    return np.clip(r, 1e-12, 1.0)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def shoreline_slope(
    z: np.ndarray,
    land: np.ndarray,
    dist_to_water: np.ndarray,
    water_level: float,
    buffer_m: float,
    half_cell: float,
) -> dict | None:
    """Median land slope (z - WL) / distance from the water's edge, over
    land cells within ``buffer_m`` of the water. None if too few cells."""
    edge_dist = dist_to_water - half_cell
    ring = land & (dist_to_water > 0) & (edge_dist <= buffer_m) & (z > water_level)
    if ring.sum() < 3:
        return None
    slopes = (z[ring] - water_level) / np.maximum(edge_dist[ring], half_cell)
    return {
        "slope": float(np.median(slopes)),
        "slope_p25": float(np.percentile(slopes, 25)),
        "slope_p75": float(np.percentile(slopes, 75)),
        "n_cells": int(ring.sum()),
        "buffer_m": float(buffer_m),
    }


def fit_hypsometry(
    heights: Sequence[float],
    areas: Sequence[float],
    ref_area: float,
    d_min: float,
    d_max: float,
    ref_height: float = 0.0,
) -> dict | None:
    """Fits A(h) = A_ref ((h + D) / (h_ref + D))^n to areas at heights h
    above the water level (h > h_ref), anchored at the area A_ref at
    h_ref (the water area at the flat-water tolerance, so that cells just
    above the water level are counted the same way on both sides). D is
    the depth of the bed below the water level; log-spaced grid search on
    D, n by least squares through the origin in log space. Returns
    {"depth", "n", "rmse", "bound", "n_points"} or None with fewer than
    HYPS_MIN_POINTS usable points."""
    h = np.asarray(heights, dtype=np.float64)
    a = np.asarray(areas, dtype=np.float64)
    ok = (h > ref_height) & (a > ref_area * 1.0001)
    if ok.sum() < HYPS_MIN_POINTS or ref_area <= 0 or d_max <= d_min:
        return None
    h, y = h[ok], np.log(a[ok] / ref_area)
    best = None
    grid = np.geomspace(d_min, d_max, HYPS_GRID)
    for depth in grid:
        x = np.log((h + depth) / (ref_height + depth))
        n = float(np.dot(x, y) / np.dot(x, x))
        if n <= 0:
            continue
        rss = float(np.sum((y - n * x) ** 2))
        if best is None or rss < best[0]:
            best = (rss, float(depth), n)
    if best is None:
        return None
    rss, depth, n = best
    bound = None
    if depth <= grid[1]:
        bound = "lower"
    elif depth >= grid[-2]:
        bound = "upper"
    return {
        "depth": depth,
        "n": n,
        "rmse": math.sqrt(rss / len(h)),
        "bound": bound,
        "n_points": int(len(h)),
    }


def sample_line(points: Sequence[tuple[float, float]], step: float) -> np.ndarray:
    """Points every ``step`` along a polyline: array of (x, y, chainage)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return np.zeros((0, 3))
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] <= 0:
        return np.zeros((0, 3))
    s = np.arange(0.0, cum[-1] + 1e-9, step)
    x = np.interp(s, cum, p[:, 0])
    y = np.interp(s, cum, p[:, 1])
    return np.column_stack([x, y, s])


def _linear_fit(d, z):
    d, z = np.asarray(d, float), np.asarray(z, float)
    if len(d) < 3 or np.ptp(d) <= 0:
        return None
    slope, intercept = np.polyfit(d, z, 1)
    return float(intercept), float(slope)


def wall_toe_from_downstream(
    d: np.ndarray, z: np.ndarray, water_level: float, step: float
):
    """Original bed level at the upstream face of the wall from the river
    profile below it. ``d`` = distance downstream from the water's edge.
    Skips the wall face (until the bed gradient falls below
    WALL_FOOT_SLOPE and the ground is below the water level), fits a line
    to the bed beyond and evaluates it at d = 0. Returns
    (z_at_wall, gradient, foot_distance) or None."""
    if len(d) < 4:
        return None
    window = max(2, min(int(round(3.0 * 2.0)), len(d) // 3))  # ~3 cells
    foot = None
    for i in range(len(d) - window):
        dz = (z[i + window] - z[i]) / max(d[i + window] - d[i], 1e-9)
        if abs(dz) < WALL_FOOT_SLOPE and z[i] < water_level:
            foot = i
            break
    if foot is None:
        return None
    fit = _linear_fit(d[foot:], z[foot:])
    if fit is None:
        return None
    intercept, gradient = fit
    # the wall face hides the bed between d = 0 and the foot: extend the bed
    # line back to the water's edge (the upstream face)
    return min(intercept, water_level), gradient, float(d[foot])


def valley_profile(
    samples: np.ndarray,
    z_samples: np.ndarray,
    in_water: np.ndarray,
    water_level: float,
    reach_m: float,
    step: float,
) -> dict | None:
    """Analyses one centreline: finds the stretch under water, decides the
    flow direction (the end with the lower ground outside is downstream),
    fits the upstream bed gradient and the bed below the wall, and returns
    the two estimates of the bed at the wall plus the profile rows."""
    idx = np.flatnonzero(in_water)
    if idx.size < 2:
        return None
    i0, i1 = int(idx[0]), int(idx[-1])
    s = samples[:, 2]
    valid = np.isfinite(z_samples)

    def outside(lo, hi, forward):
        sel = np.arange(lo, hi)
        sel = sel[valid[sel] & ~in_water[sel]]
        dist = (s[sel] - s[lo]) if forward else (s[hi - 1] - s[sel])
        keep = dist <= reach_m
        sel, dist = sel[keep], dist[keep]
        order = np.argsort(dist)
        return sel[order], dist[order]

    before, d_before = outside(0, i0, forward=False)
    after, d_after = outside(i1 + 1, len(s), forward=True)
    if before.size == 0 and after.size == 0:
        return None
    min_before = z_samples[before].min() if before.size else np.inf
    min_after = z_samples[after].min() if after.size else np.inf
    downstream_after = min_after <= min_before
    if downstream_after:
        up_sel, up_d, dn_sel, dn_d = before, d_before, after, d_after
        s_entry, s_exit = s[i0], s[i1]
    else:
        up_sel, up_d, dn_sel, dn_d = after, d_after, before, d_before
        s_entry, s_exit = s[i1], s[i0]
    length_in = abs(s_exit - s_entry)

    estimates = {}
    up_fit = _linear_fit(up_d, z_samples[up_sel]) if up_sel.size else None
    z_entry = water_level
    gradient_up = None
    if up_fit is not None:
        intercept, slope_up = up_fit  # slope_up > 0: ground rises upstream
        gradient_up = slope_up
        z_entry = min(intercept, water_level)
        if slope_up > 0:
            estimates["upstream"] = min(intercept - slope_up * length_in, water_level)
    toe = None
    if dn_sel.size:
        toe = wall_toe_from_downstream(dn_d, z_samples[dn_sel], water_level, step)
        if toe is not None:
            estimates["downstream"] = toe[0]
    if not estimates:
        return None
    return {
        "s_entry": float(s_entry),
        "s_exit": float(s_exit),
        "length_in_water_m": float(length_in),
        "z_entry": float(z_entry),
        "gradient_upstream": gradient_up,
        "gradient_downstream": None if toe is None else -toe[1],
        "wall_foot_m": None if toe is None else toe[2],
        "estimates": estimates,
        "downstream_is_after": bool(downstream_after),
    }


# ---------------------------------------------------------------------------
# Bed surfaces
# ---------------------------------------------------------------------------


def hypsometric_bed(
    water_level: float, depth: float, n: float, rank: np.ndarray
) -> np.ndarray:
    """Bed elevations z = (WL - D) + D r^(1/n) for water cells."""
    return (water_level - depth) + depth * np.power(rank, 1.0 / n)


def valley_bed(
    water_level: float,
    thalweg_xy: np.ndarray,
    thalweg_z: np.ndarray,
    cell_xy: np.ndarray,
    shore_dist: np.ndarray,
    exponent: float = 1.0,
) -> np.ndarray:
    """V-shaped valley: each water cell lies between the nearest thalweg
    point (bed z_t, distance d_t) and the shore (distance d_s):
    z = z_t + (WL - z_t) (d_t / (d_t + d_s))^p."""
    out = np.empty(len(cell_xy))
    chunk = max(1, 4_000_000 // max(len(thalweg_xy), 1))
    for i in range(0, len(cell_xy), chunk):
        c = cell_xy[i : i + chunk]
        dx = c[:, 0, None] - thalweg_xy[None, :, 0]
        dy = c[:, 1, None] - thalweg_xy[None, :, 1]
        d2 = dx * dx + dy * dy
        j = np.argmin(d2, axis=1)
        d_t = np.sqrt(d2[np.arange(len(c)), j])
        z_t = thalweg_z[j]
        d_s = shore_dist[i : i + chunk]
        frac = np.where(d_t + d_s > 0, d_t / np.maximum(d_t + d_s, 1e-12), 1.0)
        out[i : i + chunk] = z_t + (water_level - z_t) * np.power(frac, exponent)
    return np.minimum(out, water_level)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def estimate_bathymetry(
    dem_window: np.ndarray,
    dem_nodata,
    window_gt,
    window: tuple,
    body: WaterBody,
    *,
    flat_tol: float = DEFAULT_FLAT_TOL_M,
    water_level: float | None = None,
    buffer_m: float | None = None,
    reach_m: float | None = None,
    valley_exponent: float = 1.0,
    supersample: int = DEFAULT_SUPERSAMPLE,
    hydrolakes: dict | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> BathymetryResult:
    sid = body.storage_id
    sx, sy = abs(float(window_gt[1])), abs(float(window_gt[5]))
    cell_area = sx * sy
    half_cell = 0.5 * min(sx, sy)
    z = np.asarray(dem_window, dtype=np.float64)
    rows, cols = z.shape
    warnings: list[str] = []

    cover = coverage_fraction(body.polygons, window_gt, z.shape, supersample)
    invalid = nodata_mask(z, dem_nodata)
    inside = cover > 0
    domain = inside & ~invalid
    if not domain.any():
        raise BathymetryError(
            f"Storage '{sid}': no valid DEM cells inside the polygon."
        )
    zd = np.where(domain, z, np.inf)
    floor_idx = np.unravel_index(int(np.argmin(zd)), z.shape)
    floor_z = float(z[floor_idx])
    edge = edge_cells(domain, ~inside)
    levels, spill_cell = priority_flood_levels(
        z, domain, floor_idx, edge=edge, is_canceled=is_canceled
    )
    spill_z = None if spill_cell is None else float(levels[spill_cell])

    # 1. water surface -------------------------------------------------------
    if water_level is None:
        water = domain & (levels <= floor_z + flat_tol)
        if water.sum() < MIN_WATER_CELLS:
            raise BathymetryError(
                f"Storage '{sid}': no flat water surface found at the bottom of "
                f"the depression (fewer than {MIN_WATER_CELLS} cells within "
                f"{flat_tol:g} m of the lowest ground). The depression looks dry - "
                "use DEM depression stage-storage directly - or the water surface "
                "is noisier than the flat-water tolerance (raise it, or give the "
                "water level)."
            )
        wl = float(np.median(z[water]))
    else:
        wl = float(water_level)
        if wl < floor_z - 1e-9:
            raise BathymetryError(
                f"Storage '{sid}': the water level ({wl:.3f} m) is below the "
                f"lowest ground in the polygon ({floor_z:.3f} m)."
            )
        water = domain & (levels <= wl + flat_tol)
        if water.sum() < MIN_WATER_CELLS:
            raise BathymetryError(
                f"Storage '{sid}': fewer than {MIN_WATER_CELLS} cells lie at or "
                f"below the water level {wl:.3f} m."
            )
    if (water & edge).any():
        warnings.append(
            f"Storage '{sid}': the water surface reaches the polygon edge - extend "
            "the polygon beyond the shoreline so the land around the water can be "
            "used."
        )
    if spill_z is not None and spill_z <= wl + flat_tol:
        warnings.append(
            f"Storage '{sid}': the water is at the spill level ({spill_z:.3f} m) - "
            "a full dam or a polygon cut at the shoreline; the hypsometric method "
            "needs dry ground above the water and is skipped."
        )
    w_water = cover * water
    water_area = float(w_water.sum() * cell_area)
    land = domain & ~water

    shore_dist = np.maximum(distance_to_false(water, sx, sy) - half_cell, half_cell)
    # land distances are measured to the cells at or below the water level
    # itself, so cells just above it (inside the flat-water tolerance) do
    # not shift the shoreline outwards
    core = water & (z <= wl)
    dist_to_water = distance_to_false(~core, sx, sy, off_grid=True)
    dist_to_water = np.where(water, 0.0, dist_to_water)
    max_shore = float(shore_dist[water].max())

    # 2. shoreline slope -----------------------------------------------------
    if buffer_m is None:
        buffer_m = max(3.0 * min(sx, sy), 0.1 * math.sqrt(water_area))
    slope = shoreline_slope(z, land, dist_to_water, wl, buffer_m, half_cell)
    methods: dict = {}
    if slope is not None and slope["slope"] > 0:
        d_a = slope["slope"] * max_shore
        methods["slope"] = {**slope, "max_depth": d_a, "status": "ok"}
    else:
        methods["slope"] = {"status": "no dry ground near the shore"}

    # 3. hypsometry above the water ------------------------------------------
    hyps = None
    if spill_z is not None and spill_z > wl + 2 * flat_tol:
        top = spill_z
        stage_storage = StageStorage(
            z[domain], cover[domain], levels[domain], cell_area
        )
        dh = max(0.02, (top - wl) / 40.0)
        heights = np.arange(flat_tol + dh, top - wl + 1e-9, dh)
        areas = [stage_storage.at(wl + h)[0] for h in heights]
        ref_area = stage_storage.at(wl + flat_tol)[0]
        d_cap = max(5.0 * methods["slope"].get("max_depth", 0.0), 50.0)
        hyps = fit_hypsometry(
            heights, areas, ref_area, 0.02, d_cap, ref_height=flat_tol
        )
    if hyps is not None:
        status = "ok" if hyps["bound"] is None else f"on {hyps['bound']} bound"
        methods["hypsometry"] = {
            "max_depth": hyps["depth"],
            "n": hyps["n"],
            "rmse": hyps["rmse"],
            "n_points": hyps["n_points"],
            "status": status,
        }
        if hyps["bound"] is not None:
            warnings.append(
                f"Storage '{sid}': the hypsometric fit is poorly constrained "
                f"(depth on the {hyps['bound']} search bound) and is not used."
            )
    else:
        methods["hypsometry"] = {"status": "not enough dry ground above the water"}
    hyps_ok = hyps is not None and hyps["bound"] is None
    shape_n = hyps["n"] if hyps_ok else DEFAULT_SHAPE_N

    # 4. valley (centreline) -------------------------------------------------
    valley = None
    profile_rows: list[dict] = []
    step = 0.5 * min(sx, sy)
    if reach_m is None:
        reach_m = max(20.0 * min(sx, sy), 200.0)
    if body.centrelines:
        best = None
        for line in body.centrelines:
            samples = sample_line(line, step)
            if not len(samples):
                continue
            col = np.floor((samples[:, 0] - window_gt[0]) / window_gt[1]).astype(int)
            row = np.floor((samples[:, 1] - window_gt[3]) / window_gt[5]).astype(int)
            on = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
            zs = np.full(len(samples), np.nan)
            zs[on] = z[row[on], col[on]]
            zs[on & invalid[np.clip(row, 0, rows - 1), np.clip(col, 0, cols - 1)]] = (
                np.nan
            )
            wet = np.zeros(len(samples), dtype=bool)
            wet[on] = water[row[on], col[on]]
            info = valley_profile(samples, zs, wet, wl, reach_m, step)
            if info is not None and (
                best is None or info["length_in_water_m"] > best[0]["length_in_water_m"]
            ):
                best = (info, samples, zs, wet)
        if best is None:
            warnings.append(
                f"Storage '{sid}': the stream centreline does not cross the water "
                "with river bed on both sides inside the DEM - the valley method "
                "is not used. Extend the line upstream and below the wall."
            )
        else:
            valley = best
            if len(body.centrelines) > 1:
                warnings.append(
                    f"Storage '{sid}': {len(body.centrelines)} centrelines cross the "
                    "storage; the longest under water is used."
                )
    if valley is not None:
        info = valley[0]
        est = info["estimates"]
        depths = {k: wl - v for k, v in est.items()}
        methods["valley"] = {
            "max_depth": float(np.mean(list(depths.values()))),
            "depth_upstream": _opt_float(depths.get("upstream")),
            "depth_downstream": _opt_float(depths.get("downstream")),
            "gradient_upstream": info["gradient_upstream"],
            "gradient_downstream": info["gradient_downstream"],
            "length_in_water_m": info["length_in_water_m"],
            "status": "ok",
        }
        if len(depths) == 2 and abs(
            depths["upstream"] - depths["downstream"]
        ) > 0.5 * max(depths.values()):
            warnings.append(
                f"Storage '{sid}': the bed at the wall from upstream "
                f"({depths['upstream']:.2f} m deep) and from below the wall "
                f"({depths['downstream']:.2f} m) differ by more than half - check "
                "the centreline and the reach length."
            )
    else:
        methods["valley"] = {
            "status": "no centreline" if not body.centrelines else "not usable"
        }

    # 5. range ---------------------------------------------------------------
    d_slope = methods["slope"].get("max_depth")
    d_hyps = methods["hypsometry"].get("max_depth") if hyps_ok else None
    if body.max_depth is not None and body.max_depth > 0:
        basis = "user"
        dc = float(body.max_depth)
        depth_s = {
            "low": USER_DEPTH_RANGE[0] * dc,
            "central": dc,
            "high": USER_DEPTH_RANGE[1] * dc,
        }
    elif valley is not None:
        basis = "valley"
        vals = [wl - v for v in valley[0]["estimates"].values()]
        dc = float(np.mean(vals))
        lo, hi = min(vals), max(vals)
        lo = min(lo, (1 - MIN_VALLEY_SPREAD) * dc)
        hi = max(hi, (1 + MIN_VALLEY_SPREAD) * dc)
        depth_s = {"low": lo, "central": dc, "high": hi}
    elif d_slope is not None and d_hyps is not None:
        basis = "slope+hypsometry"
        dc = min(d_slope, d_hyps)
        depth_s = {"low": LOW_FACTOR * dc, "central": dc, "high": max(d_slope, d_hyps)}
    elif d_slope is not None:
        basis = "slope"
        depth_s = {"low": 0.25 * d_slope, "central": 0.5 * d_slope, "high": d_slope}
        warnings.append(
            f"Storage '{sid}': only the shoreline-slope method was available; the "
            "central estimate is taken as half of it (pans are usually flatter "
            "than their banks)."
        )
    elif d_hyps is not None:
        basis = "hypsometry"
        depth_s = {"low": LOW_FACTOR * d_hyps, "central": d_hyps, "high": 1.5 * d_hyps}
    else:
        raise BathymetryError(
            f"Storage '{sid}': no estimate possible - there is no dry ground "
            "around the water inside the polygon. Extend the polygon beyond the "
            "shoreline, or give a maximum depth."
        )

    # 6. bed surfaces ----------------------------------------------------------
    w_rows, w_cols = np.nonzero(water)
    weights = cover[water]
    sd = shore_dist[water]
    rank = area_rank(sd, weights)
    # stretch so the cell farthest from shore sits exactly at the bed level
    r_min = float(rank.min()) if rank.size else 0.0
    if r_min < 1.0:
        rank = (rank - r_min) / (1.0 - r_min)
    beds: dict = {}
    scenarios: dict = {}
    if valley is not None:
        info, samples, zs, wet = valley
        s = samples[:, 2]
        s_lo, s_hi = sorted((info["s_entry"], info["s_exit"]))
        under = wet & (s >= s_lo) & (s <= s_hi)
        t_xy = samples[under, :2]
        frac = np.abs(s[under] - info["s_entry"]) / max(info["length_in_water_m"], 1e-9)
        cell_xy = np.column_stack(
            [
                window_gt[0] + (w_cols + 0.5) * window_gt[1],
                window_gt[3] + (w_rows + 0.5) * window_gt[5],
            ]
        )
    for name in SCENARIOS:
        depth = max(float(depth_s[name]), 0.0)
        bed = np.full(z.shape, np.nan)
        if valley is not None:
            z_wall = wl - depth
            z_entry = min(info["z_entry"], wl)
            t_z = np.minimum(z_entry + (z_wall - z_entry) * frac, wl)
            values = valley_bed(wl, t_xy, t_z, cell_xy, sd, valley_exponent)
        else:
            values = hypsometric_bed(wl, depth, shape_n, rank)
        bed[w_rows, w_cols] = values
        dep = wl - values
        vol = float(np.sum(dep * weights) * cell_area)
        scenarios[name] = {
            "max_depth": float(dep.max()) if dep.size else 0.0,
            "mean_depth": vol / water_area if water_area > 0 else 0.0,
            "volume_m3": vol,
            "bed_min": float(values.min()) if values.size else wl,
        }
        beds[name] = bed

    if valley is not None:
        info, samples, zs, wet = valley
        for (x, y, ch), zz, ww in zip(samples, zs, wet, strict=True):
            rel = ch - info["s_entry"]
            if not info["downstream_is_after"]:
                rel = -rel
            row = {"chainage_m": round(float(ch), 3), "x": float(x), "y": float(y)}
            row["from_entry_m"] = round(float(rel), 3)
            row["dem_z"] = None if not np.isfinite(zz) else round(float(zz), 3)
            row["in_water"] = int(bool(ww))
            for name in SCENARIOS:
                if ww:
                    depth = max(float(depth_s[name]), 0.0)
                    f = min(max(rel / max(info["length_in_water_m"], 1e-9), 0.0), 1.0)
                    z_entry = min(info["z_entry"], wl)
                    zb = min(z_entry + ((wl - depth) - z_entry) * f, wl)
                    row[f"bed_{name}"] = round(float(zb), 3)
                else:
                    row[f"bed_{name}"] = row["dem_z"]
            profile_rows.append(row)

    # HydroLAKES check ---------------------------------------------------------
    if hydrolakes:
        hl_depth = hydrolakes.get("depth_avg_m")
        if hl_depth is not None:
            lo = scenarios["low"]["mean_depth"]
            hi = scenarios["high"]["mean_depth"]
            if not (lo <= hl_depth <= hi):
                warnings.append(
                    f"Storage '{sid}': HydroLAKES mean depth {hl_depth:.2f} m lies "
                    f"outside the estimated range of mean depth ({lo:.2f} - "
                    f"{hi:.2f} m). HydroLAKES is itself modelled; treat as a flag "
                    "to review, not a correction."
                )

    attributes = {
        "storage_id": sid,
        "src_fid": body.src_fid,
        "water_lvl": round(wl, 3),
        "floor_dem": round(floor_z, 3),
        "z_spill": None if spill_z is None else round(spill_z, 3),
        "water_ha": round(water_area / 1e4, 4),
        "basis": basis,
        "shape_n": round(shape_n, 3),
        "slope_pc": (
            None
            if methods["slope"].get("slope") is None
            else round(100 * methods["slope"]["slope"], 3)
        ),
        "d_slope": None if d_slope is None else round(d_slope, 3),
        "d_hyps": None if d_hyps is None else round(d_hyps, 3),
        "d_valley": (
            None if valley is None else round(methods["valley"]["max_depth"], 3)
        ),
        "d_user": body.max_depth,
        "hl_depth": None if not hydrolakes else hydrolakes.get("depth_avg_m"),
        "hl_vol_m3": None if not hydrolakes else hydrolakes.get("volume_m3"),
    }
    for name, key in (("low", "lo"), ("central", "c"), ("high", "hi")):
        sc = scenarios[name]
        attributes[f"dmax_{key}"] = round(sc["max_depth"], 3)
        attributes[f"dmean_{key}"] = round(sc["mean_depth"], 3)
        attributes[f"vol_{key}"] = round(sc["volume_m3"], 1)

    return BathymetryResult(
        storage_id=sid,
        src_fid=body.src_fid,
        water_level=wl,
        water_area_m2=water_area,
        water_mask=water,
        window=tuple(int(v) for v in window),
        window_gt=tuple(window_gt),
        beds=beds,
        scenarios=scenarios,
        methods=methods,
        basis=basis,
        attributes=attributes,
        profile=profile_rows,
        warnings=warnings,
    )


def depth_grid(result: BathymetryResult, scenario: str = "central") -> np.ndarray:
    """Water depth below the surveyed water level (NaN where dry)."""
    return (result.water_level - result.beds[scenario]).astype(np.float32)


def mosaic_depths(
    results: Sequence[BathymetryResult],
    scenario: str = "central",
    nodata: float = DEFAULT_NODATA,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """One depth grid covering every result window (deepest value where
    windows overlap); (array with ``nodata`` where dry, (row0, col0))."""
    results = [r for r in results if r.water_mask.size]
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
        np.fmax(view, depth_grid(r, scenario), out=view)
    return np.where(np.isnan(out), nodata, out).astype(np.float32), (r0, c0)


# ---------------------------------------------------------------------------
# File I/O (GDAL - bundled with QGIS)
# ---------------------------------------------------------------------------


def write_bed_dem(
    src_path: str,
    out_path: str,
    results: Sequence[BathymetryResult],
    scenario: str = "central",
) -> None:
    """Copies the input DEM to a float32 GeoTIFF and writes the estimated bed
    into the water cells of every result (the rest of the DEM unchanged), so
    the result can go straight into DEM depression stage-storage or a
    hydraulic model."""
    from osgeo import gdal

    gdal.UseExceptions()
    src = gdal.Open(src_path)
    if src is None:
        raise GridError(f"Could not open raster: {src_path}")
    ds = gdal.Translate(
        out_path,
        src,
        format="GTiff",
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    src = None
    band = ds.GetRasterBand(1)
    for r in results:
        r0, c0, nr, nc = r.window
        current = band.ReadAsArray(c0, r0, nc, nr).astype(np.float64)
        bed = r.beds[scenario]
        updated = np.where(np.isfinite(bed), bed, current)
        band.WriteArray(updated.astype(np.float32), c0, r0)
    band.FlushCache()
    ds = None
