"""
Core catchment-delineation logic - pure NumPy, no QGIS dependency (GDAL
is imported only inside the file-I/O helpers at the bottom).

INPUTS
- A D8 pointer raster, in this suite's own D8 Flow Direction encoding
  (WhiteboxTools default) or the ESRI encoding:

      Default: NE=1, E=2, SE=4, S=8, SW=16, W=32, NW=64, N=128
      ESRI:    E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128

  0 = pit / no downslope neighbour. Any other value that is not NoData
  is counted as INVALID, treated as NoData and reported.
- A DEM on exactly the same grid (same shape and geotransform) - used
  for the catchment DEM output and every elevation-based parameter.
- One or more outlets: pour points (snapped to the highest flow
  accumulation within a search radius) or pour lines (every cell the
  line passes through is an outlet cell; the line is rasterised as a
  4-connected chain so no D8 flow path can slip diagonally through it).

METHOD
- Flow network: each valid cell gets a flat "downstream index"; pits,
  cells draining off the grid or into NoData are terminals.
- Flow accumulation (for snapping and drainage density) is computed in
  topological order: every cell's number of D8 steps to its terminal
  ("depth") is found by pointer doubling, cells are grouped by depth and
  pushed downstream deepest-first. Upstream cells always have a larger
  depth than the cell they drain to, so each group is final before it
  is pushed. Cost is O(N log L) rather than O(N x L) for a grid of N
  cells and a longest path of L cells, which matters at basin scale.
  Results are identical to this suite's D8 Flow Accumulation tool
  (cross-checked in the tests).
- Catchment labelling: outlet cells are made terminals, and pointer
  doubling finds, for every cell, the first outlet cell it reaches
  downstream. That gives INCREMENTAL catchments (each cell belongs to
  the first outlet it meets). Outlet-to-outlet drainage links are
  recorded, and a TOTAL catchment is the incremental catchment of the
  outlet plus those of every outlet upstream of it.
- Polygon: the catchment's cell boundary is traced into rings with the
  interior kept on the left (outer rings anticlockwise). At a vertex
  where two cells touch only diagonally, the tracer turns left, so
  diagonally-touching parts become separate rings touching at a single
  point (a valid MultiPolygon). Interior holes (cells enclosed by the
  catchment that do not drain to the outlet - usually unconditioned pits
  or NoData - including holes closed off only by corner contacts) are
  filled, and their area is reported. In incremental mode a nested
  sub-catchment enclosed by another is left as a real hole, so
  incremental polygons never overlap. Geometry is
  repaired again with makeValid in the QGIS wrapper as a safeguard.

PARAMETERS (all lengths in metres - the DEM must be in a projected CRS)
- Area/perimeter, elevation statistics, centroid, longest flow path
  (LFP, measured cell centre to outlet cell centre along D8 steps),
  centroid-to-outlet length along the LFP (Lca), slopes (average
  H/L, 10-85, equal-area, grid mean Horn slope), shape indices (form
  factor, circularity, elongation, Gravelius), hypsometric integral,
  stream length and drainage density (streams = cells whose upstream
  area is at least a threshold), and time of concentration by Kirpich,
  USBR / SANRAL and Bransby-Williams.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.d8_network import (  # noqa: F401 - re-exported
    D_COL,
    D_ROW,
    MAX_DOUBLING_ITERATIONS,
    POINTER_VALS_DEFAULT,
    POINTER_VALS_ESRI,
    GridError,
    bifurcation_ratio,
    build_downstream_index,
    cell_center,
    check_geotransform,
    check_same_grid,
    flow_accumulation_cells,
    nodata_mask,
    pointer_to_direction_index,
    read_raster,
    resolve_terminals,
    segment_starts,
    snap_to_max_accumulation,
    step_lengths,
    strahler_order,
    stream_links,
    stream_order_summary,
    world_to_cell,
    write_raster,
)
from mayim_tools.hydrology._common.grid_polygons import (  # noqa: F401 - re-exported
    _point_in_ring,
    _ring_signed_area_idx,
    _winding_fill,
    fill_holes,
    group_polygons,
    ring_length,
    ring_to_map,
    rings_to_multipolygon_wkt,
    simplify_ring,
    split_ring,
    trace_rings,
)

VALID_MODES = ("total", "incremental")
DEFAULT_DEM_NODATA = -9999.0
HYPSOMETRIC_STEPS = 21  # relative heights 0.00, 0.05, ... 1.00

# Raised for invalid inputs (mismatched grids, bad mode, etc.). The same
# class as the shared module's GridError, so either name catches both.
CatchmentError = GridError


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Outlet:
    """One pour point or pour line feature, already in the raster's CRS.

    ``parts`` holds one coordinate list per geometry part: a single
    (x, y) for each point part, or the vertex list of each line part."""

    outlet_id: str
    kind: str  # "point" or "line"
    parts: list[list[tuple[float, float]]]
    src_fid: int = -1


@dataclass
class CatchmentResult:
    """Everything produced for one outlet."""

    outlet_id: str
    kind: str
    src_fid: int
    attributes: dict
    polygons: list  # [[outer, hole, ...], ...] rings in map coordinates
    flow_path: list[tuple[float, float]]  # LFP from source to outlet
    outlet_xy: tuple[float, float] | None
    hypsometric_curve: list[dict]
    detail: dict = field(default_factory=dict)  # report/figure data


@dataclass
class DelineationResult:
    catchments: list[CatchmentResult] = field(default_factory=list)
    dem_array: np.ndarray | None = None
    dem_geotransform: tuple | None = None
    dem_nodata: float = DEFAULT_DEM_NODATA
    warnings: list[str] = field(default_factory=list)
    # (outlet id, flat cell indices) of every placed outlet - lets the
    # Stream Network core extract the same network upstream of them
    outlet_cells: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pointer decoding and flow network
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Outlet placement
# ---------------------------------------------------------------------------


def line_cells(
    coords: Sequence[tuple[float, float]], geotransform, shape
) -> list[tuple[int, int]]:
    """Cells crossed by a polyline, as a 4-connected chain in order.

    The line is sampled at a quarter of the cell size; wherever two
    consecutive cells touch only diagonally, the in-between cell closer
    to the line is added, so no D8 path can cross the line without
    entering one of its cells."""
    rows, cols = shape
    sx, sy = abs(geotransform[1]), abs(geotransform[5])
    step = 0.25 * min(sx, sy)
    samples: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:], strict=True):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0.0, 1.0, n + 1)
        samples.extend(zip(x0 + t * (x1 - x0), y0 + t * (y1 - y0), strict=True))
    if len(coords) == 1:
        samples.append(tuple(coords[0]))

    chain: list[tuple[int, int]] = []
    for k, (x, y) in enumerate(samples):
        cell = world_to_cell(x, y, geotransform)
        if chain and chain[-1] == cell:
            continue
        if chain:
            pr, pc = chain[-1]
            if abs(cell[0] - pr) == 1 and abs(cell[1] - pc) == 1:
                # diagonal step: add the in-between cell nearer the line
                xm = 0.5 * (x + samples[k - 1][0])
                ym = 0.5 * (y + samples[k - 1][1])
                mid_a, mid_b = (pr, cell[1]), (cell[0], pc)
                xa, ya = cell_center(*mid_a, geotransform)
                xb, yb = cell_center(*mid_b, geotransform)
                da = (xa - xm) ** 2 + (ya - ym) ** 2
                db = (xb - xm) ** 2 + (yb - ym) ** 2
                chain.append(mid_a if da <= db else mid_b)
        chain.append(cell)

    seen: set[tuple[int, int]] = set()
    out = []
    for r, c in chain:
        if 0 <= r < rows and 0 <= c < cols and (r, c) not in seen:
            seen.add((r, c))
            out.append((r, c))
    return out


# ---------------------------------------------------------------------------
# Boundary tracing, hole filling and polygon output - shared implementation
# lives in hydrology/_common/grid_polygons.py (imported at the top of this
# module and re-exported, so existing imports of these names keep working).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Terrain / hydrological parameters
# ---------------------------------------------------------------------------


def horn_gradient(
    z_padded: np.ndarray, sx: float, sy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981) gradient for the interior of a 1-cell-padded elevation
    window: (dz/dx towards east, dz/dy towards south), both m/m. NaN
    neighbours are replaced by the centre value."""
    centre = z_padded[1:-1, 1:-1]
    rows, cols = centre.shape

    def nb(dr, dc):
        a = z_padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        return np.where(np.isnan(a), centre, a)

    a, b, c = nb(-1, -1), nb(-1, 0), nb(-1, 1)
    d, f = nb(0, -1), nb(0, 1)
    g, h, i = nb(1, -1), nb(1, 0), nb(1, 1)
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * sx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * sy)
    return dzdx, dzdy


def horn_slope(z_padded: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Horn (1981) slope (m/m) for the interior of a 1-cell-padded
    elevation window."""
    dzdx, dzdy = horn_gradient(z_padded, sx, sy)
    return np.hypot(dzdx, dzdy)


def horn_aspect_deg(dzdx: np.ndarray, dzdy: np.ndarray) -> np.ndarray:
    """Direction the slope faces (downslope), degrees clockwise from north."""
    return np.degrees(np.arctan2(-dzdx, dzdy)) % 360.0


# Slope classes (%) after FAO (2006) Guidelines for soil description.
SLOPE_CLASSES = [
    (0.0, 2.0, "0-2 % (level to very gently sloping)"),
    (2.0, 5.0, "2-5 % (gently sloping)"),
    (5.0, 10.0, "5-10 % (sloping)"),
    (10.0, 15.0, "10-15 % (strongly sloping)"),
    (15.0, 30.0, "15-30 % (moderately steep)"),
    (30.0, 60.0, "30-60 % (steep)"),
    (60.0, math.inf, "> 60 % (very steep)"),
]
ASPECT_SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
FLAT_SLOPE = 0.005  # m/m - cells flatter than this have no meaningful aspect


def slope_class_distribution(slope_pc: np.ndarray) -> list[dict]:
    s = slope_pc[~np.isnan(slope_pc)]
    n = max(s.size, 1)
    return [
        {"label": label, "pct": 100.0 * float(np.sum((s >= lo) & (s < hi))) / n}
        for lo, hi, label in SLOPE_CLASSES
    ]


def aspect_distribution(aspect: np.ndarray, slope: np.ndarray) -> list[dict]:
    ok = ~np.isnan(aspect) & ~np.isnan(slope)
    a, s = aspect[ok], slope[ok]
    n = max(a.size, 1)
    flat = s < FLAT_SLOPE
    sector = (((a + 22.5) % 360.0) // 45.0).astype(int)
    out = [
        {"label": name, "pct": 100.0 * float(np.sum(~flat & (sector == i))) / n}
        for i, name in enumerate(ASPECT_SECTORS)
    ]
    out.append({"label": "Flat", "pct": 100.0 * float(np.sum(flat)) / n})
    return out


def average_slope(x: np.ndarray, z: np.ndarray) -> float | None:
    """(z_top - z_outlet) / L along a profile; x = distance from outlet."""
    if x.size < 2 or x[-1] <= 0 or np.isnan(z).any():
        return None
    return float((z[-1] - z[0]) / x[-1])


def slope_10_85(x: np.ndarray, z: np.ndarray) -> float | None:
    """Slope between 10 % and 85 % of the length from the outlet."""
    if x.size < 2 or x[-1] <= 0 or np.isnan(z).any():
        return None
    length = x[-1]
    z10 = np.interp(0.10 * length, x, z)
    z85 = np.interp(0.85 * length, x, z)
    return float((z85 - z10) / (0.75 * length))


def equal_area_slope(x: np.ndarray, z: np.ndarray) -> float | None:
    """Slope of the line through the outlet that has the same area under
    it as the actual profile: Se = 2 * A / L^2."""
    if x.size < 2 or x[-1] <= 0 or np.isnan(z).any():
        return None
    zr = z - z[0]
    area = float(np.sum(0.5 * (zr[1:] + zr[:-1]) * np.diff(x)))
    return 2.0 * area / float(x[-1]) ** 2


def tc_kirpich_minutes(length_m: float, slope: float | None) -> float | None:
    """Kirpich (1940), metric: tc = 0.0195 L^0.77 S^-0.385 (min, L m, S m/m)."""
    if slope is None or slope <= 0 or length_m <= 0:
        return None
    return 0.0195 * length_m**0.77 * slope**-0.385


def tc_usbr_minutes(length_km: float, slope: float | None) -> float | None:
    """USBR (1973) as used in the SANRAL Drainage Manual for defined
    watercourses: tc = (0.87 L^2 / (1000 S))^0.385 hours (L km, S m/m)."""
    if slope is None or slope <= 0 or length_km <= 0:
        return None
    return 60.0 * (0.87 * length_km**2 / (1000.0 * slope)) ** 0.385


def tc_bransby_williams_minutes(
    length_km: float, area_km2: float, slope: float | None
) -> float | None:
    """Bransby-Williams (ARR form): tc = 58 L / (A^0.1 Se^0.2) minutes
    (L km, A km2, Se in m/km - the slope argument here is m/m)."""
    if slope is None or slope <= 0 or length_km <= 0 or area_km2 <= 0:
        return None
    return 58.0 * length_km / (area_km2**0.1 * (slope * 1000.0) ** 0.2)


def hypsometric_curve(z: np.ndarray, cell_area_m2: float) -> list[dict]:
    """Relative area above each relative height (0.00 ... 1.00)."""
    z = z[~np.isnan(z)]
    if z.size == 0:
        return []
    zmin, zmax = float(z.min()), float(z.max())
    relief = zmax - zmin
    out = []
    for h in np.linspace(0.0, 1.0, HYPSOMETRIC_STEPS):
        level = zmin + h * relief
        above = int(np.sum(z >= level - 1e-9)) if relief > 0 else z.size
        out.append(
            {
                "rel_height": round(float(h), 4),
                "rel_area": above / z.size,
                "elevation_m": level,
                "area_above_km2": above * cell_area_m2 / 1e6,
            }
        )
    return out


def _none_round(value, ndigits):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), ndigits)


def _report_detail(
    geotransform,
    r0,
    c0,
    w,
    x_prof,
    z_prof,
    z_values,
    slope_pc,
    aspect,
    slope,
    order,
    local_next,
    orders,
    centroid,
    lca_point,
    stream_threshold_km2,
) -> dict:
    """Data behind the report tables and figures for one catchment."""
    idx = np.arange(order.size)
    src = np.flatnonzero((order > 0) & (local_next != idx))
    dst = local_next[src]
    x0, y0 = cell_center(src // w + r0, src % w + c0, geotransform)
    x1, y1 = cell_center(dst // w + r0, dst % w + c0, geotransform)
    counts, edges = (
        np.histogram(z_values, bins=20) if z_values.size else (np.zeros(0), [])
    )
    return {
        "profile_x_m": np.asarray(x_prof, dtype=float),
        "profile_z_m": np.asarray(z_prof, dtype=float),
        "streams": np.column_stack([x0, y0, x1, y1, order[src]]),
        "elev_hist_counts": np.asarray(counts),
        "elev_hist_edges": np.asarray(edges),
        "slope_classes": slope_class_distribution(slope_pc),
        "aspect": aspect_distribution(aspect, slope),
        "stream_orders": orders,
        "centroid": centroid,
        "lca_point": lca_point,
        "stream_threshold_km2": stream_threshold_km2,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def delineate_catchments(
    pointer: np.ndarray,
    pointer_nodata,
    dem: np.ndarray,
    dem_nodata,
    geotransform,
    outlets: Sequence[Outlet],
    esri_style: bool = False,
    snap_radius_cells: int = 3,
    mode: str = "total",
    stream_threshold_km2: float = 0.1,
    progress: Callable[[float, str], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> DelineationResult:
    """Delineates one catchment per outlet. See module docstring."""
    if mode not in VALID_MODES:
        raise CatchmentError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    sx, sy = check_geotransform(geotransform)
    pointer = np.asarray(pointer)
    dem = np.asarray(dem, dtype=np.float64)
    if pointer.shape != dem.shape:
        raise CatchmentError("D8 pointer and DEM are not on the same grid.")
    rows, cols = pointer.shape
    cell_area = sx * sy
    result = DelineationResult()
    warn = result.warnings.append

    def report(fraction, message):
        if progress is not None:
            progress(fraction, message)

    def canceled():
        return is_canceled is not None and is_canceled()

    # --- flow network -----------------------------------------------------
    report(0.02, "Decoding D8 pointer")
    direction_index, n_invalid = pointer_to_direction_index(
        pointer, pointer_nodata, esri_style
    )
    if n_invalid:
        warn(
            f"{n_invalid} pointer cell(s) hold values that are not valid "
            f"{'ESRI' if esri_style else 'WhiteboxTools'} D8 codes and were "
            "treated as NoData. Check the pointer encoding setting."
        )
    valid_flat = (direction_index != -2).ravel()
    dem_flat = np.where(nodata_mask(dem, dem_nodata), np.nan, dem).ravel()

    nxt = build_downstream_index(direction_index)
    steps = step_lengths(direction_index, sx, sy)
    idx = np.arange(nxt.size, dtype=np.int64)
    steps[nxt == idx] = 0.0

    report(0.10, "Computing flow accumulation")
    acc, n_loop = flow_accumulation_cells(nxt, valid_flat)
    if n_loop:
        warn(
            f"{n_loop} cell(s) are caught in or drain into a flow loop (cells "
            "pointing at each other). They are excluded. The pointer raster "
            "is probably corrupt or not a D8 pointer."
        )
    acc2d = acc.reshape(rows, cols)
    if canceled():
        return result

    # --- outlet cells -----------------------------------------------------
    report(0.30, "Placing outlets")
    seeds_all: list[np.ndarray] = []
    meta: list[dict] = []
    for outlet in outlets:
        info = {"x_in": None, "y_in": None, "x_out": None, "y_out": None}
        info["snap_m"] = None
        info["acc_km2"] = None
        cells: list[tuple[int, int]] = []
        if outlet.kind == "point":
            for part in outlet.parts:
                x, y = part[0]
                r, c = world_to_cell(x, y, geotransform)
                snapped = snap_to_max_accumulation(r, c, acc2d, snap_radius_cells)
                if snapped is None:
                    continue
                if info["x_in"] is None:
                    xo, yo = cell_center(*snapped, geotransform)
                    info.update(
                        x_in=x,
                        y_in=y,
                        x_out=float(xo),
                        y_out=float(yo),
                        snap_m=math.hypot(float(xo) - x, float(yo) - y),
                        acc_km2=float(acc2d[snapped]) * cell_area / 1e6,
                    )
                cells.append(snapped)
        elif outlet.kind == "line":
            for part in outlet.parts:
                if len(part) >= 1:
                    cells.extend(line_cells(part, geotransform, (rows, cols)))
            cells = [rc for rc in cells if valid_flat[rc[0] * cols + rc[1]]]
        else:
            raise CatchmentError(f"Unknown outlet kind {outlet.kind!r}")

        flat = (
            np.unique(np.array([r * cols + c for r, c in cells], dtype=np.int64))
            if cells
            else np.zeros(0, dtype=np.int64)
        )
        if flat.size == 0:
            warn(
                f"Outlet '{outlet.outlet_id}' does not fall on any valid D8 cell "
                "(outside the raster or on NoData) and was skipped."
            )
        seeds_all.append(flat)
        meta.append(info)

    n_out = len(outlets)
    result.outlet_cells = [
        (o.outlet_id, flat) for o, flat in zip(outlets, seeds_all, strict=True)
    ]
    seed_owner = np.zeros(nxt.size, dtype=np.int32)  # 0 = none, k+1 = outlet k
    parents: list[set[int]] = [set() for _ in range(n_out)]
    for k, flat in enumerate(seeds_all):
        if flat.size == 0:
            continue
        owners = seed_owner[flat]
        shared = owners > 0
        for j in np.unique(owners[shared]):
            j = int(j) - 1
            parents[j].add(k)  # the shared cell's owner j also drains to k
            warn(
                f"Outlet '{outlets[k].outlet_id}' shares outlet cell(s) with "
                f"'{outlets[j].outlet_id}'; those cells are assigned to the "
                "first outlet."
            )
        seed_owner[flat[~shared]] = k + 1

    # --- labelling ---------------------------------------------------------
    report(0.40, "Labelling catchments")
    nxt_seeded = nxt.copy()
    owned = np.flatnonzero(seed_owner)
    nxt_seeded[owned] = owned
    terminal, _, unresolved = resolve_terminals(nxt_seeded)
    label = seed_owner[terminal]
    label[unresolved | ~valid_flat] = 0

    for k, flat in enumerate(seeds_all):
        for s in flat:
            d = nxt[s]
            if d == s:
                continue
            owner = int(label[d])
            if owner and owner - 1 != k:
                parents[k].add(owner - 1)

    children: list[set[int]] = [set() for _ in range(n_out)]
    for k in range(n_out):
        for p in parents[k]:
            children[p].add(k)

    def upstream_of(k: int) -> set[int]:
        found, stack = set(), list(children[k])
        while stack:
            j = stack.pop()
            if j not in found and j != k:
                found.add(j)
                stack.extend(children[j])
        return found

    labelled = np.flatnonzero(label)
    order = labelled[np.argsort(label[labelled], kind="stable")]
    bounds = np.searchsorted(label[order], np.arange(1, n_out + 2))
    cells_of = [order[bounds[k] : bounds[k + 1]] for k in range(n_out)]

    thr_cells = stream_threshold_km2 * 1e6 / cell_area
    windows: list[tuple[int, int, np.ndarray]] = []

    # --- per-catchment processing -----------------------------------------
    for k, outlet in enumerate(outlets):
        if canceled():
            return result
        report(0.45 + 0.5 * k / max(n_out, 1), f"Catchment {outlet.outlet_id}")
        if seeds_all[k].size == 0:
            continue
        members = {k} | (upstream_of(k) if mode == "total" else set())
        flat_cells = np.concatenate([cells_of[j] for j in sorted(members)])
        if flat_cells.size == 0:
            warn(
                f"Outlet '{outlet.outlet_id}' has no cells of its own in "
                "incremental mode (all its outlet cells belong to another "
                "outlet) and was skipped."
            )
            continue

        rr_all, cc_all = flat_cells // cols, flat_cells % cols
        r0, r1 = int(rr_all.min()), int(rr_all.max()) + 1
        c0, c1 = int(cc_all.min()), int(cc_all.max()) + 1
        h, w = r1 - r0, c1 - c0
        mask = np.zeros((h, w), dtype=bool)
        mask[rr_all - r0, cc_all - c0] = True

        gflat = (np.arange(h)[:, None] + r0) * cols + (np.arange(w)[None, :] + c0)
        gflat = gflat.ravel()

        keep_out = None
        if mode == "incremental":
            # cells of other outlets' sub-catchments stay out (real holes)
            keep_out = (label[gflat] != 0).reshape(h, w) & ~mask
        filled, rings_idx = fill_holes(mask, keep_out)
        n_cells = int(mask.sum())
        n_filled = int(filled.sum())

        # local downstream network restricted to the catchment
        loc = np.arange(h * w, dtype=np.int64)
        local_next = loc.copy()
        in_mask = mask.ravel()
        g_down = nxt[gflat]
        dr_, dc_ = g_down // cols - r0, g_down % cols - c0
        inside = in_mask & (dr_ >= 0) & (dr_ < h) & (dc_ >= 0) & (dc_ < w)
        down_local = np.where(inside, dr_ * w + dc_, loc)
        inside &= in_mask[np.clip(down_local, 0, h * w - 1)]
        is_seed = np.isin(gflat, seeds_all[k])
        moves = inside & ~is_seed & (g_down != gflat)
        local_next[moves] = down_local[moves]
        weights = np.where(moves, steps[gflat], 0.0)
        _, dist, _ = resolve_terminals(local_next, weights)

        z_local = dem_flat[gflat]
        dist_masked = np.where(in_mask, dist, -1.0)
        start = int(np.argmax(dist_masked))
        path = [start]
        while local_next[path[-1]] != path[-1]:
            path.append(int(local_next[path[-1]]))
        path_arr = np.array(path[::-1])  # outlet first
        x_prof = dist[path_arr]
        z_prof = z_local[path_arr]
        outlet_local = path[-1]
        lfp_m = float(dist[start])

        pr, pc = path_arr // w + r0, path_arr % w + c0
        px, py = cell_center(pr, pc, geotransform)
        flow_path = list(zip(px[::-1].tolist(), py[::-1].tolist(), strict=True))

        # polygon
        polygons = [
            [ring_to_map(simplify_ring(r), geotransform, r0, c0) for r in poly]
            for poly in group_polygons(rings_idx)
        ]
        perimeter_m = sum(ring_length(r) for poly in polygons for r in poly)
        area_km2 = n_cells * cell_area / 1e6
        poly_km2 = n_filled * cell_area / 1e6
        hole_ha = (n_filled - n_cells) * cell_area / 1e4

        # centroid and Lca
        fr, fc = np.nonzero(filled)
        cx, cy = cell_center(fr.mean() + r0, fc.mean() + c0, geotransform)
        cx, cy = float(cx), float(cy)
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        lca_m = float(x_prof[int(np.argmin(d2))])

        # elevations
        zm = z_local[in_mask]
        zv = zm[~np.isnan(zm)]
        if zv.size < zm.size:
            warn(
                f"Catchment '{outlet.outlet_id}': {zm.size - zv.size} cell(s) "
                "have no DEM value; elevation statistics ignore them."
            )
        z_min = float(zv.min()) if zv.size else None
        z_max = float(zv.max()) if zv.size else None
        z_mean = float(zv.mean()) if zv.size else None
        z_outlet = z_local[outlet_local]
        z_outlet = None if np.isnan(z_outlet) else float(z_outlet)

        # grid slope (Horn) on a padded window
        pr0, pr1 = max(r0 - 1, 0), min(r1 + 1, rows)
        pc0, pc1 = max(c0 - 1, 0), min(c1 + 1, cols)
        zwin = dem_flat.reshape(rows, cols)[pr0:pr1, pc0:pc1]
        zpad = np.full((h + 2, w + 2), np.nan)
        zr0, zc0 = 1 + pr0 - r0, 1 + pc0 - c0
        zpad[zr0 : zr0 + zwin.shape[0], zc0 : zc0 + zwin.shape[1]] = zwin
        dzdx, dzdy = horn_gradient(zpad, sx, sy)
        slope_grid = np.hypot(dzdx, dzdy)
        aspect_grid = horn_aspect_deg(dzdx, dzdy)
        s_cells = slope_grid[mask & ~np.isnan(slope_grid)]
        s_mean_pc = float(s_cells.mean() * 100.0) if s_cells.size else None

        # profile slopes
        s_avg = average_slope(x_prof, z_prof)
        s_1085 = slope_10_85(x_prof, z_prof)
        s_ea = equal_area_slope(x_prof, z_prof)

        # streams and drainage density
        acc_local = acc[gflat]
        stream = in_mask & (acc_local >= thr_cells)
        strm_m = float(np.sum(np.where(stream & moves, steps[gflat], 0.0)))
        step_local = np.where(moves, steps[gflat], 0.0)
        order = strahler_order(local_next, stream)
        max_in = segment_starts(local_next, order)
        orders = stream_order_summary(order, max_in, local_next, step_local)
        n_streams = sum(o["n_segments"] for o in orders)
        bif = bifurcation_ratio(orders)
        link_features = []
        for link in stream_links(local_next, stream, order, step_local):
            cells = np.array(link["cells"])
            lx, ly = cell_center(cells // w + r0, cells % w + c0, geotransform)
            z0 = z_local[cells[0]]
            z1 = (
                z_local[link["last_own"]]
                if link["ds_link"] is None
                else (z_local[cells[-1]])
            )
            link_features.append(
                {
                    "coords": list(zip(lx.tolist(), ly.tolist(), strict=True)),
                    "link_id": link["link_id"],
                    "ds_link": link["ds_link"],
                    "strahler": link["order"],
                    "length_m": round(link["length_m"], 2),
                    "us_km2": round(
                        float(acc_local[link["last_own"]]) * cell_area / 1e6, 6
                    ),
                    "z_start": _none_round(None if np.isnan(z0) else z0, 3),
                    "z_end": _none_round(None if np.isnan(z1) else z1, 3),
                    "slope": _none_round(
                        (
                            (z0 - z1) / link["length_m"]
                            if link["length_m"] > 0 and not np.isnan(z0 - z1)
                            else None
                        ),
                        6,
                    ),
                }
            )

        # extra geomorphometry
        h_rel = (z_max - z_outlet) if (zv.size and z_outlet is not None) else None
        ox, oy = float(px[0]), float(py[0])
        fx, fy = cell_center(fr + r0, fc + c0, geotransform)
        lb_m = float(np.sqrt(np.max((fx - ox) ** 2 + (fy - oy) ** 2)))
        straight_m = math.hypot(float(px[-1]) - ox, float(py[-1]) - oy)
        dd = strm_m / 1000.0 / area_km2 if area_km2 > 0 else 0.0

        lfp_km = lfp_m / 1000.0
        perim_km = perimeter_m / 1000.0
        shape_ok = lfp_km > 0
        attrs = {
            "outlet_id": outlet.outlet_id,
            "src_fid": outlet.src_fid,
            "out_type": outlet.kind,
            "mode": mode,
            "ds_ids": ",".join(outlets[p].outlet_id for p in sorted(parents[k])),
            "x_in": meta[k]["x_in"],
            "y_in": meta[k]["y_in"],
            "x_out": float(px[0]) if outlet.kind == "line" else meta[k]["x_out"],
            "y_out": float(py[0]) if outlet.kind == "line" else meta[k]["y_out"],
            "snap_m": _none_round(meta[k]["snap_m"], 2),
            "n_outcells": int(seeds_all[k].size),
            "area_km2": round(area_km2, 6),
            "area_ha": round(area_km2 * 100.0, 4),
            "hole_ha": round(hole_ha, 4),
            "poly_km2": round(poly_km2, 6),
            "perim_km": round(perim_km, 4),
            "n_parts": len(polygons),
            "z_outlet": _none_round(z_outlet, 3),
            "z_min": _none_round(z_min, 3),
            "z_max": _none_round(z_max, 3),
            "z_mean": _none_round(z_mean, 3),
            "relief_m": _none_round((z_max - z_min) if zv.size else None, 3),
            "cx": round(cx, 3),
            "cy": round(cy, 3),
            "lfp_km": round(lfp_km, 4),
            "lca_km": round(lca_m / 1000.0, 4),
            "s_avg": _none_round(s_avg, 6),
            "s_1085": _none_round(s_1085, 6),
            "s_ea": _none_round(s_ea, 6),
            "s_mean_pc": _none_round(s_mean_pc, 3),
            "form_f": _none_round(poly_km2 / lfp_km**2 if shape_ok else None, 4),
            "circ_r": _none_round(
                4 * math.pi * poly_km2 / perim_km**2 if perim_km > 0 else None, 4
            ),
            "elong_r": _none_round(
                2 * math.sqrt(poly_km2 / math.pi) / lfp_km if shape_ok else None, 4
            ),
            "gravel_kc": _none_round(perim_km / (2 * math.sqrt(math.pi * poly_km2)), 4),
            "hyps_int": _none_round(
                (
                    (z_mean - z_min) / (z_max - z_min)
                    if zv.size and z_max > z_min
                    else None
                ),
                4,
            ),
            "strm_km": round(strm_m / 1000.0, 4),
            "dd_kmkm2": _none_round(
                strm_m / 1000.0 / area_km2 if area_km2 > 0 else None, 4
            ),
            "lb_km": round(lb_m / 1000.0, 4),
            "relief_r": _none_round(
                h_rel / lb_m if h_rel is not None and lb_m > 0 else None, 5
            ),
            "melton_r": _none_round(
                h_rel / 1000.0 / math.sqrt(area_km2) if h_rel is not None else None,
                4,
            ),
            "rugged_n": _none_round(
                h_rel / 1000.0 * dd if h_rel is not None and dd > 0 else None, 4
            ),
            "sinuosity": _none_round(lfp_m / straight_m if straight_m > 0 else None, 4),
            "strm_ord": int(order.max()) if order.size else 0,
            "n_strm": int(n_streams),
            "bif_r": _none_round(bif, 3),
            "strm_freq": _none_round(n_streams / area_km2 if area_km2 > 0 else None, 4),
            "lo_km": _none_round(1.0 / (2.0 * dd) if dd > 0 else None, 4),
            "c_maint": _none_round(1.0 / dd if dd > 0 else None, 4),
            "tc_kirp_mn": _none_round(tc_kirpich_minutes(lfp_m, s_avg), 2),
            "tc_usbr_mn": _none_round(tc_usbr_minutes(lfp_km, s_1085), 2),
            "tc_bw_mn": _none_round(
                tc_bransby_williams_minutes(lfp_km, area_km2, s_ea), 2
            ),
        }
        if hole_ha > 0:
            pct = 100.0 * (n_filled - n_cells) / n_filled
            warn(
                f"Catchment '{outlet.outlet_id}': filled {hole_ha:.2f} ha of "
                f"interior holes ({pct:.1f} % of the outline) - cells inside the "
                "outline that do not drain to the outlet."
                + (
                    " That is a large share; the DEM probably needs "
                    "hydrological conditioning."
                    if pct >= 1.0
                    else ""
                )
            )
        if lfp_m > 0 and (s_avg is None or s_avg <= 0):
            warn(
                f"Catchment '{outlet.outlet_id}': the longest flow path does not "
                "fall towards the outlet, so slopes and Tc were not computed."
            )

        result.catchments.append(
            CatchmentResult(
                outlet_id=outlet.outlet_id,
                kind=outlet.kind,
                src_fid=outlet.src_fid,
                attributes=attrs,
                polygons=polygons,
                flow_path=flow_path,
                outlet_xy=(float(px[0]), float(py[0])),
                hypsometric_curve=hypsometric_curve(zm, cell_area),
                detail=_report_detail(
                    geotransform,
                    r0,
                    c0,
                    w,
                    x_prof,
                    z_prof,
                    zv,
                    slope_grid[mask] * 100.0,
                    aspect_grid[mask],
                    slope_grid[mask],
                    order,
                    local_next,
                    orders,
                    (cx, cy),
                    (float(px[int(np.argmin(d2))]), float(py[int(np.argmin(d2))])),
                    stream_threshold_km2,
                ),
            )
        )
        crow, ccol = world_to_cell(cx, cy, geotransform)
        lr, lc = crow - r0, ccol - c0
        inside_c = bool(0 <= lr < h and 0 <= lc < w and filled[lr, lc])
        cz = (
            dem_flat[crow * cols + ccol]
            if 0 <= crow < rows and 0 <= ccol < cols
            else np.nan
        )
        result.catchments[-1].detail.update(
            {
                "stream_links": link_features,
                "centroid_inside": inside_c,
                "centroid_z": None if np.isnan(cz) else float(cz),
            }
        )
        windows.append((r0, c0, filled))

    # --- catchment DEM (union of all filled outlines) -----------------------
    report(0.96, "Building catchment DEM")
    if windows:
        ur0 = min(w_[0] for w_ in windows)
        uc0 = min(w_[1] for w_ in windows)
        ur1 = max(w_[0] + w_[2].shape[0] for w_ in windows)
        uc1 = max(w_[1] + w_[2].shape[1] for w_ in windows)
        union = np.zeros((ur1 - ur0, uc1 - uc0), dtype=bool)
        for r0, c0, filled in windows:
            fh, fw = filled.shape
            union[r0 - ur0 : r0 - ur0 + fh, c0 - uc0 : c0 - uc0 + fw] |= filled
        out_nodata = (
            float(dem_nodata)
            if dem_nodata is not None and not np.isnan(dem_nodata)
            else DEFAULT_DEM_NODATA
        )
        zcrop = dem_flat.reshape(rows, cols)[ur0:ur1, uc0:uc1]
        dem_out = np.where(union & ~np.isnan(zcrop), zcrop, out_nodata)
        result.dem_array = dem_out.astype(np.float32)
        result.dem_nodata = out_nodata
        gt = list(geotransform)
        gt[0] = geotransform[0] + uc0 * geotransform[1]
        gt[3] = geotransform[3] + ur0 * geotransform[5]
        result.dem_geotransform = tuple(gt)
    report(1.0, "Done")
    return result


# ---------------------------------------------------------------------------
# File I/O (GDAL - bundled with QGIS)
# ---------------------------------------------------------------------------
