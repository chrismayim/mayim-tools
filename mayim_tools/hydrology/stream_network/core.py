"""
Core stream-network extraction - pure NumPy, no QGIS dependency (GDAL is
used only by the shared raster I/O helpers).

INPUTS
- A D8 pointer raster (this suite's D8 Flow Direction encoding, or ESRI)
  and the DEM it was built from, on the same grid, in a projected CRS in
  metres. The DEM should be hydrologically conditioned.
- Optional pour points. With pour points, the network is everything
  upstream of them (each point is snapped to the highest flow accumulation
  within a search radius, and also splits the stream into reaches there).
  Without pour points, every drainage network on the raster is extracted,
  one per terminal cell (raster edge, NoData edge or pit).

STREAM DEFINITION
A cell is a stream cell when its upstream area is at least the stream
threshold (km²). A constant-drop test (Tarboton, Bras & Rodriguez-Iturbe
1991) is run over a range of thresholds and reported alongside, so the
chosen threshold can be checked: for each threshold, the mean drop of
first-order Strahler streams is compared with the mean drop of all
higher-order streams with a two-sample t-test, and the smallest threshold
with |t| < 2 is the smallest one consistent with Horton's law of stream
drops.

REACHES (links)
Stream cells are split into reaches at every confluence and at every pour
point. Each reach carries four orderings:
- Strahler (1957) order,
- Shreve (1966) magnitude (number of upstream channel heads),
- Horton (1945) order: Strahler orders re-assigned so that the main
  stream at every confluence (the higher-Strahler branch, then the longer
  one) carries the downstream order all the way to its head,
- Hack (1957) order: the main stem (longest flow path) is order 1, its
  tributaries 2, and so on;
plus topological distance to the outlet, geometry (length, drop, slope,
sinuosity), upstream area and channel-routing inputs (velocity, travel
time, Muskingum K, X and the number of sub-reaches for a computational
time step).

HORTON STATISTICS
Strahler streams (chains of reaches of the same order) give N_u, mean
length, mean upstream area, mean slope and mean drop per order, and the
bifurcation, length, area and slope ratios, both as consecutive-order
means and as log-linear regression fits (Horton's laws).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.d8_network import (
    GridError,
    build_downstream_index,
    cell_center,
    check_geotransform,
    flow_accumulation_cells,
    nodata_mask,
    pointer_to_direction_index,
    resolve_terminals,
    segment_starts,
    snap_to_max_accumulation,
    step_lengths,
    strahler_order,
    stream_links,
    world_to_cell,
)

StreamNetworkError = GridError

VELOCITY_METHODS = ("constant", "manning")
DEFAULT_MIN_SLOPE = 0.0005  # m/m - floor for routing velocities on flat reaches
PROFILE_MAX_POINTS = 1000
DROP_TEST_T = 2.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PourPoint:
    point_id: str
    x: float
    y: float
    src_fid: int = -1


@dataclass
class RoutingSettings:
    method: str = "manning"  # "constant" or "manning"
    velocity_ms: float = 1.0
    manning_n: float = 0.035
    hydraulic_radius_m: float = 0.5
    muskingum_x: float = 0.2
    time_step_min: float = 5.0
    min_slope: float = DEFAULT_MIN_SLOPE


@dataclass
class NetworkResult:
    reaches: list[dict] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    networks: list[dict] = field(default_factory=list)
    order_stats: list[dict] = field(default_factory=list)
    horton_fits: list[dict] = field(default_factory=list)
    drop_test: list[dict] = field(default_factory=list)
    drop_test_threshold_km2: float | None = None
    profiles: dict = field(default_factory=dict)  # network_id -> rows
    main_stems: dict = field(default_factory=dict)  # network_id -> coords
    pour_points: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ordering on the reach graph
# ---------------------------------------------------------------------------


def _children(reaches: Sequence[dict]) -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {r["link_id"]: [] for r in reaches}
    for r in reaches:
        if r["ds_link"] is not None:
            kids[r["ds_link"]].append(r["link_id"])
    return kids


def _postorder(roots: Sequence[int], kids: dict[int, list[int]]) -> list[int]:
    """Upstream reaches before downstream ones (iterative, no recursion)."""
    out, stack = [], [(r, False) for r in roots]
    while stack:
        node, done = stack.pop()
        if done:
            out.append(node)
            continue
        stack.append((node, True))
        stack.extend((k, False) for k in kids[node])
    return out


def order_reaches(reaches: list[dict]) -> None:
    """Adds shreve, up_len_m, horton, hack, links_to_out, dist_out_m and
    is_main_stem to each reach (in place). Needs link_id, ds_link, strahler,
    length_m and us_km2."""
    by_id = {r["link_id"]: r for r in reaches}
    kids = _children(reaches)
    roots = [r["link_id"] for r in reaches if r["ds_link"] is None]
    post = _postorder(roots, kids)

    for lid in post:  # upstream first
        r, ks = by_id[lid], kids[lid]
        r["shreve"] = sum(by_id[k]["shreve"] for k in ks) if ks else 1
        r["up_len_m"] = r["length_m"] + max(
            (by_id[k]["up_len_m"] for k in ks), default=0.0
        )

    def longest(ks):
        return max(ks, key=lambda k: (by_id[k]["up_len_m"], by_id[k]["us_km2"]))

    def horton_main(ks):
        return max(
            ks,
            key=lambda k: (
                by_id[k]["strahler"],
                by_id[k]["up_len_m"],
                by_id[k]["us_km2"],
            ),
        )

    for lid in reversed(post):  # downstream first
        r = by_id[lid]
        if r["ds_link"] is None:
            r["horton"] = r["strahler"]
            r["hack"] = 1
            r["links_to_out"] = 1
            r["dist_out_m"] = 0.0
        ks = kids[lid]
        if not ks:
            continue
        h_main, k_main = horton_main(ks), longest(ks)
        for k in ks:
            c = by_id[k]
            c["horton"] = r["horton"] if k == h_main else c["strahler"]
            c["hack"] = r["hack"] if k == k_main else r["hack"] + 1
            c["links_to_out"] = r["links_to_out"] + 1
            c["dist_out_m"] = r["dist_out_m"] + r["length_m"]
    for r in reaches:
        r["is_main"] = int(r["hack"] == 1)


# ---------------------------------------------------------------------------
# Strahler streams and Horton statistics
# ---------------------------------------------------------------------------


def strahler_streams(reaches: Sequence[dict]) -> list[dict]:
    """Chains reaches into Strahler streams (maximal runs of the same
    order). Each stream: network_id, order, length_m, drop_m, slope,
    us_km2 (at its downstream end), n_reaches."""
    by_id = {r["link_id"]: r for r in reaches}
    kids = _children(reaches)
    streams = []
    for r in reaches:
        # a stream starts at a reach with no upstream reach of the same order
        if any(by_id[k]["strahler"] == r["strahler"] for k in kids[r["link_id"]]):
            continue
        chain, cur = [r], r
        while cur["ds_link"] is not None:
            nxt = by_id[cur["ds_link"]]
            if nxt["strahler"] != r["strahler"]:
                break
            chain.append(nxt)
            cur = nxt
        length = sum(c["length_m"] for c in chain)
        z0, z1 = chain[0]["z_start"], chain[-1]["z_end"]
        drop = None if z0 is None or z1 is None else z0 - z1
        streams.append(
            {
                "network_id": r["network_id"],
                "order": r["strahler"],
                "length_m": length,
                "drop_m": drop,
                "slope": drop / length if drop is not None and length > 0 else None,
                "us_km2": chain[-1]["us_km2"],
                "n_reaches": len(chain),
            }
        )
    return streams


def _loglinear(orders, values, decreasing=False):
    """Fits log10(value) = a + b * order; returns (ratio, r2) where ratio
    is 10**b (or 10**-b for quantities that fall with order)."""
    pairs = [(u, v) for u, v in zip(orders, values, strict=True) if v and v > 0]
    if len(pairs) < 2:
        return None, None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.log10([p[1] for p in pairs])
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else None
    return float(10 ** (-b if decreasing else b)), r2


def horton_statistics(streams: Sequence[dict], network_id: str):
    """Per-order rows and the ratio/regression summary for one network."""
    mine = [s for s in streams if s["network_id"] == network_id]
    if not mine:
        return [], None
    max_u = max(s["order"] for s in mine)
    rows = []
    for u in range(1, max_u + 1):
        su = [s for s in mine if s["order"] == u]
        if not su:
            continue
        slopes = [s["slope"] for s in su if s["slope"] is not None]
        drops = [s["drop_m"] for s in su if s["drop_m"] is not None]
        rows.append(
            {
                "network_id": network_id,
                "order": u,
                "n_streams": len(su),
                "total_length_km": sum(s["length_m"] for s in su) / 1000.0,
                "mean_length_km": float(np.mean([s["length_m"] for s in su])) / 1000.0,
                "mean_area_km2": float(np.mean([s["us_km2"] for s in su])),
                "mean_slope": float(np.mean(slopes)) if slopes else None,
                "mean_drop_m": float(np.mean(drops)) if drops else None,
            }
        )
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        a["rb_next"] = a["n_streams"] / b["n_streams"] if b["n_streams"] else None
        a["rl_next"] = (
            b["mean_length_km"] / a["mean_length_km"] if a["mean_length_km"] else None
        )
        a["ra_next"] = (
            b["mean_area_km2"] / a["mean_area_km2"] if a["mean_area_km2"] else None
        )
        a["rs_next"] = (
            a["mean_slope"] / b["mean_slope"]
            if a["mean_slope"] and b["mean_slope"]
            else None
        )
    if rows:
        for key in ("rb_next", "rl_next", "ra_next", "rs_next"):
            rows[-1][key] = None

    orders = [r["order"] for r in rows]

    def mean_of(key):
        vals = [r[key] for r in rows if r.get(key)]
        return float(np.mean(vals)) if vals else None

    rb, rb_r2 = _loglinear(orders, [r["n_streams"] for r in rows], decreasing=True)
    rl, rl_r2 = _loglinear(orders, [r["mean_length_km"] for r in rows])
    ra, ra_r2 = _loglinear(orders, [r["mean_area_km2"] for r in rows])
    rs, rs_r2 = _loglinear(orders, [r["mean_slope"] for r in rows], decreasing=True)
    fit = {
        "network_id": network_id,
        "max_order": max_u,
        "rb_mean": mean_of("rb_next"),
        "rl_mean": mean_of("rl_next"),
        "ra_mean": mean_of("ra_next"),
        "rs_mean": mean_of("rs_next"),
        "rb_fit": rb,
        "rb_r2": rb_r2,
        "rl_fit": rl,
        "rl_r2": rl_r2,
        "ra_fit": ra,
        "ra_r2": ra_r2,
        "rs_fit": rs,
        "rs_r2": rs_r2,
    }
    return rows, fit


def hack_exponent(reaches: Sequence[dict], network_id: str):
    """Hack's law L = c A^h fitted to (upstream main length, upstream area)
    over the reaches of one network. Returns (h, c, r2) or Nones."""
    pts = [
        (r["us_km2"], r["up_len_m"] / 1000.0)
        for r in reaches
        if r["network_id"] == network_id and r["us_km2"] > 0 and r["up_len_m"] > 0
    ]
    if len(pts) < 5:
        return None, None, None
    x = np.log10([p[0] for p in pts])
    y = np.log10([p[1] for p in pts])
    if np.ptp(x) == 0:
        return None, None, None
    h, logc = np.polyfit(x, y, 1)
    pred = logc + h * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else None
    return float(h), float(10**logc), r2


# ---------------------------------------------------------------------------
# Constant-drop test
# ---------------------------------------------------------------------------


def segment_drops(
    nxt: np.ndarray, stream: np.ndarray, order: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop (m) and order of every Strahler stream on a cell network.
    A stream runs from its first cell to the junction where its order
    increases (or to its terminal); drop = z(first) - z(end)."""
    n = nxt.size
    cells = np.flatnonzero(stream)
    if cells.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int32)
    compact = np.full(n, -1, dtype=np.int64)
    compact[cells] = np.arange(cells.size)
    down = nxt[cells]
    same = (down != cells) & stream[down] & (order[down] == order[cells])
    nxt_c = np.where(same, compact[down], np.arange(cells.size))
    term, _, _ = resolve_terminals(nxt_c)
    max_in = segment_starts(nxt, order)
    starts_c = np.flatnonzero(max_in[cells] < order[cells])
    end_cells = cells[term[starts_c]]
    after = nxt[end_cells]
    end_z = np.where(
        (after != end_cells) & stream[after],
        z[after],
        z[end_cells],
    )
    start_cells = cells[starts_c]
    drops = z[start_cells] - end_z
    ok = ~np.isnan(drops)
    return drops[ok], order[start_cells][ok]


def welch_t(a: np.ndarray, b: np.ndarray) -> float | None:
    """Two-sample t statistic with unequal variances."""
    if a.size < 2 or b.size < 2:
        return None
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    if va + vb <= 0:
        return None
    return float((a.mean() - b.mean()) / math.sqrt(va + vb))


def constant_drop_test(
    nxt: np.ndarray,
    domain: np.ndarray,
    acc_km2: np.ndarray,
    z: np.ndarray,
    thresholds_km2: Sequence[float],
    is_canceled: Callable[[], bool] | None = None,
) -> tuple[list[dict], float | None]:
    """Runs the drop test for each threshold. Returns (rows, smallest
    threshold with |t| < 2, or None)."""
    rows = []
    for thr in thresholds_km2:
        if is_canceled is not None and is_canceled():
            break
        stream = domain & (acc_km2 >= thr)
        order = strahler_order(nxt, stream)
        drops, orders = segment_drops(nxt, stream, order, z)
        low, high = drops[orders == 1], drops[orders > 1]
        t = welch_t(low, high)
        rows.append(
            {
                "threshold_km2": float(thr),
                "max_order": int(orders.max()) if orders.size else 0,
                "n_first_order": int(low.size),
                "n_higher_order": int(high.size),
                "mean_drop_first_m": float(low.mean()) if low.size else None,
                "mean_drop_higher_m": float(high.mean()) if high.size else None,
                "sd_drop_first_m": float(low.std(ddof=1)) if low.size > 1 else None,
                "sd_drop_higher_m": float(high.std(ddof=1)) if high.size > 1 else None,
                "t_statistic": t,
                "passes": t is not None and abs(t) < DROP_TEST_T,
            }
        )
    passing = [r["threshold_km2"] for r in rows if r["passes"]]
    return rows, (min(passing) if passing else None)


def threshold_series(min_km2: float, max_km2: float, n: int) -> list[float]:
    if min_km2 <= 0 or max_km2 <= min_km2 or n < 2:
        raise StreamNetworkError(
            "Drop test range needs 0 < minimum < maximum and at least 2 steps."
        )
    return [float(v) for v in np.geomspace(min_km2, max_km2, n)]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def reach_routing(length_m: float, slope, settings: RoutingSettings) -> dict:
    """Velocity, travel time, Muskingum K/X and sub-reaches for one reach."""
    s = max(slope if slope is not None else 0.0, settings.min_slope)
    if settings.method == "constant":
        v = settings.velocity_ms
    else:
        v = (
            settings.hydraulic_radius_m ** (2.0 / 3.0)
            * math.sqrt(s)
            / settings.manning_n
        )
    if v <= 0:
        return {
            "v_ms": None,
            "tt_min": None,
            "musk_k_h": None,
            "musk_x": settings.muskingum_x,
            "n_sub": None,
            "dt_check": "n/a",
        }
    tt_min = length_m / v / 60.0
    k_h = tt_min / 60.0
    dt = settings.time_step_min
    x = settings.muskingum_x
    n_sub = max(1, round(tt_min / dt)) if dt > 0 else 1
    k_sub = tt_min / n_sub  # minutes
    if dt < 2 * k_sub * x:
        check = "dt below 2KX: use fewer sub-reaches"
    elif dt > 2 * k_sub * (1 - x):
        check = "dt above 2K(1-X): reach too short for dt (merge reaches or shorten dt)"
    else:
        check = "ok"
    return {
        "v_ms": v,
        "tt_min": tt_min,
        "musk_k_h": k_h,
        "musk_x": x,
        "n_sub": n_sub,
        "dt_check": check,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _r(value, nd):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return round(float(value), nd)


def extract_stream_network(
    pointer: np.ndarray,
    pointer_nodata,
    dem: np.ndarray,
    dem_nodata,
    geotransform,
    pour_points: Sequence[PourPoint] = (),
    esri_style: bool = False,
    snap_radius_cells: int = 3,
    threshold_km2: float = 0.1,
    drop_test_range: tuple[float, float, int] | None = (0.01, 2.0, 12),
    routing: RoutingSettings | None = None,
    min_network_km2: float = 0.0,
    progress: Callable[[float, str], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
    outlet_cells: Sequence[tuple[str, Sequence[int]]] | None = None,
) -> NetworkResult:
    """Extracts, orders and characterises the stream network.

    ``outlet_cells`` is an alternative to ``pour_points`` for callers that
    have already placed their outlets (e.g. Catchment Delineation, whose
    pour points are snapped and whose pour lines cover many cells): a list
    of (name, flat cell indices). The network is then everything upstream
    of those cells, exactly as with pour points."""
    routing = routing or RoutingSettings()
    if routing.method not in VELOCITY_METHODS:
        raise StreamNetworkError(f"Unknown velocity method {routing.method!r}")
    if threshold_km2 <= 0:
        raise StreamNetworkError("The stream threshold must be greater than 0.")
    sx, sy = check_geotransform(geotransform)
    pointer = np.asarray(pointer)
    dem = np.asarray(dem, dtype=np.float64)
    if pointer.shape != dem.shape:
        raise StreamNetworkError("D8 pointer and DEM are not on the same grid.")
    rows, cols = pointer.shape
    cell_km2 = sx * sy / 1e6
    res = NetworkResult()
    warn = res.warnings.append

    def report(f, msg):
        if progress is not None:
            progress(f, msg)

    def canceled():
        return is_canceled is not None and is_canceled()

    report(0.02, "Decoding D8 pointer")
    direction, n_invalid = pointer_to_direction_index(
        pointer, pointer_nodata, esri_style
    )
    if n_invalid:
        warn(
            f"{n_invalid} pointer cell(s) hold values that are not valid "
            f"{'ESRI' if esri_style else 'WhiteboxTools'} D8 codes and were "
            "treated as NoData. Check the pointer encoding setting."
        )
    valid = (direction != -2).ravel()
    z = np.where(nodata_mask(dem, dem_nodata), np.nan, dem).ravel()
    nxt = build_downstream_index(direction)
    idx = np.arange(nxt.size, dtype=np.int64)
    steps = step_lengths(direction, sx, sy)
    steps[nxt == idx] = 0.0

    report(0.08, "Computing flow accumulation")
    acc, n_loop = flow_accumulation_cells(nxt, valid)
    if n_loop:
        warn(
            f"{n_loop} cell(s) are caught in or drain into a flow loop and were "
            "excluded; the pointer raster is probably not a valid D8 pointer."
        )
    acc_km2 = np.where(np.isnan(acc), 0.0, acc) * cell_km2
    acc2d = acc.reshape(rows, cols)
    if canceled():
        return res

    # --- domain and network graph ---------------------------------------
    report(0.15, "Defining the drainage domain")
    seed_owner = np.zeros(nxt.size, dtype=np.int32)
    names: list[str] = []
    for p in pour_points:
        r, c = world_to_cell(p.x, p.y, geotransform)
        snapped = snap_to_max_accumulation(r, c, acc2d, snap_radius_cells)
        if snapped is None:
            warn(f"Pour point '{p.point_id}' is off the grid or on NoData; skipped.")
            continue
        flat = snapped[0] * cols + snapped[1]
        if seed_owner[flat]:
            warn(
                f"Pour point '{p.point_id}' snaps to the same cell as "
                f"'{names[seed_owner[flat] - 1]}'; skipped."
            )
            continue
        names.append(p.point_id)
        seed_owner[flat] = len(names)
        xo, yo = cell_center(*snapped, geotransform)
        res.pour_points.append(
            {
                "point_id": p.point_id,
                "src_fid": p.src_fid,
                "x_in": p.x,
                "y_in": p.y,
                "x": float(xo),
                "y": float(yo),
                "snap_m": math.hypot(float(xo) - p.x, float(yo) - p.y),
                "us_km2": float(acc_km2[flat]),
            }
        )
    for name, cells in outlet_cells or ():
        cells = np.asarray(cells, dtype=np.int64)
        cells = cells[(cells >= 0) & (cells < nxt.size)]
        cells = cells[seed_owner[cells] == 0]
        if cells.size == 0:
            continue
        names.append(str(name))
        seed_owner[cells] = len(names)
    seeds = np.flatnonzero(seed_owner)

    if pour_points or outlet_cells:
        if seeds.size == 0:
            raise StreamNetworkError(
                "None of the pour points/outlets fall on the grid."
            )
        nxt_seeded = nxt.copy()
        nxt_seeded[seeds] = seeds
        term_s, _, unres = resolve_terminals(nxt_seeded)
        first_outlet = seed_owner[term_s]
        first_outlet[unres | ~valid] = 0
        domain = first_outlet > 0
        net_next = nxt.copy()
        leaves = ~domain[net_next] | (net_next == idx)
        net_next[leaves] = idx[leaves]
        net_next[~domain] = idx[~domain]
    else:
        domain = valid.copy()
        first_outlet = np.zeros(nxt.size, dtype=np.int32)
        net_next = nxt
    term, _, unres_net = resolve_terminals(net_next)
    domain &= ~unres_net

    # --- drop test -----------------------------------------------------------
    if drop_test_range is not None:
        report(0.20, "Running the constant-drop test")
        thresholds = threshold_series(*drop_test_range)
        res.drop_test, res.drop_test_threshold_km2 = constant_drop_test(
            net_next, domain, acc_km2, z, thresholds, is_canceled
        )
        if res.drop_test_threshold_km2 is None:
            warn(
                "No threshold in the drop-test range passed (|t| < 2). Widen the "
                "range, or check that the DEM is conditioned."
            )
        elif threshold_km2 < res.drop_test_threshold_km2:
            warn(
                f"The chosen threshold ({threshold_km2:g} km²) is below the "
                f"smallest that passes the drop test "
                f"({res.drop_test_threshold_km2:.4g} km²); first-order streams "
                "may be over-extended into hillslopes."
            )
    if canceled():
        return res

    # --- streams, ordering and reaches -------------------------------------
    report(0.55, "Ordering streams")
    stream = domain & (acc_km2 >= threshold_km2)
    if not stream.any():
        raise StreamNetworkError(
            "No stream cells at this threshold - lower the stream threshold."
        )
    order = strahler_order(net_next, stream)
    # intermediate pour points split reaches; a root pour point is the outlet
    breaks = (seed_owner > 0) & (net_next != idx)
    links = stream_links(net_next, stream, order, steps, break_cells=breaks)

    # networks: named by root pour point, else N1, N2... by area
    roots = {}
    for lk in links:
        if lk["ds_link"] is None:
            t = int(term[lk["cells"][0]])
            roots[t] = lk
    root_cells = sorted(roots, key=lambda t: -acc_km2[t])
    net_name = {}
    for i, t in enumerate(root_cells):
        own = int(seed_owner[t])
        net_name[t] = names[own - 1] if own else f"N{i + 1}"
    keep_nets = {t for t in root_cells if acc_km2[t] >= min_network_km2}
    if len(keep_nets) < len(root_cells):
        warn(
            f"{len(root_cells) - len(keep_nets)} network(s) smaller than "
            f"{min_network_km2:g} km² were left out."
        )

    report(0.65, "Building reaches")
    reaches = []
    for lk in links:
        t = int(term[lk["cells"][0]])
        if t not in keep_nets:
            continue
        cells = np.asarray(lk["cells"])
        xs, ys = cell_center(cells // cols, cells % cols, geotransform)
        end_cell = cells[-1]
        z0 = z[cells[0]]
        z1 = z[end_cell]
        straight = math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]))
        length = lk["length_m"]
        drop = None if np.isnan(z0) or np.isnan(z1) else float(z0 - z1)
        slope = drop / length if drop is not None and length > 0 else None
        fo = int(first_outlet[lk["last_own"]])
        reaches.append(
            {
                "link_id": lk["link_id"],
                "ds_link": lk["ds_link"],
                "network_id": net_name[t],
                "outlet_id": names[fo - 1] if fo else net_name[t],
                "strahler": lk["order"],
                "length_m": length,
                "us_km2": float(acc_km2[lk["last_own"]]),
                "z_start": None if np.isnan(z0) else float(z0),
                "z_end": None if np.isnan(z1) else float(z1),
                "drop_m": drop,
                "slope": slope,
                "sinuosity": length / straight if straight > 0 else None,
                "coords": list(zip(xs.tolist(), ys.tolist(), strict=True)),
                "_cells": cells,
                "_start_cell": int(cells[0]),
                "_end_cell": int(end_cell),
                "_root": t,
            }
        )
    if not reaches:
        raise StreamNetworkError("No stream network left after filtering.")
    order_reaches(reaches)
    for r in reaches:
        r.update(reach_routing(r["length_m"], r["slope"], routing))
    if any(r["slope"] is not None and r["slope"] <= 0 for r in reaches):
        n_flat = sum(1 for r in reaches if r["slope"] is not None and r["slope"] <= 0)
        warn(
            f"{n_flat} reach(es) have zero or negative slope (flat DEM areas). "
            f"Routing used a minimum slope of {routing.min_slope:g} m/m for them."
        )

    # --- nodes -----------------------------------------------------------------
    report(0.75, "Building nodes")
    n_in = np.zeros(nxt.size, dtype=np.int32)
    src = np.flatnonzero(stream & (net_next != idx))
    dn = net_next[src]
    np.add.at(n_in, dn[stream[dn]], 1)
    nodes = []
    for r in reaches:
        c = r["_start_cell"]
        if n_in[c] == 0:
            kind = "channel head"
        elif n_in[c] >= 2:
            kind = "confluence"
        else:
            kind = "pour point"
        x, y = r["coords"][0]
        nodes.append(
            {
                "node_type": kind,
                "network_id": r["network_id"],
                "link_id": r["link_id"],
                "x": x,
                "y": y,
                "z": r["z_start"],
                "us_km2": float(acc_km2[c]),
                "n_inflow": int(n_in[c]),
                "strahler": r["strahler"],
                "shreve": r["shreve"],
            }
        )
        if r["ds_link"] is None:
            x, y = r["coords"][-1]
            nodes.append(
                {
                    "node_type": "outlet",
                    "network_id": r["network_id"],
                    "link_id": r["link_id"],
                    "x": x,
                    "y": y,
                    "z": r["z_end"],
                    "us_km2": float(acc_km2[r["_end_cell"]]),
                    "n_inflow": int(n_in[r["_end_cell"]]),
                    "strahler": r["strahler"],
                    "shreve": r["shreve"],
                }
            )
    for i, nd in enumerate(nodes, start=1):
        nd["node_id"] = i
    res.nodes = nodes

    # --- networks, Horton, main stems -------------------------------------
    report(0.85, "Network statistics")
    streams_all = strahler_streams(reaches)
    for t in [t for t in root_cells if t in keep_nets]:
        nid = net_name[t]
        mine = [r for r in reaches if r["_root"] == t]
        root = next(r for r in mine if r["ds_link"] is None)
        area = float(acc_km2[t])
        total_km = sum(r["length_m"] for r in mine) / 1000.0
        rows_u, fit = horton_statistics(streams_all, nid)
        res.order_stats.extend(rows_u)
        if fit:
            res.horton_fits.append(fit)
        h, hc, hr2 = hack_exponent(mine, nid)

        # main stem: follow Hack-1 reaches upstream from the root
        kids_of: dict[int, list[dict]] = {}
        for r in mine:
            if r["ds_link"] is not None:
                kids_of.setdefault(r["ds_link"], []).append(r)
        chain, cur = [root], root
        while True:
            nxt_r = next(
                (k for k in kids_of.get(cur["link_id"], []) if k["hack"] == 1), None
            )
            if nxt_r is None:
                break
            chain.append(nxt_r)
            cur = nxt_r
        chain.reverse()  # head first
        stem = np.concatenate(
            [r["_cells"] if r["ds_link"] is None else r["_cells"][:-1] for r in chain]
        )
        sx_, sy_ = cell_center(stem // cols, stem % cols, geotransform)
        res.main_stems[nid] = list(zip(sx_.tolist(), sy_.tolist(), strict=True))
        seg = steps[stem]
        seg[-1] = 0.0
        d_out = np.cumsum(seg[::-1])[::-1]  # distance to outlet per cell
        z_stem = z[stem]
        main_len = float(d_out[0])
        pick = np.arange(stem.size)
        if stem.size > PROFILE_MAX_POINTS:
            pick = np.unique(
                np.linspace(0, stem.size - 1, PROFILE_MAX_POINTS).round().astype(int)
            )
        res.profiles[nid] = [
            {
                "network_id": nid,
                "dist_from_outlet_m": float(d_out[i]),
                "z_m": None if np.isnan(z_stem[i]) else float(z_stem[i]),
            }
            for i in pick
        ]
        head_z = None if np.isnan(z_stem[0]) else float(z_stem[0])
        out_z = None if np.isnan(z_stem[-1]) else float(z_stem[-1])
        x_prof, z_prof = d_out[::-1], z_stem[::-1]  # outlet first
        s1085 = None
        if x_prof.size > 1 and not np.isnan(z_prof).any() and x_prof[-1] > 0:
            z10 = np.interp(0.10 * x_prof[-1], x_prof, z_prof)
            z85 = np.interp(0.85 * x_prof[-1], x_prof, z_prof)
            s1085 = float((z85 - z10) / (0.75 * x_prof[-1]))
        zs = z_stem[~np.isnan(z_stem)]
        hx, hy = chain[0]["coords"][0]
        ox, oy = root["coords"][-1]
        straight = math.hypot(hx - ox, hy - oy)
        heads = sum(1 for r in mine if r["link_id"] not in kids_of)
        res.networks.append(
            {
                "network_id": nid,
                "outlet_x": ox,
                "outlet_y": oy,
                "outlet_z": out_z,
                "area_km2": area,
                "n_reaches": len(mine),
                "n_heads": heads,
                "max_strahler": max(r["strahler"] for r in mine),
                "magnitude": root["shreve"],
                "total_length_km": total_km,
                "drainage_density": total_km / area if area > 0 else None,
                "stream_frequency": heads / area if area > 0 else None,
                "main_length_km": main_len / 1000.0,
                "main_relief_m": (
                    (head_z - out_z)
                    if head_z is not None and out_z is not None
                    else None
                ),
                "main_slope_avg": (
                    (head_z - out_z) / main_len
                    if head_z is not None and out_z is not None and main_len > 0
                    else None
                ),
                "main_slope_1085": s1085,
                "main_sinuosity": main_len / straight if straight > 0 else None,
                "main_tt_min": sum(r["tt_min"] or 0.0 for r in chain),
                "hack_h": h,
                "hack_c": hc,
                "hack_r2": hr2,
                "z_max_stream": float(zs.max()) if zs.size else None,
            }
        )

    # tidy reach output
    for r in reaches:
        for key in ("length_m", "drop_m", "z_start", "z_end", "dist_out_m", "up_len_m"):
            r[key] = _r(r.get(key), 3)
        r["us_km2"] = _r(r["us_km2"], 6)
        r["slope"] = _r(r["slope"], 6)
        r["sinuosity"] = _r(r["sinuosity"], 4)
        r["v_ms"] = _r(r["v_ms"], 4)
        r["tt_min"] = _r(r["tt_min"], 3)
        r["musk_k_h"] = _r(r["musk_k_h"], 5)
    res.reaches = reaches
    res.settings = {
        "threshold_km2": threshold_km2,
        "drop_test_range": drop_test_range,
        "routing": routing,
        "snap_radius_cells": snap_radius_cells,
        "cell_size": (sx, sy),
        "grid": (cols, rows),
        "n_pour_points": len(res.pour_points),
        "min_network_km2": min_network_km2,
    }
    report(1.0, "Done")
    return res
