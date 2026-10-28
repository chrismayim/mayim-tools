"""Time of Concentration - calculation core (no QGIS dependency).

Per catchment:

- Empirical methods on the whole longest flow path L (no parameters):
  Kirpich (average slope), USBR / SANRAL DM Eq 3.12 (10-85 slope) and
  Bransby-Williams (equal-area slope).
- Combined Tc: L split into three parts, measured from the outlet x = 0 to
  the divide x = L:
    overland (sheet) flow   x from L - L_ov to L           Kerby, DM Eq 3.11
    shallow concentrated    x from L_ch to L - L_ov        NRCS V = k S^0.5
    channel                 x from 0 to L_ch               DM Eq 3.12
  L_ch is the channel start (Catchment Delineation's ch_head_m, or the
  user's ch_start_m); L_ov the overland length (never past the channel
  start). Each part takes its slope from the DEM profile along the path.
- Selected Tc: the method chosen per catchment (tc_method).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import tables as tb

TOOL_VERSION = "0.1.0"


# ---------------------------------------------------------------- methods ---
def tc_kirpich_minutes(length_m: float, slope: float | None) -> float | None:
    """Kirpich (1940), metric: tc = 0.0195 L^0.77 S^-0.385 (min, L m, S m/m)."""
    if slope is None or slope <= 0 or length_m <= 0:
        return None
    return 0.0195 * length_m**0.77 * slope**-0.385


def tc_usbr_minutes(length_km: float, slope: float | None) -> float | None:
    """USBR (1973), SANRAL DM Eq 3.12 for defined watercourses:
    tc = (0.87 L^2 / (1000 S))^0.385 hours (L km, S m/m)."""
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


def tc_kerby_minutes(length_m: float, r: float, slope: float) -> float:
    """Kerby (1959), DM Eq 3.11: tc = 0.604 (r L / S^0.5)^0.467 hours
    (L km, S m/m), here in minutes with L in m."""
    return 60.0 * 0.604 * (r * length_m / 1000.0 / math.sqrt(slope)) ** 0.467


def shallow_velocity(k: float, slope: float) -> float:
    """NRCS shallow concentrated flow, V = k S^0.5 (m/s; k in m/s)."""
    return k * math.sqrt(slope)


def tc_shallow_minutes(length_m: float, k: float, slope: float) -> float:
    return length_m / shallow_velocity(k, slope) / 60.0


# ---------------------------------------------------------------- profile ---
def _z_at(x: np.ndarray, z: np.ndarray, xi: float) -> float:
    return float(np.interp(xi, x, z))


def average_slope(x, z, x0: float, x1: float) -> float | None:
    """(z(x1) - z(x0)) / (x1 - x0) on a profile with x from the outlet."""
    if x1 <= x0:
        return None
    return (_z_at(x, z, x1) - _z_at(x, z, x0)) / (x1 - x0)


def slope_10_85(x, z, x0: float, x1: float) -> float | None:
    """10-85 slope of the reach x0 (downstream end) to x1 (DM Eq 3.13)."""
    length = x1 - x0
    if length <= 0:
        return None
    z10 = _z_at(x, z, x0 + 0.10 * length)
    z85 = _z_at(x, z, x0 + 0.85 * length)
    return (z85 - z10) / (0.75 * length)


def equal_area_slope(x, z) -> float | None:
    if len(x) < 2 or x[-1] <= 0:
        return None
    zr = z - z[0]
    area = float(np.sum(0.5 * (zr[1:] + zr[:-1]) * np.diff(x)))
    return 2.0 * area / float(x[-1]) ** 2


def profile_from_line(vertices_xy, z_values) -> tuple[np.ndarray, np.ndarray]:
    """(x from the outlet, z), ascending in x, from a flow-path line drawn
    divide -> outlet (Catchment Delineation's longest flow path) with the
    DEM value at each vertex. Vertices with no DEM value are dropped."""
    pts = np.asarray(vertices_xy, dtype=float)
    zs = np.asarray(z_values, dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])  # from the divide
    x = cum[-1] - cum
    order = np.argsort(x, kind="stable")
    x, zs = x[order], zs[order]
    ok = np.isfinite(zs)
    return x[ok], zs[ok]


# ----------------------------------------------------------------- inputs ---
@dataclass
class Params:
    """Per-catchment parameters (fields of the catchment layer)."""

    ov_len_m: float | None = None
    ov_r: float | None = None
    sc_type: str | None = None
    ch_start_m: float | None = None
    tc_method: str | None = None
    tc_user_mn: float | None = None


@dataclass
class CatchmentTc:
    catchment_id: str
    area_km2: float | None
    lfp_km: float | None
    s_avg: float | None = None
    s_1085: float | None = None
    s_ea: float | None = None
    ch_head_m: float | None = None  # measured by Catchment Delineation
    x: np.ndarray | None = None  # profile along the LFP, from the outlet
    z: np.ndarray | None = None
    params: Params = field(default_factory=Params)
    lc_overland: list = field(default_factory=list)  # land-cover codes
    lc_shallow: list = field(default_factory=list)
    # results
    values: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    missing: list = field(default_factory=list)


@dataclass
class Settings:
    default_ov_len_m: float = 100.0  # fills an empty ov_len_m (0: leave empty)
    fill_from_land_cover: bool = False  # fill empty ov_r / sc_type
    default_method: str = "combined"  # for an empty tc_method
    classification: str = "sanlc"


# ------------------------------------------------------------ suggestions ---
def suggest(codes, classification: str) -> tuple:
    """(r, sc_type, majority code, share %) for the land-cover codes along a
    part of the path: the most common class that has a suggestion."""
    table = tb.LAND_COVER_CLASSIFICATIONS[classification][1]
    counts = Counter(int(c) for c in codes if c is not None and int(c) in table)
    if not counts:
        return None, None, None, None
    code, n = counts.most_common(1)[0]
    r, sc = table[code]
    return r, sc, code, 100.0 * n / len(codes)


# ------------------------------------------------------------- calculate ---
def fill_parameters(c: CatchmentTc, s: Settings) -> list[str]:
    """Fills empty parameters that the run may fill; returns notes of what
    was filled (they are written to the layer, so the user can see and
    change them before the next run)."""
    p = c.params
    notes = []
    if p.ch_start_m is None and c.ch_head_m is not None:
        p.ch_start_m = round(float(c.ch_head_m), 1)
        notes.append("ch_start_m from the measured channel start")
    if p.ov_len_m is None and s.default_ov_len_m > 0:
        p.ov_len_m = float(s.default_ov_len_m)
        notes.append(f"ov_len_m set to the default {s.default_ov_len_m:g} m")
    if s.fill_from_land_cover:
        r_ov = suggest(c.lc_overland, s.classification)[0]
        sc_sh = suggest(c.lc_shallow, s.classification)[1]
        if p.ov_r is None and r_ov is not None:
            p.ov_r = r_ov
            notes.append("ov_r from the land-cover suggestion")
        if p.sc_type is None and sc_sh is not None:
            p.sc_type = sc_sh
            notes.append("sc_type from the land-cover suggestion")
    return notes


def calculate(c: CatchmentTc, s: Settings) -> CatchmentTc:
    """Fills c.values (field -> value), c.flags and c.missing."""
    v = c.values
    length_m = (c.lfp_km or 0.0) * 1000.0
    v["tc_kirp_mn"] = tc_kirpich_minutes(length_m, c.s_avg)
    v["tc_usbr_mn"] = tc_usbr_minutes(c.lfp_km or 0.0, c.s_1085)
    v["tc_bw_mn"] = tc_bransby_williams_minutes(
        c.lfp_km or 0.0, c.area_km2 or 0.0, c.s_ea
    )

    # suggestions (always reported when land cover is given)
    r_ov, _, code_ov, share_ov = suggest(c.lc_overland, s.classification)
    _, sc_sh, code_sh, share_sh = suggest(c.lc_shallow, s.classification)
    v["sug_r"] = r_ov
    v["sug_sc"] = sc_sh
    v["lc_ov"] = code_ov
    v["lc_sc"] = code_sh
    v["lc_ov_pc"] = share_ov
    v["lc_sc_pc"] = share_sh

    _combined(c, length_m)
    _select(c, s)
    return c


def _combined(c: CatchmentTc, length_m: float) -> None:
    v, p, flags = c.values, c.params, c.flags
    for k in ("l_ov_m", "s_ov", "tc_ov_mn", "l_sc_m", "s_sc", "v_sc_ms",
              "tc_sc_mn", "l_ch_m", "s_ch", "tc_ch_mn", "tc_comb_mn"):  # fmt: skip
        v[k] = None
    if c.x is None or len(c.x) < 2 or length_m <= 0:
        c.missing.append("no profile along the longest flow path")
        return
    x, z = c.x, c.z
    big_l = float(x[-1])
    if abs(big_l - length_m) > max(1.0, 0.01 * length_m):
        flags.append(
            f"flow path {big_l:.0f} m long, lfp_km gives {length_m:.0f} m: "
            "the path's own length is used"
        )
    # channel start
    ch = p.ch_start_m if p.ch_start_m is not None else c.ch_head_m
    if ch is None:
        ch = 0.0
        flags.append("no channel start: no channel part")
    ch = min(max(ch, 0.0), big_l)
    if p.ov_len_m is None:
        c.missing.append("ov_len_m")
        return
    ov = min(max(p.ov_len_m, 0.0), big_l - ch)
    if ov < p.ov_len_m:
        flags.append(
            f"overland length {p.ov_len_m:g} m cut to {ov:.0f} m at the channel "
            "start"
        )
    if ov > tb.KERBY_MAX_LENGTH_M:
        flags.append(
            f"overland length {ov:.0f} m exceeds the {tb.KERBY_MAX_LENGTH_M:.0f} m "
            "Kerby's data covered"
        )
    sc_len = big_l - ch - ov
    if ov > 0 and p.ov_r is None:
        c.missing.append("ov_r")
    if sc_len > 0 and p.sc_type is None:
        c.missing.append("sc_type")
    if sc_len > 0 and p.sc_type is not None and p.sc_type not in tb.SC_TYPES:
        c.missing.append(f"sc_type '{p.sc_type}' unknown")
    v["l_ov_m"], v["l_sc_m"], v["l_ch_m"] = ov, sc_len, ch

    def floored(sl, floor, part):
        if sl is None:
            return None
        if sl < floor:
            flags.append(f"{part} slope {100 * sl:.2f} % raised to {100 * floor:.2f} %")
            return floor
        return sl

    total = 0.0
    ok = not c.missing
    if ov > 0:
        s_ov = floored(average_slope(x, z, big_l - ov, big_l), tb.S_MIN, "overland")
        v["s_ov"] = s_ov
        if p.ov_r is not None:
            v["tc_ov_mn"] = tc_kerby_minutes(ov, p.ov_r, s_ov)
            total += v["tc_ov_mn"]
    if sc_len > 0:
        s_sc = floored(average_slope(x, z, ch, big_l - ov), tb.S_MIN, "shallow flow")
        v["s_sc"] = s_sc
        if p.sc_type in tb.SC_TYPES:
            k = tb.SC_TYPES[p.sc_type][1]
            v["v_sc_ms"] = shallow_velocity(k, s_sc)
            v["tc_sc_mn"] = tc_shallow_minutes(sc_len, k, s_sc)
            total += v["tc_sc_mn"]
    if ch > 0:
        s_ch = floored(slope_10_85(x, z, 0.0, ch), tb.S_MIN_CHANNEL, "channel")
        v["s_ch"] = s_ch
        v["tc_ch_mn"] = tc_usbr_minutes(ch / 1000.0, s_ch)
        total += v["tc_ch_mn"]
    if ok:
        v["tc_comb_mn"] = total


def _select(c: CatchmentTc, s: Settings) -> None:
    p, v = c.params, c.values
    method = (p.tc_method or s.default_method or "").strip().lower()
    v["tc_sel_by"] = method or None
    if method not in tb.TC_METHODS:
        v["tc_sel_mn"] = None
        c.flags.append(f"tc_method '{method}' unknown: no selected Tc")
        return
    value = p.tc_user_mn if method == "user" else v.get(tb.TC_FIELDS[method])
    v["tc_sel_mn"] = value
    if value is None:
        c.flags.append(
            f"selected method {method} has no value"
            + (
                ": " + ", ".join(c.missing)
                if method == "combined" and c.missing
                else ""
            )
        )


def storm_guidance(values) -> tuple[float, float] | None:
    """(shortest, longest) Tc of a catchment over the methods, for the
    design storm and time-step guidance."""
    vals = [values.get(k) for k in ("tc_kirp_mn", "tc_usbr_mn", "tc_bw_mn",
                                    "tc_comb_mn", "tc_sel_mn")]  # fmt: skip
    vals = [x for x in vals if x]
    if not vals:
        return None
    return min(vals), max(vals)


def sample_cells(array, geotransform, nodata, x, y) -> np.ndarray:
    """Value of the cell containing each point (NaN outside or on NoData)."""
    a = np.asarray(array, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gt = geotransform
    c = np.floor((x - gt[0]) / gt[1]).astype(np.int64)
    r = np.floor((y - gt[3]) / gt[5]).astype(np.int64)
    out = np.full(x.shape, np.nan)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out[ok] = a[r[ok], c[ok]]
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        out[np.isclose(out, float(nodata), rtol=0.0, atol=1e-6)] = np.nan
    return out


def part_bounds(c: CatchmentTc) -> dict:
    """{part: (x0, x1)} along the profile for the current parameters (the
    same split as calculate), for sampling the land cover."""
    if c.x is None or len(c.x) < 2:
        return {}
    big_l = float(c.x[-1])
    p = c.params
    ch = p.ch_start_m if p.ch_start_m is not None else (c.ch_head_m or 0.0)
    ch = min(max(ch, 0.0), big_l)
    ov = min(max(p.ov_len_m or 0.0, 0.0), big_l - ch)
    return {"overland": (big_l - ov, big_l), "shallow": (ch, big_l - ov),
            "channel": (0.0, ch)}  # fmt: skip
