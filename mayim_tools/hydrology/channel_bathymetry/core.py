"""
DEM Channel Bathymetry - geometry core (NumPy / SciPy; no QGIS).

A DEM made by lidar or photogrammetry shows the water surface of a river
on the day of the survey, not its bed. This module cuts an estimated
channel into the DEM below that surface:

1. Stations every ``spacing`` metres along each centreline, each with a
   section at right angles out to ``search`` metres on each side.
2. The flat water surface: cells within ``flat_tol`` of the running median
   of the DEM under the centreline, with a slope below ``max_water_slope``,
   connected to the centreline (or a water mask given by the user).
3. Per station: the water width W and level, the bank tops (where the bank
   flattens below ``break_slope``) and the slope of each bank, fitted to the
   bank cells up to ``band`` metres above the water.
4. Shape: two half-parabolas, each meeting its bank at that bank's slope
   and meeting each other level at the deepest point. The left half-width
   is a = W sR / (sL + sR) and the depth D = sL sR W / (2 (sL + sR)); with
   equal slopes s this is one parabola, D = s W / 4 (the slope of a
   parabola at its edge is 4 D / W).
5. Depth caps: W / D >= ``min_ratio`` and D <= ``max_depth`` (optional).
   Low / high beds from the 25th / 75th percentiles of the bank slopes over
   a moving window of ``window_widths`` widths.
6. Stations with one bank slope use it for both sides and tilt the depth
   toward the outer bank of a bend by (1 - A kappa n) (scour factor A,
   Ikeda et al. 1981, Odgaard 1981); stations with none take the
   depth-to-width ratio of their neighbours, or ``fallback_ratios``.
7. Bed raster: for each water cell, the distances to the water's edge
   along the local section line give its position across the channel.

Offsets across a section are measured along the left-hand normal of the
centreline (drawn downstream), so positive = left bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.d8_network import GridError

TOOL_VERSION = "1.0.1"

BASIS_TEXT = {
    "banks": "both bank slopes",
    "one_bank": "one bank slope (bend tilt applied)",
    "neighbours": "depth-to-width ratio of neighbouring stations",
    "ratio": "fallback width-to-depth ratio",
    "none": "no water found at the station",
}
CAP_TEXT = {"": "-", "ratio": "width-to-depth ratio", "max": "maximum depth"}
SCENARIOS = ("low", "central", "high")


class ChannelError(GridError):
    """Raised when the inputs cannot give a channel."""


@dataclass
class Settings:
    spacing: float = 10.0  # station spacing (m)
    search: float = 50.0  # search half-width of each section (m)
    flat_tol: float = 0.10  # flat-water tolerance (m)
    max_water_slope: float = 0.05  # maximum slope of a water cell (m/m)
    break_slope: float = 0.08  # bank-top break slope (m/m)
    band: float = 0.5  # bank-slope band above the water (m)
    smooth_n: int = 5  # stations in the running medians
    window_widths: float = 20.0  # low/high percentile window, in widths
    min_ratio: float = 5.0  # minimum width-to-depth ratio
    max_depth: float | None = None  # optional maximum depth (m)
    fallback_ratios: tuple[float, float, float] | None = (
        None  # W/D shallow, central, deep
    )
    scour: float = 4.0  # bend scour factor A (0 = no tilt)
    thalweg_limit: float = 0.15  # deepest point kept within [lim, 1 - lim] of W
    min_width_cells: int = 2  # narrowest flat strip accepted as water


@dataclass
class Station:
    line: int
    chainage: float
    x: float
    y: float
    nx: float  # left-hand unit normal
    ny: float
    width: float = np.nan
    edge_left: float = np.nan  # offset of the left water edge (m, > 0)
    edge_right: float = np.nan  # offset of the right water edge (m, < 0)
    wse: float = np.nan
    slope_left: float = np.nan  # raw bank slopes (m/m)
    slope_right: float = np.nan
    top_left: float = np.nan  # bank-top offsets (m)
    top_right: float = np.nan
    height_left: float = np.nan  # bank-top height above the water (m)
    height_right: float = np.nan
    multi: bool = False  # more than one water strip on the section
    # filled after smoothing
    sl: float = np.nan
    sr: float = np.nan
    depth: dict = field(default_factory=dict)  # scenario -> D (m)
    depth_banks: float = np.nan  # uncapped bank-slope depth (m)
    split: float = 0.5  # left half-width / W
    kappa: float = 0.0  # centreline curvature (1/m, + = turning left)
    basis: str = "none"
    cap: str = ""
    flags: list = field(default_factory=list)


@dataclass
class ChannelResult:
    window_gt: tuple
    water: np.ndarray  # bool
    beds: dict  # scenario -> bed elevation (NaN outside the water)
    depth: dict  # scenario -> depth below the water surface (NaN outside)
    wse: np.ndarray  # water level used per water cell (NaN outside)
    stations: list
    lines: list  # densified centrelines [(n, 2)]
    warnings: list
    settings: Settings
    cell_area: float

    def summary(self) -> dict:
        valid = [s for s in self.stations if s.basis != "none"]
        out = {
            "stations": len(self.stations),
            "stations_with_water": len(valid),
            "length_m": float(sum(_length(line) for line in self.lines)),
            "water_area_m2": float(self.water.sum() * self.cell_area),
        }
        for sc in SCENARIOS:
            d = self.depth[sc]
            ok = np.isfinite(d)
            out[f"volume_{sc}_m3"] = float(np.nansum(d) * self.cell_area)
            out[f"max_depth_{sc}_m"] = float(d[ok].max()) if ok.any() else np.nan
            ds = [s.depth.get(sc, np.nan) for s in valid]
            out[f"mean_station_depth_{sc}_m"] = (
                float(np.nanmean(ds)) if np.isfinite(ds).any() else np.nan
            )
        for key in BASIS_TEXT:
            out[f"basis_{key}"] = sum(s.basis == key for s in self.stations)
        out["capped_ratio"] = sum(s.cap == "ratio" for s in self.stations)
        out["capped_max"] = sum(s.cap == "max" for s in self.stations)
        widths = [s.width for s in valid if np.isfinite(s.width)]
        out["width_min_m"] = float(min(widths)) if widths else np.nan
        out["width_max_m"] = float(max(widths)) if widths else np.nan
        return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _length(line) -> float:
    p = np.asarray(line, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def densify(line, step: float):
    """Points every ``step`` metres along a polyline: (xy, chainage, unit
    tangents). Duplicate vertices are dropped."""
    p = np.asarray(line, float).reshape(-1, 2)
    keep = np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 1e-9]
    p = p[keep]
    if len(p) < 2:
        raise ChannelError("A centreline needs at least two separate vertices.")
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    s = np.arange(0.0, cum[-1], step)
    s = np.r_[s, cum[-1]] if cum[-1] - s[-1] > 1e-6 else s
    xy = np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])
    d = np.gradient(xy, axis=0)
    t = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)[:, None]
    return xy, s, t


def running_nanmedian(values, n: int) -> np.ndarray:
    """Centred running median over ``n`` values, ignoring NaN (NaN only
    where the whole window is NaN)."""
    v = np.asarray(values, float)
    if n <= 1 or len(v) < 2:
        return v.copy()
    h = n // 2
    padded = np.r_[np.full(h, np.nan), v, np.full(h, np.nan)]
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * h + 1)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(win).any(axis=1)
    if ok.any():
        with np.errstate(all="ignore"):
            out[ok] = np.nanmedian(win[ok], axis=1)
    return out


def half_parabola_split(sl: float, sr: float) -> tuple[float, float]:
    """(depth / W, left half-width / W) of two half-parabolas meeting the
    banks at slopes ``sl`` and ``sr``."""
    return sl * sr / (2.0 * (sl + sr)), sr / (sl + sr)


def bank_slope_depth(sl: float, sr: float, width: float) -> float:
    """D = sL sR W / (2 (sL + sR)); s W / 4 when one slope is NaN."""
    fl, fr = np.isfinite(sl) and sl > 0, np.isfinite(sr) and sr > 0
    if fl and fr:
        return half_parabola_split(sl, sr)[0] * width
    if fl or fr:
        return (sl if fl else sr) * width / 4.0
    return np.nan


def _cell_size(gt) -> float:
    return float(0.5 * (abs(gt[1]) + abs(gt[5])))


def _to_rc(xy, gt):
    cols = np.floor((xy[..., 0] - gt[0]) / gt[1]).astype(np.int64)
    rows = np.floor((xy[..., 1] - gt[3]) / gt[5]).astype(np.int64)
    return rows, cols


def _sample(arr, rows, cols, fill):
    nr, nc = arr.shape
    ok = (rows >= 0) & (rows < nr) & (cols >= 0) & (cols < nc)
    out = np.full(rows.shape, fill, dtype=arr.dtype)
    out[ok] = arr[rows[ok], cols[ok]]
    return out, ok


def flat_slope(dem: np.ndarray, gt) -> np.ndarray:
    """Slope from the smaller of the forward and backward differences in
    each direction, so a cell at the edge of a flat (one flat side) still
    reads as flat. NaN where a neighbour is missing on both sides."""
    out = []
    for axis, size in ((1, abs(gt[1])), (0, abs(gt[5]))):
        d = np.abs(np.diff(dem, axis=axis)) / size
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        back = np.pad(d, pad, constant_values=np.nan)
        pad[axis] = (0, 1)
        fwd = np.pad(d, pad, constant_values=np.nan)
        out.append(np.fmin(back, fwd))
    return np.hypot(out[0], out[1])


# ---------------------------------------------------------------------------
# 1-2. Water mask
# ---------------------------------------------------------------------------


def water_mask(dem: np.ndarray, gt, lines, s: Settings, user_mask=None):
    """(water mask, list of warnings). ``lines`` are densified centrelines
    [(xy, chainage, tangent)]."""
    from scipy import ndimage

    valid = np.isfinite(dem)
    warnings = []
    if user_mask is not None:
        water = np.asarray(user_mask, bool) & valid
        if not water.any():
            raise ChannelError("The water polygon does not cover any DEM cell.")
        return water, warnings
    cell = _cell_size(gt)
    # cells near a centreline and their reference level
    near = np.zeros(dem.shape, bool)
    zref = np.full(dem.shape, np.nan)
    best = np.full(dem.shape, np.inf)
    seed = np.zeros(dem.shape, bool)
    for i, (xy, ch, _) in enumerate(lines):
        r, c = _to_rc(xy, gt)
        z, ok = _sample(dem, r, c, np.nan)
        if not np.isfinite(z).any():
            warnings.append(f"Centreline {i + 1} lies outside the DEM.")
            continue
        k = max(5, int(round(2 * s.spacing / (cell / 2))) | 1)
        prof = running_nanmedian(z, k)
        prof = np.fmin.accumulate(np.where(np.isfinite(prof), prof, np.inf))
        prof[~np.isfinite(prof)] = np.nan
        line_mask = np.zeros(dem.shape, bool)
        line_mask[r[ok], c[ok]] = True
        dist, (ir, ic) = ndimage.distance_transform_edt(
            ~line_mask, sampling=(abs(gt[5]), abs(gt[1])), return_indices=True
        )
        inside = dist <= s.search
        # chainage of the nearest centreline cell: look it up by cell index
        lut = np.full(dem.shape, np.nan)
        lut[r[ok], c[ok]] = ch[ok]
        ch_cell = lut[ir, ic]
        z_here = np.interp(ch_cell, ch[np.isfinite(prof)], prof[np.isfinite(prof)])
        take = inside & (dist < best)  # the nearest centreline sets the level
        zref = np.where(take, z_here, zref)
        best = np.where(take, dist, best)
        near |= inside
        seed[r[ok], c[ok]] = True
    if not near.any():
        raise ChannelError("No centreline overlaps the DEM.")
    slope = flat_slope(dem, gt)
    cand = (
        near
        & valid
        & (np.abs(dem - zref) <= s.flat_tol)
        & (np.nan_to_num(slope, nan=np.inf) <= s.max_water_slope)
    )
    labels, n = ndimage.label(cand, structure=np.ones((3, 3), bool))
    keep = np.unique(labels[seed & cand])
    keep = keep[keep > 0]
    water = np.isin(labels, keep)
    if not water.any():
        raise ChannelError(
            "No flat water surface was found along the centreline(s). Check "
            "that the centreline follows the water, raise the flat-water "
            "tolerance, or give a water polygon (needed for photogrammetry "
            "DEMs, whose water surface is noisy)."
        )
    return water, warnings


# ---------------------------------------------------------------------------
# 3. Stations
# ---------------------------------------------------------------------------


def _bank(offsets, z, rc, edge_index, direction, edge_offset, wse, s, cell):
    """(slope, bank-top offset, bank-top height, edge shift outward) of one
    bank, walking out
    from the last water sample ``edge_index`` in ``direction`` (+1 / -1)."""
    n = len(offsets)
    idx = []
    j = edge_index + direction
    top = None
    while 0 <= j < n and np.isfinite(z[j]):
        dz = z[j] - wse
        idx.append(j)
        k = j + 4 * direction  # two cells further out (samples are half cells)
        if dz > s.flat_tol and 0 <= k < n and np.isfinite(z[k]):
            if (z[k] - z[j]) / (2 * cell) < s.break_slope:
                top = j
                break
        j += direction
    if not idx:
        return np.nan, np.nan, np.nan, 0.0
    if top is None:
        top = idx[-1]
    idx = idx[: idx.index(top) + 1]
    dx = np.abs(offsets[idx] - edge_offset)
    dz = z[idx] - wse
    cells = [rc[i] for i in idx]
    above = [i for i, h in enumerate(dz) if h > 0]
    band = [i for i in above if dz[i] <= s.band]
    if len({cells[i] for i in band}) < 2:  # steep bank: the first two cells up
        band, seen = [], set()
        for i in above:
            band.append(i)
            seen.add(cells[i])
            if len(seen) >= 2:
                break
        if len(seen) < 2:
            return np.nan, offsets[top], z[top] - wse, 0.0
    bx, bz = dx[band], dz[band]
    shift = 0.0
    slope = np.nan
    if len(set(bx.tolist())) >= 2:
        # free line: where it crosses the water level is the water's edge,
        # to better than the cell size the water mask gives
        fit_s, fit_c = np.polyfit(bx, bz, 1)
        if fit_s > 0 and abs(-fit_c / fit_s) <= cell:
            slope, shift = float(fit_s), float(-fit_c / fit_s)
    if not np.isfinite(slope):  # a line through the mask edge
        slope = float(np.sum(bx * bz) / np.sum(bx * bx))
    if slope <= 0:
        return np.nan, offsets[top], z[top] - wse, 0.0
    return slope, offsets[top], z[top] - wse, shift


def measure_stations(dem, gt, water, lines, s: Settings, labels=None):
    """[Station] for every centreline (water width, level, banks)."""
    from scipy import ndimage

    cell = _cell_size(gt)
    step = cell / 2.0
    offsets = np.arange(-s.search, s.search + step / 2, step)
    centre = int(np.argmin(np.abs(offsets)))
    if labels is None:
        labels, _ = ndimage.label(water, structure=np.ones((3, 3), bool))
    stations = []
    for li, (xy, ch, t) in enumerate(lines):
        st = np.arange(0.0, ch[-1] + 1e-9, s.spacing)
        if ch[-1] - st[-1] > 0.25 * s.spacing:
            st = np.r_[st, ch[-1]]
        px = np.interp(st, ch, xy[:, 0])
        py = np.interp(st, ch, xy[:, 1])
        # tangent smoothed over about one station spacing
        k = max(1, int(round(s.spacing / step)))
        tx = np.convolve(t[:, 0], np.ones(k) / k, mode="same")
        ty = np.convolve(t[:, 1], np.ones(k) / k, mode="same")
        tx, ty = np.interp(st, ch, tx), np.interp(st, ch, ty)
        norm = np.maximum(np.hypot(tx, ty), 1e-12)
        nx, ny = -ty / norm, tx / norm
        for i in range(len(st)):
            sta = Station(li, float(st[i]), float(px[i]), float(py[i]),
                          float(nx[i]), float(ny[i]))  # fmt: skip
            stations.append(sta)
            pts = np.column_stack([px[i] + offsets * nx[i], py[i] + offsets * ny[i]])
            r, c = _to_rc(pts, gt)
            w, _ = _sample(water, r, c, False)
            z, _ = _sample(dem, r, c, np.nan)
            lab, _ = _sample(labels, r, c, 0)
            if not w.any():
                sta.flags.append("no water")
                continue
            wi = np.flatnonzero(w)
            near = wi[np.argmin(np.abs(wi - centre))]
            if abs(offsets[near]) > s.search / 2:
                sta.flags.append("no water near the centreline")
                continue
            lo = near
            while lo > 0 and w[lo - 1]:
                lo -= 1
            hi = near
            while hi < len(w) - 1 and w[hi + 1]:
                hi += 1
            if (hi - lo + 1) * step < s.min_width_cells * cell - 1e-9:
                sta.flags.append("narrow")
            main = lab[near]
            other = w & (lab == main)
            other[lo : hi + 1] = False
            sta.multi = bool(other.any())  # e.g. both sides of an island
            if sta.multi:
                sta.flags.append("more than one water strip")
            sta.edge_left = float(offsets[hi] + step / 2)
            sta.edge_right = float(offsets[lo] - step / 2)
            sta.width = sta.edge_left - sta.edge_right
            sta.wse = float(np.nanmedian(z[lo : hi + 1]))
            rc = list(zip(r.tolist(), c.tolist(), strict=True))
            if hi >= len(w) - 1 or lo <= 0:
                sta.flags.append("water reaches the search limit")
            sl, tl, hl, el = _bank(
                offsets, z, rc, hi, +1, sta.edge_left, sta.wse, s, cell
            )
            sr, tr, hr, er = _bank(
                offsets, z, rc, lo, -1, sta.edge_right, sta.wse, s, cell
            )
            sta.edge_left += el
            sta.edge_right -= er
            sta.width = sta.edge_left - sta.edge_right
            sta.slope_left, sta.top_left, sta.height_left = sl, tl, hl
            sta.slope_right, sta.top_right, sta.height_right = sr, tr, hr
    return stations


# ---------------------------------------------------------------------------
# 4-6. Depths
# ---------------------------------------------------------------------------


def _curvature(st_chainage, lines_entry, smooth: float) -> np.ndarray:
    xy, ch, t = lines_entry
    theta = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    step = ch[1] - ch[0] if len(ch) > 1 else 1.0
    k = max(1, int(round(smooth / step)))
    if k > 1 and len(theta) > k:
        theta = np.convolve(
            np.pad(theta, k // 2, mode="edge"), np.ones(k) / k, mode="valid"
        )[: len(ch)]
    kappa = np.gradient(theta, ch) if len(ch) > 1 else np.zeros(len(ch))
    return np.interp(st_chainage, ch, kappa)


def assign_depths(stations, lines, s: Settings) -> list[str]:
    """Smooths widths, levels and slopes per centreline and sets each
    station's low / central / high depth, thalweg split, basis and cap.
    Returns warnings."""
    warnings = []
    for li, entry in enumerate(lines):
        group = [st for st in stations if st.line == li]
        if not group:
            continue
        ch = np.array([st.chainage for st in group])
        has = np.array([st.width > 0 for st in group])
        if not has.any():
            warnings.append(f"Centreline {li + 1}: no water found at any station.")
            continue
        W = running_nanmedian([st.width for st in group], s.smooth_n)
        # widths far outside the reach's spread (Tukey fences) are flagged
        q1, q3 = np.nanpercentile(W[has], [25, 75])
        fence = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        wse = running_nanmedian([st.wse for st in group], s.smooth_n)
        wse = np.fmin.accumulate(np.where(np.isfinite(wse), wse, np.inf))
        sl_raw = np.array([st.slope_left for st in group])
        sr_raw = np.array([st.slope_right for st in group])
        sl = running_nanmedian(sl_raw, s.smooth_n)
        sr = running_nanmedian(sr_raw, s.smooth_n)
        half = 0.5 * s.window_widths * float(np.nanmedian(W[has]))
        kappa = _curvature(ch, entry, max(2 * float(np.nanmedian(W[has])), s.spacing))
        for i, st in enumerate(group):
            if not has[i]:
                st.basis = "none"
                continue
            st.width = float(W[i]) if np.isfinite(W[i]) else st.width
            if not fence[0] <= st.width <= fence[1]:
                st.flags.append("width outlier")
            st.wse = float(wse[i]) if np.isfinite(wse[i]) else st.wse
            st.sl, st.sr, st.kappa = float(sl[i]), float(sr[i]), float(kappa[i])
            win = np.abs(ch - ch[i]) <= half
            pct = {}
            for side, raw, smooth in (("l", sl_raw, sl[i]), ("r", sr_raw, sr[i])):
                vals = raw[win & np.isfinite(raw)]
                if np.isfinite(smooth) and vals.size:
                    lo_, hi_ = np.percentile(vals, [25, 75])
                    pct[side] = (min(lo_, smooth), smooth, max(hi_, smooth))
                else:
                    pct[side] = (np.nan, np.nan, np.nan)
            n_bank = sum(np.isfinite(v) for v in (st.sl, st.sr))
            if n_bank == 0:
                continue  # filled from neighbours below
            st.basis = "banks" if n_bank == 2 else "one_bank"
            if n_bank == 1:
                st.flags.append("one bank slope")
            for k, sc in enumerate(SCENARIOS):
                st.depth[sc] = bank_slope_depth(pct["l"][k], pct["r"][k], st.width)
            st.depth_banks = st.depth["central"]
            if n_bank == 2 and not st.multi:
                st.split = half_parabola_split(st.sl, st.sr)[1]
            st.split = float(np.clip(st.split, s.thalweg_limit, 1 - s.thalweg_limit))
        # stations with no bank slope: depth-to-width ratio of neighbours
        ok = [i for i, st in enumerate(group) if st.depth]
        missing = [i for i, st in enumerate(group) if has[i] and not st.depth]
        for i in missing:
            st = group[i]
            if ok:
                for sc in SCENARIOS:
                    ratio = np.interp(
                        ch[i],
                        ch[ok],
                        [group[j].depth[sc] / group[j].width for j in ok],
                    )
                    st.depth[sc] = float(ratio * st.width)
                st.basis = "neighbours"
                st.flags.append("no bank slope")
            elif s.fallback_ratios:
                shallow, central, deep = s.fallback_ratios
                st.depth = {
                    "low": st.width / shallow,
                    "central": st.width / central,
                    "high": st.width / deep,
                }
                st.basis = "ratio"
                st.flags.append("no bank slope")
        if missing and not ok and not s.fallback_ratios:
            warnings.append(
                f"Centreline {li + 1}: no bank slope could be measured at any "
                "station and no fallback width-to-depth ratio was given, so "
                "no bed was cut along it."
            )
        # caps
        for st in group:
            if not st.depth:
                continue
            cap = st.width / s.min_ratio if s.min_ratio > 0 else np.inf
            which = "ratio"
            if s.max_depth is not None and s.max_depth > 0 and s.max_depth < cap:
                cap, which = s.max_depth, "max"
            if st.depth["central"] > cap:
                st.cap = which
                st.flags.append(f"capped ({CAP_TEXT[which]})")
            d = {sc: float(min(st.depth[sc], cap)) for sc in SCENARIOS}
            d["low"] = min(d["low"], d["central"])
            d["high"] = max(d["high"], d["central"])
            st.depth = d
    return warnings


# ---------------------------------------------------------------------------
# 7. Bed raster
# ---------------------------------------------------------------------------


def _edge_distances(water, gt, xy, normals, max_dist):
    """Distance from each cell centre ``xy`` to the water's edge along
    +normal and -normal (vectorised ray march in half-cell steps)."""
    cell = _cell_size(gt)
    step = cell / 2.0
    n_steps = int(np.ceil(max_dist / step)) + 1
    out = []
    for sign in (1.0, -1.0):
        dist = np.full(len(xy), n_steps * step)
        active = np.arange(len(xy))
        for k in range(1, n_steps + 1):
            if active.size == 0:
                break
            p = xy[active] + sign * k * step * normals[active]
            r, c = _to_rc(p, gt)
            w, _ = _sample(water, r, c, False)
            stop = ~w
            dist[active[stop]] = (k - 0.5) * step
            active = active[~stop]
        out.append(dist)
    return out[0], out[1]


def build_bed(dem, gt, water, stations, lines, s: Settings):
    """(beds, depths, wse) per scenario: arrays over the window, NaN where
    no bed was cut."""
    from scipy.spatial import cKDTree

    shape = dem.shape
    beds = {sc: np.full(shape, np.nan) for sc in SCENARIOS}
    wse_grid = np.full(shape, np.nan)
    rows, cols = np.nonzero(water)
    if rows.size == 0:
        return beds, {sc: beds[sc].copy() for sc in SCENARIOS}, wse_grid
    xc = gt[0] + (cols + 0.5) * gt[1]
    yc = gt[3] + (rows + 0.5) * gt[5]
    cxy = np.column_stack([xc, yc])
    # nearest centreline point of every water cell
    pts = np.vstack([e[0] for e in lines])
    owner = np.concatenate([np.full(len(e[0]), i) for i, e in enumerate(lines)])
    chain = np.concatenate([e[1] for e in lines])
    tang = np.vstack([e[2] for e in lines])
    _, idx = cKDTree(pts).query(cxy)
    line_id, ch_cell, t = owner[idx], chain[idx], tang[idx]
    normals = np.column_stack([-t[:, 1], t[:, 0]])
    max_w = max((st.width for st in stations if st.width > 0), default=s.search)
    dl, dr = _edge_distances(water, gt, cxy, normals, min(2 * s.search, 2 * max_w))
    w_cell = dl + dr
    for li in range(len(lines)):
        group = [st for st in stations if st.line == li and st.depth]
        sel = line_id == li
        if not group or not sel.any():
            continue
        ch = np.array([st.chainage for st in group])

        def at(values, ch=ch, sel=sel):
            return np.interp(ch_cell[sel], ch, values)

        W = at([st.width for st in group])
        level = at([st.wse for st in group])
        split = at([st.split for st in group])
        tilt_on = at([0.0 if st.basis == "banks" else 1.0 for st in group]) > 0.5
        kappa = at([st.kappa for st in group])
        wc = w_cell[sel]
        x = dl[sel]  # distance from the left edge
        a = split * wc
        u = np.where(x <= a, (a - x) / np.maximum(a, 1e-9),
                     (x - a) / np.maximum(wc - a, 1e-9))  # fmt: skip
        shape_f = np.clip(1.0 - u**2, 0.0, 1.0)
        scale = np.clip(wc / np.maximum(W, 1e-9), 0.0, 1.0)
        n_off = 0.5 * (dr[sel] - dl[sel])  # offset from the local centre (+ left)
        c = s.scour * kappa
        c = np.clip(c, -0.8 / np.maximum(wc / 2, 1e-9), 0.8 / np.maximum(wc / 2, 1e-9))
        tilt = np.where(tilt_on, 1.0 - c * n_off, 1.0)
        r_, c_ = rows[sel], cols[sel]
        wse_grid[r_, c_] = level
        for sc in SCENARIOS:
            D = at([st.depth[sc] for st in group])
            depth = np.maximum(D * scale * shape_f * tilt, 0.0)
            beds[sc][r_, c_] = np.minimum(dem[r_, c_], level - depth)
    depths = {sc: wse_grid - beds[sc] for sc in SCENARIOS}
    for sc in SCENARIOS:
        depths[sc] = np.where(
            np.isfinite(depths[sc]), np.maximum(depths[sc], 0), np.nan
        )
    return beds, depths, wse_grid


def rises_downstream(dem, gt, xy, tol: float = 0.05) -> bool:
    """True when the DEM under the first fifth of a line is clearly lower
    than under the last fifth (the line was drawn upstream)."""
    r, c = _to_rc(xy, gt)
    z, _ = _sample(dem, r, c, np.nan)
    n = max(1, len(z) // 5)
    head, tail = z[:n], z[-n:]
    if not (np.isfinite(head).any() and np.isfinite(tail).any()):
        return False
    return bool(np.nanmedian(tail) - np.nanmedian(head) > tol)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_channel_bathymetry(
    dem: np.ndarray,
    nodata,
    gt,
    centrelines,
    settings: Settings | None = None,
    user_mask=None,
    is_canceled=lambda: False,
) -> ChannelResult:
    """DEM window, its geotransform and centrelines [(n, 2) map coords,
    drawn downstream] -> ChannelResult."""
    s = settings or Settings()
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise ChannelError("Rotated rasters are not supported.")
    if s.spacing <= 0 or s.search <= 0:
        raise ChannelError("The station spacing and search width must be > 0.")
    z = np.asarray(dem, float)
    if nodata is not None and np.isfinite(nodata):
        z = np.where(z == nodata, np.nan, z)
    cell = _cell_size(gt)
    if not centrelines:
        raise ChannelError("No centreline was given.")
    warnings = []
    lines = []
    for i, line in enumerate(centrelines):
        entry = densify(line, cell / 2.0)
        if rises_downstream(z, gt, entry[0]):
            entry = densify(np.asarray(line, float)[::-1], cell / 2.0)
            warnings.append(
                f"Centreline {i + 1} was drawn upstream (the water surface rises "
                "along it); it was reversed."
            )
        lines.append(entry)
    water, mask_warnings = water_mask(z, gt, lines, s, user_mask)
    warnings += mask_warnings
    if is_canceled():
        raise ChannelError("Cancelled.")
    stations = measure_stations(z, gt, water, lines, s)
    warnings += assign_depths(stations, lines, s)
    if not any(st.depth for st in stations):
        raise ChannelError(
            "No bed could be estimated: no bank slope was measured. Check the "
            "water surface and banks, or give a fallback width-to-depth ratio."
        )
    if is_canceled():
        raise ChannelError("Cancelled.")
    # water not reached by any station's depth stays as surveyed
    beds, depths, wse = build_bed(z, gt, water, stations, lines, s)
    return ChannelResult(
        window_gt=tuple(gt),
        water=water,
        beds=beds,
        depth=depths,
        wse=wse,
        stations=stations,
        lines=[e[0] for e in lines],
        warnings=warnings,
        settings=s,
        cell_area=abs(gt[1] * gt[5]),
    )
