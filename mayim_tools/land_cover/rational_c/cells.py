"""
Per-cell classes for the Run-off Coefficients (SANRAL Deterministic) (pure NumPy, no
GDAL, no QGIS): land-cover role, slope class, permeability class (with the
P05 / P95 classes from the soil-group probabilities), vegetation class
(design, lower, upper) and the urban C2 (design, lower, upper end).

Permeability is held as a half-class index ``perm2`` (0-6): 0 very
permeable, 2 permeable, 4 semi-permeable, 6 impermeable, and the odd values
for the SCS-SA intermediate groups A/B, B/C and C/D, which take the average
of the neighbouring Cp values (half of the cell's area to each class).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lookup as lk
from . import tables as tb

NONE = -1
# soil-group code (Hydrological Soil Groups (HSG) tool) -> perm2
HSG_PERM2 = {
    1: 0,  # A  -> very permeable
    2: 2,  # B  -> permeable
    3: 4,  # C  -> semi-permeable
    4: 6,  # D  -> impermeable
    11: 6,  # A/D, B/D, C/D: undrained -> D
    12: 6,
    13: 6,
    21: 1,  # SCS-SA A/B
    22: 3,  # SCS-SA B/C
    23: 5,  # SCS-SA C/D
}
PERM2_LABELS = {
    0: "Very permeable",
    1: "Very permeable / permeable",
    2: "Permeable",
    3: "Permeable / semi-permeable",
    4: "Semi-permeable",
    5: "Semi-permeable / impermeable",
    6: "Impermeable",
}
PROB_LOW, PROB_HIGH = 5.0, 95.0  # quantiles (%) of the soil-group probabilities


@dataclass
class CellOptions:
    band: int  # MAP band index (for the per-cell C1 raster)
    position: str = "upper"  # urban C2 within the Table 3.7 range
    wetlands_impermeable: bool = False


def slope_class(slope_pct: np.ndarray) -> np.ndarray:
    """0-3 (Table 3.7 slope classes); a value on a limit goes to the class
    above; -1 where unknown."""
    s = np.asarray(slope_pct, dtype=float)
    out = np.digitize(np.where(np.isfinite(s), s, 0.0), tb.SLOPE_LIMITS_PCT).astype(
        np.int8
    )
    out[~np.isfinite(s)] = NONE
    return out


def perm2_from_hsg(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=float)
    ok = np.isfinite(c)
    ci = np.where(ok, c, 0).astype(np.int64)
    out = np.full(c.shape, NONE, dtype=np.int8)
    for code, p in HSG_PERM2.items():
        out[ok & (ci == code)] = p
    return out


def perm2_quantiles(prob: np.ndarray, design: np.ndarray):
    """perm2 of the soil-group quantiles P05 and P95 from the probabilities
    of groups A-D (``prob`` shape (4, rows, cols), %). The P05 group is the
    first group (from A) at which the cumulative probability reaches 5 %
    (the most permeable plausible group), the P95 group the first at which
    it reaches 95 %. Cells without probabilities keep the design class."""
    p = np.asarray(prob, dtype=float)
    tot = np.nansum(p, axis=0)
    ok = np.isfinite(p).all(axis=0) & (tot > 0)
    cum = np.cumsum(np.where(np.isfinite(p), p, 0.0), axis=0) * (
        100.0 / np.where(tot > 0, tot, 1.0)
    )
    g05 = np.argmax(cum >= PROB_LOW - 1e-9, axis=0)
    g95 = np.argmax(cum >= PROB_HIGH - 1e-9, axis=0)
    lo = np.where(ok, 2 * g05, design).astype(np.int8)
    hi = np.where(ok, 2 * g95, design).astype(np.int8)
    # never let the range exclude the design class
    known = design >= 0
    lo = np.where(known, np.minimum(lo, design), lo).astype(np.int8)
    hi = np.where(known, np.maximum(hi, design), hi).astype(np.int8)
    return lo, hi


def cp_value(perm2: np.ndarray, band: int) -> np.ndarray:
    col = np.array([r[band] for r in tb.CP], dtype=float)
    half = np.array(
        [col[0], 0.5 * (col[0] + col[1]), col[1], 0.5 * (col[1] + col[2]), col[2]]
        + [0.5 * (col[2] + col[3]), col[3]]
    )
    p = np.asarray(perm2)
    return np.where(p >= 0, half[np.clip(p, 0, 6)], np.nan)


def table_value(cls: np.ndarray, table, band: int) -> np.ndarray:
    col = np.array([r[band] for r in table], dtype=float)
    c = np.asarray(cls)
    return np.where(c >= 0, col[np.clip(c, 0, 3)], np.nan)


def _lawn(slope_pct, perm2, which: str):
    """Lawn C2 interpolated between the flat (< 2 %) and steep (> 7 %) rows,
    sandy for soil groups A and B (A/B), heavy soil otherwise (unknown soil
    takes heavy soil). ``which``: 'lower', 'middle' or 'upper'."""
    sandy = (perm2 >= 0) & (perm2 <= 2)
    vals = {}
    for soil in ("sandy", "heavy"):
        flat = tb.urban_value(f"lawn_{soil}_flat", which)
        steep = tb.urban_value(f"lawn_{soil}_steep", which)
        s = np.where(np.isfinite(slope_pct), slope_pct, tb.LAWN_FLAT_PCT)
        t = np.clip(
            (s - tb.LAWN_FLAT_PCT) / (tb.LAWN_STEEP_PCT - tb.LAWN_FLAT_PCT), 0, 1
        )
        vals[soil] = flat + t * (steep - flat)
    return np.where(sandy, vals["sandy"], vals["heavy"])


def urban_c2(row_idx: np.ndarray, slope_pct, perm2, position: str):
    """C2 (design at ``position``, lower end, upper end) for urban cells.
    ``row_idx`` indexes URBAN_ORDER (-1 where not urban)."""
    out = []
    for which in (position, "lower", "upper"):
        vals = np.array([tb.urban_value(k, which) for k in tb.URBAN_KEYS] + [np.nan])
        v = np.where(
            row_idx >= 0, vals[np.clip(row_idx, 0, len(tb.URBAN_KEYS))], np.nan
        )
        lawn = row_idx == LAWN_IDX
        if lawn.any():
            v = np.where(lawn, _lawn(slope_pct, perm2, which), v)
        out.append(v)
    return tuple(out)


LAWN_IDX = len(tb.URBAN_KEYS)  # row index used for lawns
URBAN_INDEX = {k: i for i, k in enumerate(tb.URBAN_KEYS)} | {lk.LAWN: LAWN_IDX}


def lookup_arrays(lookup: lk.Lookup, max_code: int = 255):
    """Arrays indexed by land-cover code: role (0 unknown, 1-4), cv, cv_low,
    cv_high (-1 none) and urban row index (-1 none)."""
    n = max(max_code, max(lookup.rules, default=0)) + 1
    role = np.zeros(n, dtype=np.int8)
    cv = np.full((3, n), NONE, dtype=np.int8)
    urb = np.full(n, NONE, dtype=np.int16)
    for code, r in lookup.rules.items():
        if code < 0 or code >= n:
            continue
        role[code] = lk.ROLE_CODES[r.role]
        if r.role == lk.RURAL:
            cv[0, code] = r.cv
            cv[1, code] = r.cv_low if r.cv_low is not None else r.cv
            cv[2, code] = r.cv_high if r.cv_high is not None else r.cv
        if r.role == lk.URBAN and r.urban:
            urb[code] = URBAN_INDEX[r.urban]
    return role, cv, urb


def classify(
    lc,
    slope_pct,
    hsg,
    lookup: lk.Lookup,
    opts: CellOptions,
    prob=None,
    texture=None,
    override=None,
) -> dict:
    """All per-cell classes and values. ``lc`` land-cover codes (NaN or
    any code missing from the lookup -> no role), ``slope_pct`` slope (%),
    ``hsg`` soil-group codes, ``prob`` optional (4, r, c) group
    probabilities (%), ``texture`` optional USDA texture codes 1-12,
    ``override`` optional dict of arrays (see ``apply_overrides``)."""
    lcf = np.asarray(lc, dtype=float)
    # code 0 is no data (outside South Africa in SANLC) unless the lookup has it
    lc_ok = np.isfinite(lcf) & (lcf >= 0)
    if 0 not in lookup.rules:
        lc_ok &= lcf != 0
    lci = np.where(lc_ok, lcf, 0).astype(np.int64)
    role_a, cv_a, urb_a = lookup_arrays(lookup, int(lci.max(initial=0)))
    role = np.where(lc_ok, role_a[lci], 0).astype(np.int8)
    unknown = lc_ok & (role == 0)

    slope = np.asarray(slope_pct, dtype=float)
    scls = slope_class(slope)
    perm = perm2_from_hsg(hsg)
    if prob is not None:
        p_lo, p_hi = perm2_quantiles(prob, perm)
    else:
        p_lo, p_hi = perm.copy(), perm.copy()
    forced = (
        lc_ok & (lci == lk.SANLC_ROCK)
        if lookup.key == "sanlc"
        else np.zeros(role.shape, bool)
    )
    if opts.wetlands_impermeable and lookup.key == "sanlc":
        forced |= lc_ok & np.isin(lci, lk.SANLC_WETLANDS)
    for a in (perm, p_lo, p_hi):
        a[forced] = 6

    veg = [np.where(role == 1, cv_a[i][lci], NONE).astype(np.int8) for i in range(3)]
    urb = np.where(role == 2, urb_a[lci], NONE).astype(np.int16)
    c2 = c2_lo = c2_hi = None

    if override:
        role, veg, urb, c2_fixed = apply_overrides(role, veg, urb, override)
    else:
        c2_fixed = None
    c2, c2_lo, c2_hi = urban_c2(urb, slope, perm, opts.position)
    if c2_fixed is not None:
        m = np.isfinite(c2_fixed) & (role == 2)
        for a in (c2, c2_lo, c2_hi):
            a[m] = c2_fixed[m]
    urban = role == 2
    for a in (c2, c2_lo, c2_hi):
        a[~urban] = np.nan

    tex = None
    if texture is not None:
        t = np.asarray(texture, dtype=float)
        tex = np.full(t.shape, NONE, dtype=np.int8)
        ok = np.isfinite(t)
        ti = np.where(ok, t, 0).astype(np.int64)
        for code, p in lk.TEXTURE_TO_PERMEABILITY.items():
            tex[ok & (ti == code)] = p

    rural = role == 1
    b = opts.band
    c1 = table_value(scls, tb.CS, b) + cp_value(perm, b) + table_value(veg[0], tb.CV, b)
    c1[~rural] = np.nan
    return {
        "lc": np.where(lc_ok, lci, -1),
        "role": role,
        "unknown": unknown,
        "slope": slope,
        "scls": scls,
        "perm": perm,
        "perm_lo": p_lo,
        "perm_hi": p_hi,
        "forced_impermeable": forced,
        "veg": veg[0],
        "veg_lo": veg[1],
        "veg_hi": veg[2],
        "urb": urb,
        "c2": c2,
        "c2_lo": c2_lo,
        "c2_hi": c2_hi,
        "c1": c1,
        "tex": tex,
    }


def apply_overrides(role, veg, urb, ov: dict):
    """Site overrides (already rasterised): ``ov['cv']`` vegetation class
    (-1 none) forces the cell rural with that class; ``ov['urban']`` urban
    row index (-1 none) forces it urban; ``ov['c2']`` (NaN none) fixes C2 of
    an urban cell. A land-cover code override is applied to ``lc`` before
    ``classify`` is called."""
    role = role.copy()
    veg = [v.copy() for v in veg]
    urb = urb.copy()
    cvo = ov.get("cv")
    if cvo is not None:
        m = cvo >= 0
        role[m] = 1
        for v in veg:
            v[m] = cvo[m]
        urb[m] = NONE
    uo = ov.get("urban")
    if uo is not None:
        m = uo >= 0
        role[m] = 2
        urb[m] = uo[m]
        for v in veg:
            v[m] = NONE
    c2 = ov.get("c2")
    return role, veg, urb, c2


def perm_shares(perm2: np.ndarray) -> np.ndarray:
    """Shares (fractions) of the four permeability classes for the cells
    given (odd half-classes split half to each neighbour); zeros if none."""
    p = np.asarray(perm2)
    p = p[p >= 0]
    if p.size == 0:
        return np.zeros(4)
    cnt = np.bincount(p.astype(np.int64), minlength=7)[:7].astype(float)
    sh = np.zeros(4)
    for k in range(7):
        if k % 2 == 0:
            sh[k // 2] += cnt[k]
        else:
            sh[k // 2] += 0.5 * cnt[k]
            sh[k // 2 + 1] += 0.5 * cnt[k]
    return sh / p.size


def class_shares(cls: np.ndarray, n: int = 4) -> np.ndarray:
    c = np.asarray(cls)
    c = c[c >= 0]
    if c.size == 0:
        return np.zeros(n)
    return np.bincount(c.astype(np.int64), minlength=n)[:n] / c.size
