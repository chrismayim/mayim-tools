"""
Catchment run-off coefficients from area shares - the arithmetic of the
SANRAL Drainage Manual (2013) calculation form (Appendix 3A), pure Python.

Given the rural class shares (slope, permeability, vegetation; design,
lower and upper), the urban C2 (design, lower, upper), the area fractions
alpha / beta / gamma, the MAP, the dolomite share and the Ft class, it
returns C1, C1D, the Ft class, C2, C3 and the C_T table for 2-100 years and
the PMF.

Uncertainty: C1 P05 / P95 combine the soil (permeability) and land-cover
(vegetation) ranges as independent 90 % ranges by root-sum-square on each
side, as in the Curve numbers tool; C2 P05 / P95 are the ends of the
Table 3.7 ranges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import tables as tb

MAP_EDGE_MM = 50.0  # report C1 for the neighbouring band this close to a band edge
WATER_FLAG = 0.05


@dataclass
class Shares:
    slope: tuple  # 4 fractions of the rural area
    perm: tuple
    veg: tuple
    perm_lo: tuple | None = None
    perm_hi: tuple | None = None
    veg_lo: tuple | None = None
    veg_hi: tuple | None = None


@dataclass
class CatchmentC:
    alpha: float
    beta: float
    gamma: float
    map_mm: float
    band: int
    cs: float = 0.0
    cp: float = 0.0
    cv: float = 0.0
    c1: float = 0.0
    c1_p05: float = 0.0
    c1_p95: float = 0.0
    soil_share: float = math.nan  # share of the C1 uncertainty from the soil
    d_fraction: float = 0.0
    d_factor: float = 0.0
    c1d: float = 0.0
    ft_score: float = 0.0
    ft_class: str = "flat_permeable"
    ft_source: str = "score"
    ft_borderline: bool = False
    c2: float = 0.0
    c2_p05: float = 0.0
    c2_p95: float = 0.0
    c3: float = 1.0
    ft_on_c2: bool = False
    c_t: dict = field(default_factory=dict)  # period -> C_T (and "PMF")
    c_t_other_ft: dict = field(default_factory=dict)  # borderline: the other row
    c1_neighbour_band: tuple | None = None  # (band, C1) near a MAP band edge


def _c(shares, table, band):
    return sum(f * table[i][band] for i, f in enumerate(shares))


def _rss(d1: float, d2: float) -> float:
    return math.sqrt(max(d1, 0.0) ** 2 + max(d2, 0.0) ** 2)


def neighbour_band(map_mm: float):
    """Index of the neighbouring MAP band when within MAP_EDGE_MM of a band
    edge, else None."""
    for edge, below, above in ((600.0, 0, 1), (900.0, 1, 2)):
        if abs(map_mm - edge) <= MAP_EDGE_MM:
            return above if tb.map_band(map_mm) == below else below
    return None


def catchment_c(
    sh: Shares | None,
    alpha: float,
    beta: float,
    gamma: float,
    map_mm: float,
    c2: tuple = (0.0, 0.0, 0.0),
    c3: float = 1.0,
    d_fraction: float = 0.0,
    ft_override: str | None = None,
    ft_on_c2: bool = False,
) -> CatchmentC:
    band = tb.map_band(map_mm)
    r = CatchmentC(alpha, beta, gamma, map_mm, band, c3=c3, ft_on_c2=ft_on_c2)
    r.c2, r.c2_p05, r.c2_p95 = (float(v) for v in c2)
    if sh is not None and alpha > 0:
        r.cs = _c(sh.slope, tb.CS, band)
        r.cp = _c(sh.perm, tb.CP, band)
        r.cv = _c(sh.veg, tb.CV, band)
        r.c1 = r.cs + r.cp + r.cv
        cp_lo = _c(sh.perm_lo or sh.perm, tb.CP, band)
        cp_hi = _c(sh.perm_hi or sh.perm, tb.CP, band)
        cv_lo = _c(sh.veg_lo or sh.veg, tb.CV, band)
        cv_hi = _c(sh.veg_hi or sh.veg, tb.CV, band)
        ds_lo, dv_lo = r.cp - cp_lo, r.cv - cv_lo
        ds_hi, dv_hi = cp_hi - r.cp, cv_hi - r.cv
        r.c1_p05 = r.c1 - _rss(ds_lo, dv_lo)
        r.c1_p95 = r.c1 + _rss(ds_hi, dv_hi)
        soil = max(ds_lo, 0) ** 2 + max(ds_hi, 0) ** 2
        tot = soil + max(dv_lo, 0) ** 2 + max(dv_hi, 0) ** 2
        r.soil_share = soil / tot if tot > 0 else math.nan
        r.d_fraction = d_fraction
        r.d_factor = tb.dolomite_factor(sh.slope)
        r.c1d = tb.c1_dolomite(r.c1, d_fraction, r.d_factor)
        r.ft_score = tb.ft_score(sh.slope, sh.perm)
        r.ft_class = tb.ft_class_from_score(r.ft_score)
        lo, hi = tb.FT_BORDERLINE
        r.ft_borderline = lo <= r.ft_score <= hi
        nb = neighbour_band(map_mm)
        if nb is not None:
            c1n = (
                _c(sh.slope, tb.CS, nb) + _c(sh.perm, tb.CP, nb) + _c(sh.veg, tb.CV, nb)
            )
            r.c1_neighbour_band = (nb, tb.c1_dolomite(c1n, d_fraction, r.d_factor))
    if ft_override:
        r.ft_class, r.ft_source = ft_override, "override"
        r.ft_borderline = False
    r.c_t = c_t_table(r, r.ft_class, sh)
    if r.ft_borderline:
        other = (
            "flat_permeable"
            if r.ft_class == "steep_impermeable"
            else "steep_impermeable"
        )
        r.c_t_other_ft = c_t_table(r, other, sh)
    return r


def c_t_table(r: CatchmentC, ft_class: str, sh: Shares | None) -> dict:
    out = {}
    for t in tb.RETURN_PERIODS:
        f = tb.ft(ft_class, t)
        c2 = r.c2 * f if r.ft_on_c2 else r.c2
        out[t] = r.alpha * r.c1d * f + r.beta * c2 + r.gamma * r.c3
    c1p = tb.c1_pmf(sh.slope, r.band) if sh is not None and r.alpha > 0 else 0.0
    if r.d_fraction > 0:
        c1p = tb.c1_dolomite(c1p, r.d_fraction, r.d_factor)
    out["PMF"] = r.alpha * c1p + r.beta * 1.0 + r.gamma * r.c3
    return out
