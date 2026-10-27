"""Peak-flow methods of SANRAL Drainage Manual Chapter 3 and DWS (2025)
CAPA (no QGIS dependency). Each function returns a list of MethodResult,
one per return period; a method is always calculated where its inputs
exist, and anything outside its range of application is flagged, not
refused."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import tables as tb
from .rainfall import TR102Station, areal_reduction, arf_eq310, tr102_point_depth

METHODS = {
    # id: (field code, label)
    "rational_alt2": ("RAT2", "Rational (Alternative 2: TR102 + Hershfield)"),
    "rational_alt3": ("RAT3", "Rational (Alternative 3: design rainfall DDF)"),
    "sdf": ("SDF", "Standard Design Flood"),
    "midgley_pitman": ("MP", "Midgley & Pitman (Eq 3.34)"),
    "rmf": ("RMF", "Regional Maximum Flood (Kovacs)"),
    "rmf_ratio": ("RMFR", "RMF ratios (QT/QRMF, Appendix 3D)"),
    "capa": ("CAPA", "Catchment Parameter method (CAPA, DWS 2025)"),
}
METHOD_ORDER = tuple(METHODS)

TC_FLOOR_H = 0.25  # DM p 3-20: storms shorter than 15 min are not significant


@dataclass
class MethodResult:
    method: str
    return_period: int | None  # None for the RMF itself
    q_m3s: float | None
    derivation: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)

    @property
    def code(self) -> str:
        return METHODS[self.method][0]

    @property
    def label(self) -> str:
        return METHODS[self.method][1]

    @property
    def field_name(self) -> str:
        if self.return_period is None:
            return f"Q_{self.code}"
        return f"Q_{self.code}_{self.return_period}"


@dataclass
class RunoffCoefficients:
    """Rational C components for one catchment (C input CSV): rural share
    alpha with C1 (= Cs + Cp + Cv, any dolomite reduction applied), urban
    share beta with C2, lakes share gamma with C3 (DM Eq 3.10 - 3.12 of the
    AG, Section 3.5.1.2 of the DM)."""

    alpha: float
    c1: float
    ft_class: str
    beta: float = 0.0
    c2: float = 0.0
    gamma: float = 0.0
    c3: float = 0.0
    # Optional C1 breakdown from the C CSV (Rational C tool): class shares
    # cs1_pct..cv4_pct, cs, cp, cv, c1_base (before dolomite), d_pct,
    # d_factor, map_mm. Used only by the calculation sheets.
    detail: dict = field(default_factory=dict, compare=False, repr=False)

    def ft(self, T: int) -> float:
        if T == 200:
            return tb.FT_200
        return tb.FT_FACTORS[self.ft_class][T]

    def c_t(self, T: int) -> float:
        return (
            self.alpha * self.c1 * self.ft(T)
            + self.beta * self.c2
            + self.gamma * self.c3
        )


def _area_flags(method: str, area: float) -> list[str]:
    flags = []
    if method in ("rational_alt2", "rational_alt3") and area > 15.0:
        flags.append("A > 15 km2: outside the usual Rational range (DM Table 3.2)")
    if method == "midgley_pitman":
        if area < 10.0:
            flags.append("A < 10 km2: outside the Midgley & Pitman range (DM p 3-62)")
        elif area < 100.0:
            flags.append("10 < A < 100 km2: use with caution (DM p 3-62)")
    if method in ("rmf", "rmf_ratio"):
        if area < 10.0:
            flags.append("A < 10 km2: other methods are better (DM p 3-66)")
        elif area < 300.0 or area > 20000.0:
            flags.append("A outside 300 - 20 000 km2, the best RMF range (TR137 p 17)")
    if method == "capa" and (area < 10.0 or area > 100000.0):
        flags.append("A outside 10 - 100 000 km2 (DWS 2025 Section 5.4)")
    return flags


# --------------------------------------------------------------- Rational ---
def rational(
    alt: int,
    area: float,
    tc_h: float,
    coeffs: RunoffCoefficients,
    return_periods,
    *,
    tc_floor: bool,
    apply_arf: bool,
    arf_method: str,
    arf_region: int | None,
    ddf=None,
    station: TR102Station | None = None,
) -> list[MethodResult]:
    """Rational method, DM Section 3.5.1 (AG example 3.1, steps 6 - 11):
    Q_T = C_T I_T A / 3.6 with I_T = P_T / Tc x ARF / 100.
    alt 2: P_T from TR102 + Hershfield; alt 3: P_T from the DDF table."""
    method = "rational_alt2" if alt == 2 else "rational_alt3"
    base_flags = _area_flags(method, area)
    tc_used = tc_h
    if tc_floor and tc_h < TC_FLOOR_H:
        tc_used = TC_FLOOR_H
        base_flags.append(f"Tc {tc_h * 60:.1f} min raised to 15 min (DM p 3-20)")
    total = coeffs.alpha + coeffs.beta + coeffs.gamma
    if abs(total - 1.0) > 0.01:
        base_flags.append(f"alpha + beta + gamma = {total:.3f}, not 1")
    out = []
    for T in return_periods:
        flags = list(base_flags)
        d = {"Tc_h": tc_h, "Tc_used_h": tc_used}
        if alt == 2:
            depth, how, f = tr102_point_depth(tc_used, T, station, cap_at_1day=True)
            d["rain_source"] = f"{station.label}; {how}"
            d["M_mm"], d["R_days"] = station.m_mm, station.r_days
        else:
            depth, how = ddf.depth_at_ari(tc_used * 60.0, T)
            f = []
            if how != "interp":
                f.append(f"DDF {how.replace('_', ' ')} at Tc")
            d["rain_source"] = "DDF table"
        flags += f
        d["P_mm"] = depth
        d["Pi_mmh"] = depth / tc_used
        if apply_arf:
            arf, arf_label, f = areal_reduction(
                arf_method, area, tc_used, T, arf_region
            )
            flags += f
        else:
            arf, arf_label = 100.0, "not applied"
        d["ARF_pct"], d["ARF_method"] = arf, arf_label
        d["I_mmh"] = d["Pi_mmh"] * arf / 100.0
        d["Ft"] = coeffs.ft(T)
        d["C_T"] = coeffs.c_t(T)
        q = d["C_T"] * d["I_mmh"] * area / 3.6
        out.append(MethodResult(method, T, q, d, flags))
    return out


# -------------------------------------------------------------------- SDF ---
def sdf_tc_hours(lfp_km: float, s1085: float) -> float:
    """DM Eq 3.17 (US SCS, as in HRU 1/72): Tc = (0.87 L^2 / (1000 S))^0.385."""
    return (0.87 * lfp_km**2 / (1000.0 * s1085)) ** 0.385


def sdf(
    area: float,
    lfp_km: float,
    s1085: float,
    basin: int,
    station: TR102Station,
    return_periods,
) -> list[MethodResult]:
    """Standard Design Flood, DM Section 3.5.3.5, steps 4 - 9. The procedure
    is calibrated and used exactly as published (DM p 3-47): its own Tc
    (Eq 3.17), the basin's TR102 station and Eq 3.19 ARF, whatever the
    Rational settings are."""
    info = tb.SDF_BASINS[basin]
    c2, c100 = info[4] / 100.0, info[5] / 100.0
    tc = sdf_tc_hours(lfp_km, s1085)
    arf = arf_eq310(area, tc)
    out = []
    for T in return_periods:
        depth, how, flags = tr102_point_depth(tc, T, station, cap_at_1day=False)
        c_t = c2 + tb.SDF_YT[T] / 2.33 * (c100 - c2)
        i = depth * arf / 100.0 / tc
        d = {
            "basin": basin,
            "Tc_h": tc,
            "rain_source": f"{station.label}; {how}",
            "M_mm": station.m_mm,
            "R_days": station.r_days,
            "P_mm": depth,
            "ARF_pct": arf,
            "I_mmh": i,
            "C2": c2,
            "C100": c100,
            "Y_T": tb.SDF_YT[T],
            "C_T": c_t,
        }
        out.append(MethodResult("sdf", T, c_t * i * area / 3.6, d, flags))
    return out


# -------------------------------------------------------- Midgley & Pitman ---
def midgley_pitman(
    area: float, map_mm: float, lfp_km: float, lca_km: float, s1085: float, zone: str
) -> list[MethodResult]:
    """DM Eq 3.34 - 3.35: Q_T = 0.0377 K_T P A^0.6 C^0.2, C = A sqrt(S) / (L Lc),
    K_T from Table 3.17 for T = 10, 20, 50 and 100 years."""
    c = area * math.sqrt(s1085) / (lfp_km * lca_km)
    name, kts = tb.VELD_ZONES[zone]
    flags = _area_flags("midgley_pitman", area)
    out = []
    for T, kt in zip(tb.MP_RETURN_PERIODS, kts, strict=True):
        q = 0.0377 * kt * map_mm * area**0.6 * c**0.2
        d = {"veld_zone": f"{zone} {name}", "K_T": kt, "C_param": c, "MAP_mm": map_mm}
        out.append(MethodResult("midgley_pitman", T, q, d, list(flags)))
    return out


# -------------------------------------------------------------------- RMF ---
def rmf(area: float, region: str | None, k_custom: float | None) -> MethodResult:
    """Kovacs regional maximum flood (DM Section 3.6.4, Table 3.18; TR137).
    With a custom K the Francou-Rodier equation (DM Eq 3.36) is used:
    Q = 10^6 (Ae / 10^8)^(1 - 0.1 K)."""
    flags = _area_flags("rmf", area)
    if k_custom is not None:
        q = 1e6 * (area / 1e8) ** (1.0 - 0.1 * k_custom)
        d = {"K": k_custom, "equation": "DM Eq 3.36 (custom K)"}
        return MethodResult("rmf", None, q, d, flags)
    k, trans, flood = tb.KOVACS_REGIONS[region]
    if area <= trans[0]:
        coef, exp, zone = trans[1], trans[2], "transition zone"
    else:
        coef, exp, zone = flood[1], flood[2], "flood zone"
        if area > flood[0]:
            flags.append(f"Ae above the {region} flood-zone range ({flood[0]:.0f} km2)")
    q = coef * area**exp
    d = {"region": region, "K": k, "equation": f"{coef:g} Ae^{exp:g} ({zone})"}
    return MethodResult("rmf", None, q, d, flags)


def rmf_ratio_value(region: str, T: int, area: float) -> tuple[float | None, list]:
    """Q_T/Q_RMF from Table 3D.1, linear in Ae between the area columns (this
    reproduces AG example 3.2: 3951 and 4796 m3/s). Below 10 km2 the 10 km2
    ratio is used; beyond the region's last column there is no value."""
    row = tb.RMF_RATIOS[region][T]
    areas = tb.RMF_RATIO_AREAS[: len(row)]
    flags = []
    if area <= areas[0]:
        return row[0], flags
    if area < 100.0:
        flags.append("ratio from the 'estimated' 10 - 30 km2 columns (Table 3D.1)")
    if area > areas[-1]:
        flags.append(f"Ae above the last Table 3D.1 column ({areas[-1]:.0f} km2)")
        return None, flags
    i = max(j for j in range(len(areas)) if areas[j] <= area)
    if i == len(areas) - 1:
        return row[i], flags
    w = (area - areas[i]) / (areas[i + 1] - areas[i])
    return row[i] + w * (row[i + 1] - row[i]), flags


def rmf_ratios(
    area: float, region: str, q_rmf: float, return_periods
) -> list[MethodResult]:
    out = []
    for T in return_periods:
        if T not in tb.RMF_RATIO_RETURN_PERIODS:
            continue
        ratio, flags = rmf_ratio_value(region, T, area)
        flags = _area_flags("rmf_ratio", area) + flags
        if region == "K1":
            flags.append("K1 uses the K2 ratios (Table 3D.1 note)")
        q = None if ratio is None else ratio * q_rmf
        d = {"region": region, "ratio": ratio, "Q_RMF": q_rmf}
        out.append(MethodResult("rmf_ratio", T, q, d, flags))
    return out


# ------------------------------------------------------------------- CAPA ---
def _capa_intercept(m: float) -> float:
    pts = tb.CAPA_INTERCEPTS
    lm = math.log10(m)
    for (m0, c0), (m1, c1) in zip(pts[:-1], pts[1:], strict=True):
        if m0 <= m <= m1:
            l0, l1 = math.log10(m0), math.log10(m1)
            return c0 + (lm - l0) / (l1 - l0) * (c1 - c0)
    raise ValueError("M outside the CAPA diagram")


def _capa_kp(map_mm: float) -> tuple[tuple, list]:
    rows = tb.CAPA_KP
    flags = []
    if map_mm <= rows[0][0]:
        if map_mm < rows[0][0]:
            flags.append("MAP below 100 mm: Table 5-2 K_P for 100 mm used")
        return rows[0][1], flags
    if map_mm >= rows[-1][0]:
        if map_mm > rows[-1][0]:
            flags.append("MAP above 2000 mm: Table 5-2 K_P for 2000 mm used")
        return rows[-1][1], flags
    for (m0, k0), (m1, k1) in zip(rows[:-1], rows[1:], strict=True):
        if m0 <= map_mm <= m1:
            w = (map_mm - m0) / (m1 - m0)
            return tuple(a + w * (b - a) for a, b in zip(k0, k1, strict=True)), flags
    raise AssertionError("unreachable")


def capa(
    area: float, map_mm: float, s_mean: float, lfp_km: float, return_periods
) -> list[MethodResult]:
    """Catchment Parameter method, DWS (2025) Section 5.4:
    M = MAP (100 S_A sqrt(Ae) / L)^0.5, Q50% from the digitised Figure 5-2,
    Q_P = K_P Q50% (Table 5-2, linear in MAP). S_A = mean catchment slope
    (m/m)."""
    m = map_mm * math.sqrt(100.0 * s_mean * math.sqrt(area) / lfp_km)
    flags = _area_flags("capa", area) + [
        "Q50% read from a digitised diagram (DWS 2025 Figure 5-2)"
    ]
    lo, hi = tb.CAPA_INTERCEPTS[0][0], tb.CAPA_INTERCEPTS[-1][0]
    if not lo <= m <= hi:
        flags.append(
            f"M = {m:.0f} outside the diagram ({lo:.0f} - {hi:.0f}): no result"
        )
        return [
            MethodResult("capa", T, None, {"M": m}, list(flags))
            for T in return_periods
            if T == 2 or T in tb.CAPA_KP_RETURN_PERIODS
        ]
    if not tb.CAPA_DIAGRAM_AREA[0] <= area <= tb.CAPA_DIAGRAM_AREA[1]:
        flags.append("Ae outside the diagram (1 - 10 000 km2): line extended")
    q50 = 10 ** (tb.CAPA_SLOPE * math.log10(area) + _capa_intercept(m))
    kps, f = _capa_kp(map_mm)
    flags += f
    kp_by_t = dict(zip(tb.CAPA_KP_RETURN_PERIODS, kps, strict=True))
    out = []
    for T in return_periods:
        if T == 2:
            kp = 1.0
        elif T in kp_by_t:
            kp = kp_by_t[T]
        else:
            continue
        d = {"M": m, "Q50_m3s": q50, "K_P": kp, "MAP_mm": map_mm, "S_A": s_mean}
        out.append(MethodResult("capa", T, kp * q50, d, list(flags)))
    return out
