"""Saxton & Rawls (2006) soil water characteristics from texture and
organic matter (Table 1, Eq. 1-22). No QGIS, no GDAL.

Units follow the paper: sand S and clay C as decimal fractions (% / 100),
organic matter OM in percent by weight; water contents as decimal volume
fractions (m3/m3); tensions in kPa; conductivity in mm/h.

Verified against the paper's Table 3 (texture-class examples at 2.5 % OM,
no density, gravel or salinity adjustment) in the test suite.

Green-Ampt wetting-front suction from the Brooks-Corey parameters
(Rawls & Brakensiek, 1983; Rawls et al., 1993):
    psi_f = (2 + 3 lambda) / (1 + 3 lambda) * psi_b / 2
with the air-entry tension psi_b = Ye (Eq. 4) and lambda = 1 / B (Eq. 18).

Eq. 4 has a standard error of 2.9 kPa and returns values near or below
zero for sands (-0.96 kPa for the paper's sand example), which would give a
negative suction. Ye is therefore bounded below at YE_MIN_KPA = 0.7 kPa,
the geometric-mean bubbling pressure of sand (7.26 cm; Rawls, Brakensiek &
Saxton, 1982), and such cells are flagged (FLAG_AIR_ENTRY).
"""

from __future__ import annotations

import numpy as np

CITATION = (
    "Saxton, K.E. and Rawls, W.J. (2006). Soil water characteristic "
    "estimates by texture and organic matter for hydrologic solutions. "
    "Soil Science Society of America Journal 70: 1569-1578. "
    "doi:10.2136/sssaj2005.0117"
)

KPA_TO_MM = 101.97162  # 1 kPa of suction = 101.97 mm of water
MINERAL_DENSITY = 2.65  # g/cm3
DF_RANGE = (0.9, 1.3)  # density factor limits (paper, Eq. 7)
YE_MIN_KPA = 0.7  # lower bound on the air-entry tension (sand, see above)

# Validity flags (bit mask) - inputs outside the calibration data.
FLAG_CLAY = 1  # clay > 60 %
FLAG_OM = 2  # OM > 8 %
FLAG_DENSITY = 4  # normal density outside 1.0-1.8 g/cm3
FLAG_DRAINABLE = 8  # theta_s - theta_33 < 0.01 (drainable porosity limited)
FLAG_AIR_ENTRY = 16  # Eq. 4 air-entry tension below YE_MIN_KPA (bounded)
FLAG_LABELS = {
    FLAG_CLAY: "clay above 60 % (outside the calibration data)",
    FLAG_OM: "organic matter above 8 % (outside the calibration data)",
    FLAG_DENSITY: "normal density outside 1.0-1.8 g/cm3",
    FLAG_DRAINABLE: "drainable porosity below 0.01 m3/m3 (Ksat unreliable)",
    FLAG_AIR_ENTRY: "air-entry tension below 0.7 kPa (bounded at 0.7 kPa)",
}


def saxton_rawls(
    sand_pct,
    clay_pct,
    om_pct,
    bulk_density=None,
    gravel_vol_pct=None,
    use_density: bool = False,
    use_gravel: bool = False,
) -> dict:
    """Saxton & Rawls (2006) for arrays (or scalars) of inputs.

    sand_pct, clay_pct : % by weight (USDA limits); om_pct : % by weight.
    bulk_density : g/cm3, used only with ``use_density`` (Eq. 7-10: the
        density factor DF = bulk density / normal density, limited to
        0.9-1.3).
    gravel_vol_pct : coarse fragments, % by volume, used only with
        ``use_gravel`` (Eq. 19-22, bulk-soil values).
    """
    S = np.asarray(sand_pct, dtype=np.float64) / 100.0
    C = np.asarray(clay_pct, dtype=np.float64) / 100.0
    OM = np.asarray(om_pct, dtype=np.float64)

    # Eq. 1 - 1500 kPa
    t1500t = (
        -0.024 * S
        + 0.487 * C
        + 0.006 * OM
        + 0.005 * (S * OM)
        - 0.013 * (C * OM)
        + 0.068 * (S * C)
        + 0.031
    )
    t1500 = t1500t + (0.14 * t1500t - 0.02)
    # Eq. 2 - 33 kPa
    t33t = (
        -0.251 * S
        + 0.195 * C
        + 0.011 * OM
        + 0.006 * (S * OM)
        - 0.027 * (C * OM)
        + 0.452 * (S * C)
        + 0.299
    )
    t33 = t33t + (1.283 * t33t**2 - 0.374 * t33t - 0.015)
    # Eq. 3 - SAT-33 kPa
    ts33t = (
        0.278 * S
        + 0.034 * C
        + 0.022 * OM
        - 0.018 * (S * OM)
        - 0.027 * (C * OM)
        - 0.584 * (S * C)
        + 0.078
    )
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    # Eq. 4 - air-entry tension (kPa)
    yet = (
        -21.67 * S
        - 27.93 * C
        - 81.97 * ts33
        + 71.12 * (S * ts33)
        + 8.29 * (C * ts33)
        + 14.05 * (S * C)
        + 27.16
    )
    ye_raw = yet + (0.02 * yet**2 - 0.113 * yet - 0.70)
    ye = np.maximum(ye_raw, YE_MIN_KPA)
    # Eq. 5-6 - saturation and normal density
    ts = t33 + ts33 - 0.097 * S + 0.043
    rho_n = (1.0 - ts) * MINERAL_DENSITY

    flags = np.zeros(np.broadcast(S, C, OM).shape, dtype=np.int16)
    flags |= np.where(C > 0.60, FLAG_CLAY, 0).astype(np.int16)
    flags |= np.where(OM > 8.0, FLAG_OM, 0).astype(np.int16)
    flags |= np.where((rho_n < 1.0) | (rho_n > 1.8), FLAG_DENSITY, 0).astype(np.int16)

    # Eq. 7-10 - density adjustment (optional)
    df = np.ones_like(ts)
    rho_adj = rho_n
    if use_density and bulk_density is not None:
        bd = np.asarray(bulk_density, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            df_raw = bd / rho_n
        df = np.where(np.isfinite(df_raw), np.clip(df_raw, *DF_RANGE), 1.0)
        rho_adj = rho_n * df
        ts_df = 1.0 - rho_adj / MINERAL_DENSITY
        t33 = t33 - 0.2 * (ts - ts_df)
        ts = ts_df

    # Eq. 11-18 - moisture-tension and conductivity
    with np.errstate(invalid="ignore", divide="ignore"):
        B = (np.log(1500.0) - np.log(33.0)) / (np.log(t33) - np.log(t1500))
        A = np.exp(np.log(33.0) + B * np.log(t33))
        lam = 1.0 / B
        drainable = np.maximum(ts - t33, 0.0)
        ksat = 1930.0 * drainable ** (3.0 - lam)
    flags |= np.where(ts - t33 < 0.01, FLAG_DRAINABLE, 0).astype(np.int16)
    flags |= np.where(ye_raw < YE_MIN_KPA, FLAG_AIR_ENTRY, 0).astype(np.int16)

    paw = t33 - t1500
    out = {
        "theta_wp": t1500,
        "theta_fc": t33,
        "theta_s": ts,
        "paw": paw,
        "rho_normal": rho_n,
        "rho_adjusted": rho_adj,
        "density_factor": df,
        "B": B,
        "A": A,
        "lambda": lam,
        "psi_e_kpa": ye,
        "psi_e_raw_kpa": ye_raw,
        "ksat_matric_mm_h": ksat,
        "ksat_mm_h": ksat,
        "gravel_weight_fraction": np.zeros_like(ts),
        "flags": flags,
    }

    # Eq. 19-22 - gravel (bulk soil), optional
    if use_gravel and gravel_vol_pct is not None:
        rv = np.clip(np.asarray(gravel_vol_pct, dtype=np.float64) / 100.0, 0, 0.9)
        rv = np.where(np.isfinite(rv), rv, 0.0)
        alpha = rho_adj / MINERAL_DENSITY
        # Eq. 19 solved for the weight fraction Rw
        rw = rv / (alpha + rv * (1.0 - alpha))
        kb_ks = (1.0 - rw) / (1.0 - rw * (1.0 - 1.5 * alpha))  # Eq. 22
        out["ksat_mm_h"] = ksat * kb_ks
        out["paw"] = paw * (1.0 - rv)  # Eq. 21
        # Bulk-soil water contents: gravel holds no water (same basis as Eq. 21)
        out["theta_wp"] = t1500 * (1.0 - rv)
        out["theta_fc"] = t33 * (1.0 - rv)
        out["theta_s"] = ts * (1.0 - rv)
        out["gravel_weight_fraction"] = rw

    with np.errstate(invalid="ignore"):
        out["psi_f_mm"] = (2.0 + 3.0 * lam) / (1.0 + 3.0 * lam) * ye / 2.0 * KPA_TO_MM
    return out


def run(inputs: dict, options: dict) -> dict:
    """Ensemble interface (see ptf/__init__.py)."""
    r = saxton_rawls(
        inputs["sand"],
        inputs["clay"],
        inputs["om"],
        bulk_density=inputs.get("bulk_density"),
        gravel_vol_pct=inputs.get("gravel"),
        use_density=options.get("density", False),
        use_gravel=options.get("gravel", False),
    )
    return {
        "theta_s": r["theta_s"],
        "theta_fc": r["theta_fc"],
        "theta_wp": r["theta_wp"],
        "paw": r["paw"],
        "ksat": r["ksat_mm_h"],
        "psi_b": r["psi_e_kpa"] * KPA_TO_MM,
        "lambda": r["lambda"],
        "psi_f": r["psi_f_mm"],
        "flags": r["flags"],
    }
