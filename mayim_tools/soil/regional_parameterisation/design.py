"""Candidate design parameters per zone and layer (no QGIS, no GDAL).

Turns the Monte Carlo results into one central value per zone, layer and
parameter, with a sensitivity range, the spread between products and
methods, a confidence rating, the main source of uncertainty and a
recommendation. The values are CANDIDATES for the engineer to confirm: the
tool stays guideline-neutral.

Areal aggregation (cells in the zone):
- Ksat and the suctions are log-normally distributed, so the areal value is
  the geometric mean; water contents use the arithmetic mean.
- The sensitivity range aggregates the cell P5 and P95 maps the same way.
  This treats the errors as fully correlated in space (every cell at its P5
  together), a deliberately wide, conservative range for a zone value.
- The product x method range uses each product's central soil run through
  each method (see core.central).

Confidence:
- log parameters: U = sqrt(P95 / P5) of the zone range and R = max / min of
  the product x method values. High: U <= 2 and R <= 2; Medium: U <= 4 and
  R <= 4; otherwise Low.
- water contents: h = (P95 - P5) / (2 x central) and s = (max - min) /
  central. High: h <= 0.15 and s <= 0.10; Medium: h <= 0.30 and s <= 0.25;
  otherwise Low.
"""

from __future__ import annotations

import math

import numpy as np

from .ptf import PARAMETER_BY_CODE

DESIGN_PARAMS = ("ksat", "psi_f", "theta_s", "theta_fc", "theta_wp", "paw")
EFFECTIVE_K_FACTOR = 0.5  # Bouwer (1966): wetted-zone K about half of Ksat

CONFIDENCE_RULES = {
    "log": {"High": (2.0, 2.0), "Medium": (4.0, 4.0)},
    "linear": {"High": (0.15, 0.10), "Medium": (0.30, 0.25)},
}
SOURCE_TEXT = {
    "input": "input data (the soil maps' own uncertainty)",
    "product": "disagreement between the soil products",
    "method": "choice of pedotransfer method",
    "interaction": "product x method interaction",
}
ACTION_TEXT = {
    "input": "site soil sampling (texture, organic carbon, bulk density) or "
    "infiltration tests",
    "product": "field verification of soil texture (the two maps disagree)",
    "method": "infiltration tests or calibration on observed runoff (the "
    "methods disagree)",
    "interaction": "infiltration tests or calibration on observed runoff",
}


def areal(values: np.ndarray, log: bool) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
        return float(10 ** np.mean(np.log10(v))) if v.size else math.nan
    return float(np.mean(v)) if v.size else math.nan


def confidence(code: str, central, low, high, spread_min, spread_max) -> str:
    log = PARAMETER_BY_CODE[code].log
    if not all(np.isfinite([central, low, high])) or central <= 0:
        return "Low"
    if log:
        u = math.sqrt(high / low) if low > 0 else math.inf
        r = (
            spread_max / spread_min
            if np.isfinite(spread_min) and spread_min > 0
            else 1.0
        )
        rules = CONFIDENCE_RULES["log"]
    else:
        u = (high - low) / (2.0 * central)
        r = (
            (spread_max - spread_min) / central
            if np.isfinite(spread_min) and np.isfinite(spread_max)
            else 0.0
        )
        rules = CONFIDENCE_RULES["linear"]
    for level in ("High", "Medium"):
        lim_u, lim_r = rules[level]
        if u <= lim_u and r <= lim_r:
            return level
    return "Low"


def recommendation(level: str, source: str) -> str:
    if level == "High":
        return "Use the candidate value; confirm against local data where available."
    if level == "Medium":
        return (
            "Use the candidate value and run the model at the sensitivity "
            "bounds; reduce the range with "
            + ACTION_TEXT.get(source, "local data")
            + "."
        )
    return (
        "Indicative only: run the model at both sensitivity bounds and confirm "
        "with " + ACTION_TEXT.get(source, "local data") + " before design."
    )


def design_rows(result, short_names: dict) -> list[dict]:
    """One row per zone x layer x design parameter."""
    rows = []
    methods = result.settings.mc.methods
    for z, zname in enumerate(result.zone_names, start=1):
        zmask = result.zone_raster == z
        for lab, _, _ in result.settings.layers:
            if lab not in result.texture_class:
                continue
            for code in DESIGN_PARAMS:
                if code not in result.stats:
                    continue
                prm = PARAMETER_BY_CODE[code]
                st = result.stats[code][lab]
                cvalue = areal(st[1][zmask], prm.log)
                low = areal(st[0][zmask], prm.log)
                high = areal(st[2][zmask], prm.log)
                members = {}
                for p in result.products:
                    pc = result.central.get(p.name, {}).get(lab, {})
                    by = pc.get("by_method", {})
                    for m in methods:
                        if code in by.get(m, {}):
                            val = areal(by[m][code][zmask], prm.log)
                            if np.isfinite(val):
                                members[f"{short_names.get(p.name, p.name)} {m}"] = val
                vals = list(members.values())
                smin = min(vals) if vals else math.nan
                smax = max(vals) if vals else math.nan
                parts = result.variance.get(z, {}).get(lab, {}).get(code, {})
                total = parts.get("total", 0.0)
                shares = {
                    k: parts.get(k, 0.0) / total if total > 0 else 0.0
                    for k in ("input", "product", "method", "interaction")
                }
                source = max(shares, key=shares.get)
                level = confidence(code, cvalue, low, high, smin, smax)
                row = {
                    "Zone": zname,
                    "Layer": lab,
                    "Parameter": code,
                    "Label": prm.label,
                    "Units": prm.units,
                    "Candidate value": cvalue,
                    "Sensitivity low (P5)": low,
                    "Sensitivity high (P95)": high,
                    "Product x method min": smin,
                    "Product x method max": smax,
                    "Confidence": level,
                    "Main uncertainty source": source,
                    "Share of variance (%)": 100.0 * shares[source],
                    "Recommendation": recommendation(level, source),
                    "Aggregation": "geometric mean" if prm.log else "arithmetic mean",
                }
                for name, val in members.items():
                    row[f"Member {name}"] = val
                if code == "ksat":
                    row["Effective K (0.5 Ksat)"] = EFFECTIVE_K_FACTOR * cvalue
                rows.append(row)
    return rows
