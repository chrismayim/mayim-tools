"""Probability of each hydrologic soil group from the Ksat P5 / P50 / P95
of the Regional soil parameterisation. No QGIS, no GDAL.

Per cell, ln(Ksat) is described by a two-piece normal distribution with
its median at ln(P50): sigma_lo = (ln P50 - ln P5) / 1.645 below the
median and sigma_hi = (ln P95 - ln P50) / 1.645 above it. Each half
carries half the probability, so the distribution reproduces P5, P50 and
P95 exactly (the same footing as the SoilGrids input uncertainty).

The least transmissive layer of a depth range is the minimum over the
layers. Its quantiles are taken as the minimum of the layer quantiles,
which is exact when the layers' uncertainties move together (comonotonic):
they share the same soil products and pedotransfer functions, so a wet
draw in one layer is mostly a wet draw in the others.
"""

from __future__ import annotations

import numpy as np

from . import neh630

Z95 = 1.6448536269514722


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (vectorised, |error| < 1.5e-7; Abramowitz &
    Stegun 7.1.26 for erf). +-inf map to 1 / 0."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    with np.errstate(over="ignore", invalid="ignore"):
        erf = 1.0 - poly * np.exp(-z * z)
    erf = np.where(np.isinf(z), 1.0, erf)
    return 0.5 * (1.0 + np.sign(x) * erf)


def cdf(threshold: float, p05, p50, p95) -> np.ndarray:
    """P(Ksat <= threshold) per cell; NaN where P50 is unknown."""
    p05, p50, p95 = (np.asarray(a, dtype=float) for a in (p05, p50, p95))
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log(p50)
        lt = np.log(threshold)
        s_lo = (lm - np.log(p05)) / Z95
        s_hi = (np.log(p95) - lm) / Z95
        below = lt < lm
        sig = np.where(below, s_lo, s_hi)
        d = lt - lm
        z = np.where(sig > 0, d / np.where(sig > 0, sig, 1.0), np.sign(d) * np.inf)
        z = np.where(d == 0, 0.0, z)
    # Missing bounds: the distribution collapses to its median (a step).
    out = np.where(np.isfinite(sig) | (d == 0), _phi(z), np.where(d > 0, 1.0, 0.0))
    out = np.where(np.isfinite(lm) & (p50 > 0), out, np.nan)
    return out


def group_probabilities(p05, p50, p95, thresholds) -> np.ndarray:
    """(4, cells...) probabilities of A, B, C, D for Ksat quantiles of the
    least transmissive layer and one threshold set (A|B, B|C, C|D)."""
    t_ab, t_bc, t_cd = thresholds
    f_ab = cdf(t_ab, p05, p50, p95)
    f_bc = cdf(t_bc, p05, p50, p95)
    f_cd = cdf(t_cd, p05, p50, p95)
    pa = 1.0 - f_ab
    pb = f_ab - f_bc
    pc = f_bc - f_cd
    pd = f_cd
    out = np.stack([pa, pb, pc, pd])
    return np.clip(out, 0.0, 1.0)


def probabilities(quantiles: dict, case: np.ndarray) -> np.ndarray:
    """(4, rows, cols) probabilities of the drained letter A, B, C, D per
    cell (for dual cells: A/D, B/D, C/D, D). ``quantiles`` maps 'P05' /
    'P50' / 'P95' to {layer: array}. NaN where unknown."""
    case = np.asarray(case)
    out = np.full((4,) + case.shape, np.nan)
    sel1 = case == neh630.CASE_IMPERMEABLE_SHALLOW
    out[:, sel1] = np.array([0.0, 0.0, 0.0, 1.0])[:, None]
    for code, (_, rng, thresholds, _dual) in neh630.CASES.items():
        if rng is None:
            continue
        sel = case == code
        if not sel.any():
            continue
        q = {
            tag: neh630.least_transmissive(quantiles[tag], rng)
            for tag in ("P05", "P50", "P95")
        }
        p = group_probabilities(q["P05"], q["P50"], q["P95"], thresholds)
        out[:, sel] = p[:, sel]
    return out
