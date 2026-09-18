"""
Sample L-moment computation - clean-room implementation of Hosking's
(1990) unbiased probability-weighted-moment estimators, from the
published formulas in Hosking & Wallis (1997), "Regional Frequency
Analysis: An Approach Based on L-Moments". Not a wrapper around any
third-party L-moments package.
"""

from __future__ import annotations

import numpy as np


def sample_l_moments(x) -> dict:
    """Returns {'l1', 'l2', 'l3', 'l4', 't2', 't3', 't4'} for a 1-D
    sample. t2 = L-CV, t3 = L-skewness, t4 = L-kurtosis.

    Uses the unbiased PWM estimators (Landwehr et al. 1979):
        b0 = mean(x)
        b_r = (1/n) * sum_{i=r+1}^{n} [ C(i-1,r) / C(n-1,r) ] * x(i)
    for sorted x(1) <= ... <= x(n), then converts PWMs to L-moments via
    the standard linear combinations (Hosking 1990, eq. 2.3).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 2:
        raise ValueError(f"Need at least 2 values to compute L-moments, got {n}.")

    i = np.arange(1, n + 1, dtype=float)  # 1-indexed rank

    b0 = x.mean()

    b1 = np.nan
    b2 = np.nan
    b3 = np.nan
    if n >= 2:
        b1 = np.sum(((i - 1) / (n - 1)) * x) / n
    if n >= 3:
        b2 = np.sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2))) * x) / n
    if n >= 4:
        b3 = (
            np.sum(((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3))) * x)
            / n
        )

    l1 = b0
    l2 = 2 * b1 - b0 if n >= 2 else np.nan
    l3 = (6 * b2 - 6 * b1 + b0) if n >= 3 else np.nan
    l4 = (20 * b3 - 30 * b2 + 12 * b1 - b0) if n >= 4 else np.nan

    t2 = l2 / l1 if l1 else np.nan
    t3 = l3 / l2 if (n >= 3 and l2) else np.nan
    t4 = l4 / l2 if (n >= 4 and l2) else np.nan

    return {
        "l1": float(l1),
        "l2": float(l2),
        "l3": float(l3) if n >= 3 else None,
        "l4": float(l4) if n >= 4 else None,
        "t2": float(t2),
        "t3": float(t3) if n >= 3 else None,
        "t4": float(t4) if n >= 4 else None,
        "n": n,
    }
