"""
Percentile Huff curve summarization (paper Section 3.10).

Uses numpy's 'median_unbiased' quantile method, which corresponds to
Hyndman & Fan (1996) type 8 - the paper's explicit recommendation
("a documented quantile estimator, such as Hyndman-Fan type 8, kept
fixed across analyses").
"""

from __future__ import annotations

import numpy as np

from .schemas import EventCurve, HuffCurveSet

DEFAULT_PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_SAMPLE_DEFAULT = 5  # paper item 49: "require a minimum sample size per quartile and stratum"


def summarize_curves(
    curves: list[EventCurve],
    grid: np.ndarray,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    min_sample: int = MIN_SAMPLE_DEFAULT,
) -> list[HuffCurveSet]:
    """One HuffCurveSet per quartile present in `curves` (only curves
    with a definite, non-tied quartile are included - see
    classification.py). Returns an empty list if no curves qualify."""
    out = []
    for q in (1, 2, 3, 4):
        q_curves = [c for c in curves if c.quartile == q]
        n = len(q_curves)
        if n == 0:
            continue

        y_matrix = np.array([c.y_grid for c in q_curves])  # shape (n_events, len(grid))

        pct_results = {}
        for p in percentiles:
            pct_results[p] = tuple(np.percentile(y_matrix, p * 100, axis=0, method="median_unbiased"))

        out.append(HuffCurveSet(
            quartile=q,
            x_grid=tuple(grid),
            percentiles=pct_results,
            n_events=n,
            mean_curve=tuple(y_matrix.mean(axis=0)),
            min_curve=tuple(y_matrix.min(axis=0)),
            max_curve=tuple(y_matrix.max(axis=0)),
            insufficient_sample=n < min_sample,
            min_sample_threshold=min_sample,
        ))
    return out
