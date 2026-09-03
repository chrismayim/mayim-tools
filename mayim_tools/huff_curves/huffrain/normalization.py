"""
Dimensionless event curve construction and resampling (paper Section
3.7-3.8).

Events with missing interior data are never normalized into a curve -
there is no defensible way to build a cumulative rainfall sequence
through an unknown gap without inventing a value, which the paper
explicitly prohibits. Such events remain visible in the event
inventory (for auditability) but simply don't contribute a curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import EventCurve, RainfallEvent


def normalize_event(event: RainfallEvent, intervals: "pd.DataFrame") -> EventCurve | None:
    if event.contains_missing:
        return None
    if event.wet_duration_s <= 0 or event.total_depth_mm <= 0:
        return None

    sub = intervals.loc[list(event.interval_positions)]
    wet_only = sub[sub["depth_mm"].notna()]
    if wet_only.empty:
        return None

    t0 = wet_only["timestamp_start"].iloc[0]
    cum = wet_only["depth_mm"].cumsum().to_numpy()
    t_end = (wet_only["timestamp_end"] - t0).dt.total_seconds().to_numpy()

    D = event.wet_duration_s
    P = event.total_depth_mm

    # Prepend the (0, 0) origin point explicitly (paper: "both endpoints
    # must be included: (0,0) and (1,1)").
    x_raw = np.concatenate([[0.0], t_end / D])
    y_raw = np.concatenate([[0.0], cum / P])
    # Numerical safety: clip to [0, 1] and force the final point to
    # exactly (1, 1) rather than trusting floating-point accumulation.
    x_raw = np.clip(x_raw, 0.0, 1.0)
    y_raw = np.clip(y_raw, 0.0, 1.0)
    x_raw[-1] = 1.0
    y_raw[-1] = 1.0

    nominal_interval_s = float(np.median(wet_only["duration_s"])) if len(wet_only) else D
    n_eff = D / nominal_interval_s if nominal_interval_s > 0 else float(len(wet_only))

    return EventCurve(
        event_id=event.event_id,
        quartile=event.quartile,
        x_raw=tuple(x_raw), y_raw=tuple(y_raw),
        x_grid=(), y_grid=(),  # filled in by resample_to_grid
        n_eff=n_eff, resolution_warning=False,
    )


def resample_to_grid(curve: EventCurve, grid: np.ndarray) -> EventCurve:
    """Linear interpolation onto the common grid - the paper's
    'authoritative empirical curve' method (Section 3.8). No
    extrapolation beyond [0,1]; monotonicity and bounds are enforced
    (np.interp is inherently monotone-preserving for a monotone input,
    which y_raw always is here since it's a cumulative sum of
    non-negative depths)."""
    x_raw = np.asarray(curve.x_raw)
    y_raw = np.asarray(curve.y_raw)

    # Deduplicate any repeated x (can happen with same-timestamp events
    # collapsed to a single grid point) - keep the last (highest
    # cumulative) y at each unique x, since np.interp requires strictly
    # increasing x.
    if len(np.unique(x_raw)) < len(x_raw):
        x_raw, idx = np.unique(x_raw, return_index=False), None
        # rebuild y by taking the max y at each unique x
        y_dedup = []
        xr = np.asarray(curve.x_raw)
        yr = np.asarray(curve.y_raw)
        for xv in x_raw:
            y_dedup.append(yr[xr == xv].max())
        y_raw = np.array(y_dedup)

    y_grid = np.interp(grid, x_raw, y_raw, left=0.0, right=1.0)
    y_grid = np.clip(y_grid, 0.0, 1.0)
    y_grid = np.maximum.accumulate(y_grid)  # belt-and-braces monotonicity guard

    # Resolution warning: flag if the grid is finer than the event's
    # effective observation count can support (paper Section 3.8).
    resolution_warning = curve.n_eff > 0 and (len(grid) - 1) > curve.n_eff

    return EventCurve(
        event_id=curve.event_id, quartile=curve.quartile,
        x_raw=curve.x_raw, y_raw=curve.y_raw,
        x_grid=tuple(grid), y_grid=tuple(y_grid),
        n_eff=curve.n_eff, resolution_warning=bool(resolution_warning),
    )
