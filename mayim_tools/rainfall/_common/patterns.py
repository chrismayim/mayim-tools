"""
Temporal-pattern helpers shared by the rainfall tools: the target
duration / timestep / increment-count table, Early/Middle/Late
peak-position classification and normalised mass curves. Moved from
design_storm_ensembles/core.py (behaviour unchanged).
"""

from __future__ import annotations

import numpy as np

TARGET_DURATIONS = [
    # (duration_minutes, timestep_minutes, increments)
    (10, 5, 2),
    (15, 5, 3),
    (20, 5, 4),
    (25, 5, 5),
    (30, 5, 6),
    (45, 5, 9),
    (60, 5, 12),
    (90, 5, 18),
    (120, 5, 24),
    (180, 15, 12),
    (270, 15, 18),
    (360, 15, 24),
    (540, 30, 18),
    (720, 30, 24),
    (1080, 60, 18),
    (1440, 60, 24),
    (1800, 120, 15),
    (2160, 120, 18),
    (2880, 120, 24),
    (4320, 180, 24),
    (5760, 180, 32),
    (7200, 180, 40),
    (8640, 180, 48),
    (10080, 180, 56),
]


SHAPES = ["Early", "Middle", "Late"]


def classify_shape(window):
    vals = window["values"]
    n = len(vals)
    if n == 0:
        return None
    peak_pos = int(np.argmax(vals))
    frac = (peak_pos + 0.5) / n
    if frac < 1.0 / 3.0:
        return "Early"
    elif frac < 2.0 / 3.0:
        return "Middle"
    return "Late"


def normalized_cumulative_curve(values):
    """Return (cum_time_frac, cum_depth_frac) each starting at (0,0)
    and ending at (1,1), from a native-resolution increment array."""
    n = len(values)
    total = values.sum()
    cum_time = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    if total <= 0:
        cum_depth = np.concatenate([[0.0], np.linspace(0, 1, n)])
    else:
        cum_depth = np.concatenate([[0.0], np.cumsum(values) / total])
    return cum_time, cum_depth
