"""
Native interval detection and target-duration list handling (Stage 1.1
setup, before AMS extraction itself).
"""

from __future__ import annotations

import pandas as pd

# Standard target durations requested, in minutes.
STANDARD_DURATIONS_MIN = (
    30,
    60,
    120,
    180,
    240,
    300,
    360,
    480,
    600,
    720,
    840,
    960,
    1200,
    1440,
)


def detect_native_interval(timestamps: pd.Series) -> float:
    """Mode of positive timestamp differences, in minutes - robust to
    a handful of gaps/duplicates, same approach used in the Design
    Rainfall plugin's timebase.py."""
    diffs_min = timestamps.diff().dt.total_seconds() / 60.0
    positive = diffs_min.dropna()
    positive = positive[positive > 0]
    if positive.empty:
        raise ValueError(
            "Could not determine a native time interval - fewer than 2 valid timestamps."
        )
    mode_vals = positive.mode()
    return float(mode_vals.iloc[0]) if not mode_vals.empty else float(positive.median())


def applicable_durations(
    native_interval_min: float, requested_min: tuple = STANDARD_DURATIONS_MIN
) -> list:
    """Filters the standard duration list to those the data can
    actually support (duration >= native interval), per the user's own
    'depending on the data interval obviously' framing. Durations not
    an exact multiple of the native interval are still included (the
    rolling window just uses round(duration/native_interval) steps)
    but flagged - see ams.py."""
    return [d for d in requested_min if d >= native_interval_min]


def duration_label(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)} min"
    hours = minutes / 60
    if hours == int(hours):
        return f"{int(hours)} h"
    return f"{hours:g} h"
