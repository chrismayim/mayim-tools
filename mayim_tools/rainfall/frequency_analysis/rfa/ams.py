"""
Annual maximum series (AMS) extraction (Stage 1.1).

Two-layer missing-data handling (see the conversation record for the
full reasoning):
    1. WINDOW level: a rolling-sum window touching ANY missing native-
       interval reading is NaN, never treated as zero. This uses
       pandas' rolling(...).sum() with min_periods equal to the full
       window size, which achieves exactly this (pandas only omits NaN
       within a window when computing the sum itself, but a window
       containing fewer than min_periods non-NaN values is NaN - so
       setting min_periods to the full window size is what forces "any
       NaN in the window -> NaN result", not the default omit-and-sum
       behaviour).
    2. YEAR level: a calendar year is only included in a duration's AMS
       if its overall data completeness (valid native-interval readings
       / expected readings for that year) meets a threshold (default
       90%). This is a systematic safeguard against a single missing
       period silently coinciding with the year's true peak event -
       every year's completeness and inclusion status is still reported
       regardless, so the decision stays visible rather than silent.
"""

from __future__ import annotations

import pandas as pd

from .schemas import DurationSeries, YearRecord
from .timebase import duration_label


def build_regular_grid(parsed: pd.DataFrame, native_interval_min: float) -> pd.Series:
    """Reindexes to a complete regular grid at the native interval,
    covering the full record span. Missing timestamps become NaN depth
    - never zero."""
    full_index = pd.date_range(
        parsed["timestamp"].min(),
        parsed["timestamp"].max(),
        freq=pd.Timedelta(minutes=native_interval_min),
    )
    series = parsed.set_index("timestamp")["depth_mm"]
    return series.reindex(full_index)


def compute_year_completeness(
    grid: pd.Series, native_interval_min: float, min_completeness: float
) -> list:
    """Returns one YearRecord per calendar year spanned by the grid.

    Completeness is computed against a FULL calendar year's expected
    reading count, not against however many grid points happen to
    exist for that year in the record. This matters at the record's
    start/end: a partial year (e.g. the record starts in June) must
    NOT look artificially 100% complete just because every reading
    within its truncated span happens to be valid - a half-year of
    data cannot represent a genuine annual maximum, and should be
    excluded by the completeness threshold, not accidentally pass it.
    """
    records = []
    for year, group in grid.groupby(grid.index.year):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year + 1, month=1, day=1)
        n_expected = int(
            (year_end - year_start).total_seconds() / 60 / native_interval_min
        )
        n_valid = int(group.notna().sum())
        completeness = n_valid / n_expected if n_expected else 0.0
        included = completeness >= min_completeness
        reason = (
            None
            if included
            else f"completeness {completeness:.1%} below threshold {min_completeness:.0%}"
        )
        records.append(
            YearRecord(
                year=int(year),
                n_expected=n_expected,
                n_valid=n_valid,
                completeness=completeness,
                included=included,
                exclusion_reason=reason,
            )
        )
    return records


def extract_ams_for_duration(
    grid: pd.Series,
    duration_min: float,
    native_interval_min: float,
    year_records: list,
) -> DurationSeries:
    """Extracts the annual maximum series for one duration, restricted
    to years that passed the completeness check."""
    window = max(1, round(duration_min / native_interval_min))
    rolling_sum = grid.rolling(window=window, min_periods=window).sum()

    included_years = {yr.year for yr in year_records if yr.included}

    years_out = []
    values_out = []
    for year in sorted(included_years):
        year_mask = rolling_sum.index.year == year
        year_vals = rolling_sum[year_mask]
        if year_vals.notna().any():
            years_out.append(year)
            values_out.append(float(year_vals.max()))
        # a year can pass the OVERALL completeness check yet still have
        # zero valid windows for this specific duration in a rare edge
        # case (e.g. the one gap happens to be exactly duration-sized
        # and centred) - simply contributes no value for this duration,
        # which is correct, not an error.

    return DurationSeries(
        duration_label=duration_label(duration_min),
        duration_minutes=duration_min,
        years=tuple(years_out),
        values_mm=tuple(values_out),
        year_records=year_records,
        n_native_intervals_per_window=window,
    )
