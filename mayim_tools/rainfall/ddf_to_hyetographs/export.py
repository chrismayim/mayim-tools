"""
CSV export for Alternating Block Method hyetographs.

Three wide-format outputs, all sharing the same row/column structure
(one row per Return Period + Target Duration, one column block per
peak ratio in order 25/33/50/66/75%, each column labelled by the
actual cumulative time it represents) so they're directly comparable
side by side - only the VALUE at each cell differs:
- incremental depth (mm) - the original, primary output
- cumulative depth (mm) - running total within each block's peak ratio
- incremental intensity (mm/hr) - incremental depth divided by that
  block's own time-step width in hours

Shorter-duration rows leave trailing columns blank, same as before.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .core import PEAK_RATIOS, build_hyetograph


def _n_steps_for_duration(target_duration_min: float, timestep_min: float) -> int:
    """Exact step count for a given target duration - matches
    compute_incremental_blocks()'s own logic in core.py precisely, so
    the CSV header's column count always matches the real data every
    row can produce. Found via a real off-by-one bug: the original
    header sizing unconditionally added one extra column (intended for
    a duration that isn't an exact multiple of the timestep), even for
    a duration that divides evenly - producing a spurious, always-
    empty trailing column (confirmed directly: 60 min at 5 min steps
    generated a 13th '65min' column with no corresponding data)."""
    n_full_steps = int(target_duration_min // timestep_min)
    remainder = target_duration_min - n_full_steps * timestep_min
    return max(1, n_full_steps + (1 if remainder > 1e-9 else 0))


def _format_time_label(minutes: float) -> str:
    if minutes == int(minutes):
        return f"{int(minutes)}min"
    return f"{minutes:g}min"


def _write_wide(
    return_periods_and_durations: list,
    timestep_min: float,
    tables: dict,
    value_key: str,
    path: str | Path,
    decimals: int,
    allow_short_duration_extrapolation: bool = False,
) -> int:
    """Shared writer for all three value types. return_periods_and_durations:
    list of (return_period_years, target_duration_min) pairs, in the
    order they should appear as rows. tables: {return_period_years:
    table_pairs} from parse_idf_table(). value_key: 'incremental',
    'cumulative', or 'intensity_mmhr'. Returns the number of rows
    written.

    Column labels use fixed multiples of timestep_min (e.g. '25%_5min',
    '25%_10min', ...) rather than generic step numbers, so a reader
    can tell at a glance what time each column represents. This is
    exact for every row whose target duration is an exact multiple of
    timestep_min (the common case); a row whose duration isn't an
    exact multiple has its own final populated step clipped to that
    row's actual duration - a known, minor labelling approximation for
    that one cell only, not a computation error (the underlying value
    is still that step's correct clipped depth/intensity).

    allow_short_duration_extrapolation: passed through to
    build_hyetograph() - needed whenever timestep_min is finer than
    the table's own shortest tabulated duration (e.g. a 5-min timestep
    against an ERA5-derived hourly-only DDF table). See
    core.py's extrapolate_short_duration() for the method and its
    known limitation.
    """
    max_steps = max(
        _n_steps_for_duration(d, timestep_min) for _, d in return_periods_and_durations
    )

    header = ["ReturnPeriodYears", "TargetDurationMin"]
    for ratio in PEAK_RATIOS:
        pct = f"{int(ratio * 100)}%"
        for step in range(1, max_steps + 1):
            header.append(f"{pct}_{_format_time_label(step * timestep_min)}")

    n_rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for rt, target_duration in return_periods_and_durations:
            hyeto = build_hyetograph(
                target_duration,
                timestep_min,
                tables[rt],
                allow_short_duration_extrapolation,
            )
            row = [rt, target_duration]
            for ratio in PEAK_RATIOS:
                values = hyeto[ratio][value_key]
                padded = [round(v, decimals) for v in values] + [""] * (
                    max_steps - len(values)
                )
                row.extend(padded)
            w.writerow(row)
            n_rows += 1

    return n_rows


def write_incremental_depth(
    return_periods_and_durations,
    timestep_min,
    tables,
    path,
    allow_short_duration_extrapolation: bool = False,
) -> int:
    return _write_wide(
        return_periods_and_durations,
        timestep_min,
        tables,
        "incremental",
        path,
        decimals=4,
        allow_short_duration_extrapolation=allow_short_duration_extrapolation,
    )


def write_cumulative_depth(
    return_periods_and_durations,
    timestep_min,
    tables,
    path,
    allow_short_duration_extrapolation: bool = False,
) -> int:
    return _write_wide(
        return_periods_and_durations,
        timestep_min,
        tables,
        "cumulative",
        path,
        decimals=4,
        allow_short_duration_extrapolation=allow_short_duration_extrapolation,
    )


def write_incremental_intensity(
    return_periods_and_durations,
    timestep_min,
    tables,
    path,
    allow_short_duration_extrapolation: bool = False,
) -> int:
    return _write_wide(
        return_periods_and_durations,
        timestep_min,
        tables,
        "intensity_mmhr",
        path,
        decimals=3,
        allow_short_duration_extrapolation=allow_short_duration_extrapolation,
    )
