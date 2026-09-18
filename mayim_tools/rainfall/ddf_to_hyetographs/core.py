"""
Core logic for the Generalized Alternating Block Method (ABM) design
hyetograph generator - "DDF to Alternating Block Design Hyetographs".

WHAT THIS ACTUALLY IS: an Alternating Block Method implementation with
a configurable peak-position ratio - NOT the classical SCS/NRCS Type
I/IA/II/III design storm distributions. The plugin's original working
name, "SCS Design Storm", was misleading for exactly this reason and
has been replaced.

METHOD: for each return period in the input DDF/IDF table -
1. Parse the depth-duration table (see parse_idf_table - tolerant of
   either the {RT}yr Depth (mm) header format used by Design Rainfall
   and Rainfall Frequency Analysis's recommended-DDF output, or a bare
   AEP-percent header format).
2. Interpolate a continuous cumulative-depth-vs-duration curve,
   piecewise log-log, exact at every tabulated point - no
   extrapolation beyond the table's own duration range.
3. Divide the target storm duration into N equal-width time steps and
   compute each step's INCREMENTAL depth from the interpolated
   cumulative curve.
4. Reorder those incremental blocks via the alternating block method,
   at each of five peak-position ratios (fraction of the storm
   duration occurring before the peak block: 25/33/50/66/75%). Pure
   reordering - total depth is conserved by construction, never
   altered.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

PEAK_RATIOS = (0.25, 0.33, 0.50, 0.66, 0.75)

_RETURN_PERIOD_COL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*yr\s*Depth", re.IGNORECASE)
_AEP_PERCENT_COL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")
_DURATION_WITH_UNIT_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(min|h|hr|hour)s?\s*$", re.IGNORECASE
)


def parse_idf_table(df: pd.DataFrame, duration_col: str | None = None) -> dict:
    """Returns {return_period_years: [(duration_min, depth_mm), ...]},
    sorted by duration, tolerant of two header conventions:
    - '{RT}yr Depth (mm)' (Design Rainfall / Rainfall Frequency
      Analysis's recommended-DDF output format)
    - bare AEP-percent columns (e.g. '50%', '20%', '1%'), converted to
      return period via T = 100 / AEP%.

    duration_col: explicit column name, or None to auto-detect
    ('Duration', falling back to 'Time' and other common variants).
    """
    if duration_col is None:
        for candidate in ("Duration", "Time", "duration", "time"):
            if candidate in df.columns:
                duration_col = candidate
                break
        else:
            raise ValueError(
                f"Could not find a duration column. Available: {list(df.columns)}"
            )
    elif duration_col not in df.columns:
        raise ValueError(
            f"Duration column {duration_col!r} not found. Available: {list(df.columns)}"
        )

    durations_min = df[duration_col].apply(_parse_duration_label)

    table: dict = {}
    for col in df.columns:
        if col == duration_col:
            continue
        rt = None
        m = _RETURN_PERIOD_COL_RE.match(str(col))
        if m:
            rt = float(m.group(1))
        else:
            m = _AEP_PERCENT_COL_RE.match(str(col))
            if m:
                aep_percent = float(m.group(1))
                if aep_percent > 0:
                    rt = round(100.0 / aep_percent, 4)
        if rt is None:
            continue

        pairs = []
        for dur, depth in zip(durations_min, df[col], strict=True):
            if pd.isna(dur) or pd.isna(depth):
                continue
            pairs.append((float(dur), float(depth)))
        pairs.sort(key=lambda p: p[0])
        if pairs:
            table[rt] = pairs

    if not table:
        raise ValueError(
            f"No return-period columns recognized. Available columns: {list(df.columns)}"
        )
    return table


def _parse_duration_label(value) -> float:
    """Parses a duration label as minutes - either a bare number
    (assumed minutes) or a number with an explicit unit (min/h/hr)."""
    if pd.isna(value):
        return float("nan")
    s = str(value).strip()
    m = _DURATION_WITH_UNIT_RE.match(s)
    if m:
        num = float(m.group(1))
        unit = m.group(2).lower()
        return num * 60 if unit in ("h", "hr", "hour") else num
    try:
        return float(s)
    except ValueError:
        return float("nan")


def extrapolate_short_duration(duration_min: float, table_pairs: list) -> float:
    """Extends the DDF curve BELOW the table's shortest tabulated
    duration, down toward zero, using a power law anchored at the
    origin: D(t) = D1 * (t/t1)^b, where (t1, D1) is the table's
    shortest tabulated point and b is the log-log slope of the FIRST
    tabulated segment (between the first two points) - i.e. the same
    power-law form already used for interpolation between every other
    pair of tabulated points (see interpolate_cumulative_depth),
    simply extended down to t=0 instead of stopping at t1. This is
    data-driven, not an externally-imposed coefficient: no new
    assumption is introduced beyond "the curve continues to behave
    like its own first segment," which is the most defensible default
    without a specific regional short-duration formula to anchor to.

    Exact at t=t1 (returns D1 there) and D(t) -> 0 as t -> 0, both by
    construction.

    KNOWN LIMITATION, stated plainly rather than left implicit: real
    DDF curves often STEEPEN at very short durations - sub-hourly
    convective bursts are typically more intense, relative to their
    duration, than a curve's longer-duration segments would suggest.
    Using the first tabulated segment's slope to extrapolate down
    therefore risks UNDERESTIMATING short-duration intensity,
    particularly when the shortest tabulated duration is already
    coarse (e.g. an ERA5-derived DDF table with no data below 60 min).
    If a recognized short-duration formula or locally-measured
    sub-hourly data is available, use that instead - this is
    explicitly a fallback for when nothing better exists, not a
    general-purpose substitute for one.
    """
    if len(table_pairs) < 2:
        raise ValueError(
            "Need at least 2 tabulated points to extrapolate a short-duration slope."
        )
    t1, d1 = table_pairs[0]
    t2, d2 = table_pairs[1]
    if duration_min <= 0:
        return 0.0
    if d1 <= 0 or t1 <= 0 or t2 <= 0:
        raise ValueError(
            "Cannot compute a power-law slope with a non-positive duration or depth."
        )
    b = np.log(d2 / d1) / np.log(t2 / t1)
    return float(d1 * (duration_min / t1) ** b)


def interpolate_cumulative_depth(
    duration_min: float,
    table_pairs: list,
    allow_short_duration_extrapolation: bool = False,
) -> float:
    """Piecewise log-log interpolation of cumulative depth at an
    arbitrary duration, exact at every tabulated point. Raises if
    duration_min falls outside the table's own range, UNLESS
    allow_short_duration_extrapolation=True and duration_min is below
    the table's shortest tabulated duration specifically - in that
    one case, falls back to extrapolate_short_duration() rather than
    raising (see that function's docstring for the method and its
    known limitation). Above the table's longest tabulated duration is
    NEVER extrapolated, regardless of this flag - there is no
    equivalent "curve continues toward infinity" assumption that makes
    sense the way "curve continues toward zero at zero duration" does.
    """
    durations = [p[0] for p in table_pairs]
    depths = [p[1] for p in table_pairs]

    if duration_min < durations[0]:
        if allow_short_duration_extrapolation:
            return extrapolate_short_duration(duration_min, table_pairs)
        raise ValueError(
            f"Duration {duration_min:g} min is below the table's shortest tabulated duration "
            f"({durations[0]:g} min) - no extrapolation by default. Enable short-duration "
            f"extrapolation to extend the curve toward zero using the first segment's own "
            f"power-law slope (see core.py's extrapolate_short_duration for the method and "
            f"its known limitation), or supply a table with finer tabulated durations."
        )
    if duration_min > durations[-1]:
        raise ValueError(
            f"Duration {duration_min:g} min is above the table's longest tabulated duration "
            f"({durations[-1]:g} min) - no extrapolation in this direction, ever."
        )

    log_d = np.log(durations)
    log_v = np.log(depths)
    log_target = np.log(duration_min)
    log_result = np.interp(log_target, log_d, log_v)
    return float(np.exp(log_result))


def compute_incremental_blocks(
    target_duration_min: float,
    timestep_min: float,
    table_pairs: list,
    allow_short_duration_extrapolation: bool = False,
) -> list:
    """Divides target_duration_min into equal-width timestep_min
    blocks and returns each block's INCREMENTAL depth (mm), derived
    from the interpolated cumulative-depth curve. The final block may
    be shorter than timestep_min if target_duration_min isn't an exact
    multiple - handled by clipping the last step's end to the target
    duration exactly, not by extrapolating past it.

    allow_short_duration_extrapolation: passed through to
    interpolate_cumulative_depth() - needed whenever timestep_min is
    finer than the table's own shortest tabulated duration (e.g. a
    5-min timestep against an ERA5-derived hourly-only DDF table),
    since every timestep boundary below that shortest tabulated point
    would otherwise raise. See that function's docstring for the
    method and its known limitation.
    """
    n_full_steps = int(target_duration_min // timestep_min)
    remainder = target_duration_min - n_full_steps * timestep_min
    step_ends = [
        min((i + 1) * timestep_min, target_duration_min) for i in range(n_full_steps)
    ]
    if remainder > 1e-9:
        step_ends.append(target_duration_min)

    cumulative = [
        interpolate_cumulative_depth(t, table_pairs, allow_short_duration_extrapolation)
        for t in step_ends
    ]
    incremental = [cumulative[0]] + [
        cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))
    ]
    return incremental


def alternating_block_arrange(incremental_blocks: list, peak_ratio: float) -> list:
    """Reorders incremental blocks via the alternating block method:
    sorts blocks descending by depth, places the largest at the peak
    position (determined by peak_ratio - the fraction of the sequence
    before the peak), then alternates placing the next-largest blocks
    immediately before and after the peak, working outward. Pure
    reordering - sum of the output exactly equals sum of the input."""
    n = len(incremental_blocks)
    sorted_desc = sorted(incremental_blocks, reverse=True)

    peak_index = round(peak_ratio * (n - 1)) if n > 1 else 0
    peak_index = max(0, min(n - 1, peak_index))

    result = [None] * n
    result[peak_index] = sorted_desc[0]

    left = peak_index - 1
    right = peak_index + 1
    toggle_left = True
    for val in sorted_desc[1:]:
        if toggle_left and left >= 0:
            result[left] = val
            left -= 1
        elif right < n:
            result[right] = val
            right += 1
        elif left >= 0:
            result[left] = val
            left -= 1
        toggle_left = not toggle_left

    assert all(
        v is not None for v in result
    ), "alternating block arrangement left a gap"
    return result


def build_hyetograph(
    target_duration_min: float,
    timestep_min: float,
    table_pairs: list,
    allow_short_duration_extrapolation: bool = False,
) -> dict:
    """For one (return period already selected via table_pairs, target
    duration) combination: returns {peak_ratio: {'incremental': [...],
    'cumulative': [...], 'intensity_mmhr': [...], 'times_min': [...]}}
    for every ratio in PEAK_RATIOS."""
    incremental = compute_incremental_blocks(
        target_duration_min,
        timestep_min,
        table_pairs,
        allow_short_duration_extrapolation,
    )
    n = len(incremental)
    times_min = [min((i + 1) * timestep_min, target_duration_min) for i in range(n)]
    step_widths_hr = []
    prev_t = 0.0
    for t in times_min:
        step_widths_hr.append((t - prev_t) / 60.0)
        prev_t = t

    out = {}
    for ratio in PEAK_RATIOS:
        arranged = alternating_block_arrange(incremental, ratio)
        cumulative = list(np.cumsum(arranged))
        intensity = [
            d / w if w > 0 else 0.0
            for d, w in zip(arranged, step_widths_hr, strict=True)
        ]
        out[ratio] = {
            "incremental": arranged,
            "cumulative": cumulative,
            "intensity_mmhr": intensity,
            "times_min": times_min,
        }
    return out
