"""
Event delineation and screening (paper Section 3.5-3.6).

Delineation algorithm (paper's pseudocode):
    w_i = 1 if r_i > wet_threshold, else 0
    merge adjacent wet spells separated by a dry gap < MIT
    close an event only after a dry gap >= MIT

Missing (NaN) intervals are treated as neither wet nor dry - the dry-
gap clock does not advance during missing data (so a data outage can't
masquerade as a genuine dry period and artificially close/split a real
storm), but an event containing missing intervals is flagged
(`contains_missing`) so screening can exclude it under strict quality
mode. This is a conservative default consistent with the paper's core
instruction: missing rainfall must never be silently treated as zero
or, by extension, as "definitely dry".
"""

from __future__ import annotations

import pandas as pd

from .schemas import RainfallEvent


def delineate_events(intervals: "pd.DataFrame", mit_s: float, wet_threshold_mm: float) -> list[RainfallEvent]:
    events: list[RainfallEvent] = []

    in_event = False
    event_rows: list[int] = []
    dry_accum_s = 0.0
    contains_missing = False
    dry_run_active = False
    n_dry_gaps = 0
    event_id = 0

    def _finalize(rows: list[int]) -> RainfallEvent:
        nonlocal event_id
        sub = intervals.loc[rows]
        wet_rows = sub[sub["quality_code"] != "missing"]
        if wet_rows.empty:
            return None  # an event made entirely of missing intervals is not a real event
        wet_only = wet_rows[wet_rows["depth_mm"] > wet_threshold_mm]
        if wet_only.empty:
            return None
        start = wet_only["timestamp_start"].iloc[0]
        end = wet_only["timestamp_end"].iloc[-1]
        wet_duration_s = (end - start).total_seconds()
        total_depth = float(wet_rows["depth_mm"].sum(skipna=True))
        max_depth = float(wet_rows["depth_mm"].max(skipna=True))
        event_id += 1
        return RainfallEvent(
            event_id=event_id,
            start=start, end=end,
            wet_duration_s=wet_duration_s,
            analysis_duration_s=wet_duration_s,  # v1: no dry padding, see module docstring
            total_depth_mm=total_depth,
            max_interval_depth_mm=max_depth,
            n_wet_intervals=int(len(wet_only)),
            n_dry_gaps=n_dry_gaps,
            mit_used_s=mit_s,
            source_rows=tuple(int(r) for r in sub["source_row"]),
            contains_missing=contains_missing,
            interval_positions=tuple(rows),
        )

    for i, row in intervals.iterrows():
        depth = row["depth_mm"]
        dur = row["duration_s"] if pd.notna(row["duration_s"]) else 0.0

        if pd.isna(depth):
            if in_event:
                event_rows.append(i)
                contains_missing = True
            dry_run_active = False  # missing data doesn't count as a dry run for gap counting
            continue

        is_wet = depth > wet_threshold_mm
        if is_wet:
            if not in_event:
                in_event = True
                event_rows = []
                contains_missing = False
                n_dry_gaps = 0
            event_rows.append(i)
            dry_accum_s = 0.0
            dry_run_active = False
        else:
            if in_event:
                dry_accum_s += dur
                if dry_accum_s >= mit_s:
                    ev = _finalize(event_rows)
                    if ev is not None:
                        events.append(ev)
                    in_event = False
                    event_rows = []
                    dry_accum_s = 0.0
                    dry_run_active = False
                else:
                    event_rows.append(i)
                    if not dry_run_active:
                        n_dry_gaps += 1
                        dry_run_active = True

    if in_event and event_rows:
        ev = _finalize(event_rows)
        if ev is not None:
            events.append(ev)

    return events


def screen_events(
    events: list[RainfallEvent],
    depth_threshold_mm: float | None = None,
    min_duration_s: float | None = None,
    quality_mode: str = "strict",
) -> list[RainfallEvent]:
    """Flags (and, in 'strict' mode, excludes) events per paper Section
    3.6. Always returns the full list with `excluded`/`exclusion_reason`
    set - callers decide whether to filter, so nothing is silently
    dropped without a recorded reason."""
    out = []
    for ev in events:
        reasons = []
        if quality_mode == "strict" and ev.contains_missing:
            reasons.append("contains missing interval(s) within the event window")
        if depth_threshold_mm is not None and ev.total_depth_mm < depth_threshold_mm:
            reasons.append(f"total depth {ev.total_depth_mm:.2f}mm below threshold {depth_threshold_mm}mm")
        if min_duration_s is not None and ev.wet_duration_s < min_duration_s:
            reasons.append(f"duration {ev.wet_duration_s/3600:.2f}h below minimum {min_duration_s/3600:.2f}h")

        ev.excluded = bool(reasons)
        ev.exclusion_reason = "; ".join(reasons) if reasons else None
        out.append(ev)
    return out
