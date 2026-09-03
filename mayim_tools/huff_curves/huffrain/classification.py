"""
Quartile classification (paper Section 3.9, formalizing Section 1.2).

Primary/default rule (Section 3.9's formal statement): classify by the
quartile CONTAINING THE MAXIMUM SINGLE INTERVAL DEPTH - i.e. find the
single native-resolution reading with the largest depth, and assign
the event to whichever quarter of the event's duration that reading
falls in. This is the paper's recommended default specifically because
it doesn't imply sub-interval timing precision the data doesn't support.

Also reported (Section 1.2's original formulation, kept for
comparison): the classic method of summing rainfall separately within
each quartile time-window and taking the largest sum. Both are stored
on the event so the discrepancy is visible when they disagree, rather
than silently picking one.

Ties are handled deterministically per the paper: an event tied
between quartiles gets quartile=None, quartile_tied=True, and is
excluded from quartile-specific curve sets by default (screening
already gives users a lever to include/exclude via quality_mode; tied
events are always excluded from PERCENTILE curves regardless, since
"which curve does it belong to" has no defensible single answer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import RainfallEvent


def classify_quartile(event: RainfallEvent, intervals: "pd.DataFrame") -> RainfallEvent:
    if event.contains_missing or event.wet_duration_s <= 0:
        event.quartile = None
        event.quartile_tied = False
        event.quartile_depths = None
        return event

    sub = intervals.loc[list(event.interval_positions)]
    wet_only = sub[sub["depth_mm"].notna() & (sub["depth_mm"] > 0)]
    if wet_only.empty:
        event.quartile = None
        event.quartile_tied = False
        event.quartile_depths = None
        return event

    t0 = event.start
    D = event.wet_duration_s
    # midpoint of each interval, relative to event start, as a fraction of D
    midpoint_frac = ((wet_only["timestamp_start"] + (wet_only["timestamp_end"] - wet_only["timestamp_start"]) / 2)
                      - t0).dt.total_seconds() / D
    midpoint_frac = midpoint_frac.clip(0.0, 0.999999)  # keep the last instant inside quartile 4, not a 5th bucket
    quartile_bucket = np.floor(midpoint_frac * 4).astype(int) + 1  # 1..4

    depths = wet_only["depth_mm"].to_numpy()
    buckets = quartile_bucket.to_numpy()

    # Secondary/classic method: sum depth per quartile bucket.
    quartile_sums = tuple(float(depths[buckets == j].sum()) for j in (1, 2, 3, 4))
    event.quartile_depths = quartile_sums

    # Primary/default method: quartile of the single maximum-depth interval.
    max_depth = depths.max()
    tied_mask = np.isclose(depths, max_depth)
    tied_buckets = set(buckets[tied_mask].tolist())
    if len(tied_buckets) > 1:
        event.quartile = None
        event.quartile_tied = True
    else:
        event.quartile = int(next(iter(tied_buckets)))
        event.quartile_tied = False

    return event
