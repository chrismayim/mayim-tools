"""
IETD-based independent storm event catalogue and per-duration best
window extraction, shared by the rainfall tools. Moved from
design_storm_ensembles/core.py (behaviour unchanged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_IETD_HOURS = 6.0
DEFAULT_MIN_EVENT_DEPTH_MM = 2.0


def extract_events(
    series, ietd_hours=DEFAULT_IETD_HOURS, min_depth_mm=DEFAULT_MIN_EVENT_DEPTH_MM
):
    ietd = pd.Timedelta(hours=ietd_hours)
    wet_mask = series > 1e-6
    if not wet_mask.any():
        return []

    wet_times = pd.Series(series.index[wet_mask])
    gaps = wet_times.diff()
    new_event = gaps.isna() | (gaps > ietd)
    event_id = new_event.cumsum()

    events = []
    for eid, grp in wet_times.groupby(event_id):
        start, end = grp.iloc[0], grp.iloc[-1]
        window = series.loc[start:end]
        depth = float(window.sum())
        if depth < min_depth_mm:
            continue
        events.append(
            {
                "event_id": int(eid),
                "start": start,
                "end": end,
                "depth_mm": depth,
                "peak_time": window.idxmax(),
                "peak_value": float(window.max()),
            }
        )
    return events


# ---------------------------------------------------------------------------
# 4. Per-event, per-duration sub-window extraction
# ---------------------------------------------------------------------------


def extract_duration_window(series, native_minutes, event, duration_minutes):
    """Find the highest-depth duration_minutes-long window in the
    vicinity of this event's peak. Search range extends one full
    target duration either side of the peak, so the window always
    contains the peak."""
    n = max(1, round(duration_minutes / native_minutes))
    peak_time = event["peak_time"]
    search_start = peak_time - pd.Timedelta(minutes=duration_minutes)
    search_end = peak_time + pd.Timedelta(minutes=duration_minutes)
    segment = series.loc[search_start:search_end]
    if segment.empty:
        return None

    vals = segment.values.astype(float)
    if len(vals) <= n:
        window_vals = vals
        window_index = segment.index
    else:
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        sums = csum[n:] - csum[:-n]
        best_start = int(np.argmax(sums))
        window_vals = vals[best_start : best_start + n]
        window_index = segment.index[best_start : best_start + n]

    if len(window_vals) == 0:
        return None

    return {
        "event_id": event["event_id"],
        "duration_minutes": duration_minutes,
        "start": window_index[0],
        "end": window_index[-1],
        "depth_mm": float(window_vals.sum()),
        "values": window_vals,
    }


def dedupe_windows(windows):
    """Collapse overlapping duration windows (from adjacent events
    mapping to the same physical storm period) down to the
    highest-depth window in each overlapping cluster."""
    if not windows:
        return []
    ws = sorted(windows, key=lambda w: w["start"])
    out = []
    cluster = [ws[0]]
    cluster_end = ws[0]["end"]
    for w in ws[1:]:
        if w["start"] <= cluster_end:
            cluster.append(w)
            cluster_end = max(cluster_end, w["end"])
        else:
            out.append(max(cluster, key=lambda x: x["depth_mm"]))
            cluster = [w]
            cluster_end = w["end"]
    out.append(max(cluster, key=lambda x: x["depth_mm"]))
    return out
