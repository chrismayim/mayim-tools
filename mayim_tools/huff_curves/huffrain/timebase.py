"""
Canonical interval representation and MIT (minimum inter-event time)
recommendation (paper Section 3.3 and Section 2.2).
"""

from __future__ import annotations

import pandas as pd

# Candidate MIT values, expressed relative to the nominal interval where
# noted (paper Section 2.2: MIT in {2dt, 4dt, 6h, 12h, 24h, 48h, 72h}).
_FIXED_MIT_HOURS = (6, 12, 24, 48, 72)


def normalize_interval_representation(
    parsed: "pd.DataFrame",
    timestamp_semantics: str = "interval_end",
) -> tuple["pd.DataFrame", float]:
    """Builds the canonical per-interval table (paper Section 3.3):
    timestamp_start, timestamp_end, depth_mm, duration_s, quality_code,
    source_row. Returns (intervals_df, nominal_interval_s).

    timestamp_semantics: 'interval_end' (default - the common convention
    for gauge data: the value at 13:00 is the depth accumulated from
    12:00 to 13:00) or 'interval_start'. This is explicit rather than
    assumed, per the paper's requirement (Section 3.1) that timestamp
    semantics never be silently assumed.

    Actual (not assumed-uniform) interval duration is preserved, so
    irregular spacing and gaps are handled correctly downstream rather
    than masked by a fixed-step assumption.
    """
    df = parsed.copy().reset_index(drop=True)

    diffs_s = df["timestamp"].diff().dt.total_seconds()
    # Nominal interval: mode of the positive diffs, falling back to median
    # if there's no clear mode (robust to a handful of gaps/duplicates).
    positive_diffs = diffs_s.dropna()
    positive_diffs = positive_diffs[positive_diffs > 0]
    if positive_diffs.empty:
        raise ValueError("Could not determine a nominal time interval - fewer than 2 valid timestamps.")
    mode_vals = positive_diffs.mode()
    nominal_interval_s = float(mode_vals.iloc[0]) if not mode_vals.empty else float(positive_diffs.median())

    if timestamp_semantics == "interval_end":
        timestamp_end = df["timestamp"]
        # First interval's start is inferred as end - nominal interval,
        # since there's no preceding timestamp to derive it from.
        timestamp_start = df["timestamp"].shift(1)
        timestamp_start.iloc[0] = df["timestamp"].iloc[0] - pd.Timedelta(seconds=nominal_interval_s)
    elif timestamp_semantics == "interval_start":
        timestamp_start = df["timestamp"]
        timestamp_end = df["timestamp"].shift(-1)
        timestamp_end.iloc[-1] = df["timestamp"].iloc[-1] + pd.Timedelta(seconds=nominal_interval_s)
    else:
        raise ValueError(f"Unknown timestamp_semantics: {timestamp_semantics!r}")

    duration_s = (timestamp_end - timestamp_start).dt.total_seconds()

    quality_code = pd.Series("valid", index=df.index)
    quality_code[df["depth_mm"].isna()] = "missing"
    quality_code[df["depth_mm"] == 0] = "zero"

    out = pd.DataFrame({
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "depth_mm": df["depth_mm"],
        "duration_s": duration_s,
        "quality_code": quality_code,
        "source_row": df["source_row"],
    })

    return out, nominal_interval_s


def recommend_mit(intervals: "pd.DataFrame", nominal_interval_s: float, wet_threshold_mm: float) -> dict:
    """Scans the MIT candidate set and reports how event count and mean
    event duration respond (paper Section 2.2) - a lightweight version
    of the paper's fuller MIT-stability diagnostic (serial-dependence
    and inter-arrival CV tests are noted as future work, not
    implemented in v1). Returns a dict with a 'candidates' table and a
    'recommended_s' value (the candidate closest to the paper's most
    commonly cited default, 6 hours, that is still >= 2x the nominal
    interval so it isn't finer than the data can resolve)."""
    from .events import delineate_events  # local import - avoids a circular import at module load time

    candidates_s = sorted(set(
        [2 * nominal_interval_s, 4 * nominal_interval_s] + [h * 3600 for h in _FIXED_MIT_HOURS]
    ))
    rows = []
    for mit_s in candidates_s:
        if mit_s < nominal_interval_s:
            continue
        events = delineate_events(intervals, mit_s=mit_s, wet_threshold_mm=wet_threshold_mm)
        n = len(events)
        mean_dur_h = (sum(e.wet_duration_s for e in events) / n / 3600) if n else 0.0
        rows.append({"mit_hours": round(mit_s / 3600, 3), "n_events": n, "mean_event_duration_h": round(mean_dur_h, 2)})

    recommended_s = 6 * 3600.0
    if recommended_s < 2 * nominal_interval_s:
        # 6h default would be finer than the data can resolve at 2x
        # nominal spacing - fall back to the smallest sensible candidate.
        recommended_s = 2 * nominal_interval_s

    return {"candidates": rows, "recommended_s": recommended_s, "nominal_interval_s": nominal_interval_s}
