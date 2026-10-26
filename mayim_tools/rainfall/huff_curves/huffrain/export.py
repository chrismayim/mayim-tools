"""
CSV table exporters (paper Section 7.1, scoped to what v1 computes).

Mayim output standard: UTF-8 with a byte-order mark (so Excel opens the
files cleanly); the event inventory, event curves and Huff curves CSVs
start with a column key as '# column: meaning' lines (read with pandas
``comment='#'``; the Prepare HEC-HMS / SWMM project readers skip them).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .schemas import HuffAnalysis

CSV_ENCODING = "utf-8-sig"
KEY_PREFIX = "# "

_QUARTILE = (
    "Huff quartile 1-4 - the quarter of the event's duration that holds its "
    "largest reading"
)

EVENT_INVENTORY_KEY = {
    "EventID": "event number",
    "Start": "start of the event (time as in the input)",
    "End": "end of the event (time as in the input)",
    "WetDurationHours": "wet duration of the event (h)",
    "TotalDepthMM": "total event depth (mm)",
    "MaxIntervalDepthMM": "largest single-interval depth in the event (mm)",
    "NWetIntervals": "number of wet intervals",
    "NDryGaps": "number of dry gaps inside the event",
    "MITHoursUsed": "minimum inter-event time used to separate events (h)",
    "Quartile": _QUARTILE + " (blank if tied)",
    "QuartileTied": "True if two quarters share the largest reading",
    "Q1DepthMM": "depth in the first quarter of the event (mm)",
    "Q2DepthMM": "depth in the second quarter of the event (mm)",
    "Q3DepthMM": "depth in the third quarter of the event (mm)",
    "Q4DepthMM": "depth in the fourth quarter of the event (mm)",
    "ContainsMissing": "True if the event has missing data inside it",
    "Excluded": "True if the event is left out of the curves",
    "ExclusionReason": "why the event was left out",
}

EVENT_CURVES_KEY = {
    "EventID": "event number (see the event inventory)",
    "Quartile": _QUARTILE,
    "X": "fraction of the event duration elapsed (0-1)",
    "Y": "cumulative fraction of the event depth (0-1)",
    "NEff": "effective number of observations (event duration / data time step)",
    "ResolutionWarning": "True if the time grid is finer than the data supports",
}

HUFF_CURVES_KEY = {
    "Quartile": _QUARTILE,
    "X": "fraction of the storm duration elapsed (0-1)",
    "NEvents": "number of events in the quartile",
    "InsufficientSample": "True if fewer events than the minimum sample",
    "Mean": "mean cumulative fraction of storm depth (0-1)",
    "Min": "smallest cumulative fraction of storm depth (0-1)",
    "Max": "largest cumulative fraction of storm depth (0-1)",
    "P10 ...": (
        "percentile curves P10 / P25 / P50 / P75 / P90 of the cumulative "
        "fraction of storm depth (0-1; Hyndman-Fan type 8)"
    ),
}


def _write_key(f, key: dict, columns: list[str]) -> None:
    f.write(f"{KEY_PREFIX}Column key\n")
    for col, meaning in key.items():
        if col in columns or col.endswith("..."):
            f.write(f"{KEY_PREFIX}{col}: {meaning}\n")


def write_event_inventory(analysis: HuffAnalysis, path: str | Path) -> None:
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, EVENT_INVENTORY_KEY, list(EVENT_INVENTORY_KEY))
        w = csv.writer(f)
        w.writerow(
            [
                "EventID",
                "Start",
                "End",
                "WetDurationHours",
                "TotalDepthMM",
                "MaxIntervalDepthMM",
                "NWetIntervals",
                "NDryGaps",
                "MITHoursUsed",
                "Quartile",
                "QuartileTied",
                "Q1DepthMM",
                "Q2DepthMM",
                "Q3DepthMM",
                "Q4DepthMM",
                "ContainsMissing",
                "Excluded",
                "ExclusionReason",
            ]
        )
        for e in analysis.events:
            qd = e.quartile_depths or (None, None, None, None)
            w.writerow(
                [
                    e.event_id,
                    e.start,
                    e.end,
                    round(e.wet_duration_s / 3600, 3),
                    round(e.total_depth_mm, 3),
                    round(e.max_interval_depth_mm, 3),
                    e.n_wet_intervals,
                    e.n_dry_gaps,
                    round(e.mit_used_s / 3600, 3),
                    e.quartile if e.quartile is not None else "",
                    e.quartile_tied,
                    *(round(v, 3) if v is not None else "" for v in qd),
                    e.contains_missing,
                    e.excluded,
                    e.exclusion_reason or "",
                ]
            )


def write_event_curves(analysis: HuffAnalysis, path: str | Path) -> None:
    """Long format: one row per event per grid point - the 'normalized
    event curves' table (paper Section 7.1)."""
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, EVENT_CURVES_KEY, list(EVENT_CURVES_KEY))
        w = csv.writer(f)
        w.writerow(list(EVENT_CURVES_KEY))
        for c in analysis.curves:
            for x, y in zip(c.x_grid, c.y_grid):
                w.writerow(
                    [
                        c.event_id,
                        c.quartile if c.quartile is not None else "",
                        round(x, 4),
                        round(y, 5),
                        round(c.n_eff, 2),
                        c.resolution_warning,
                    ]
                )


def write_huff_curves(analysis: HuffAnalysis, path: str | Path) -> None:
    """Wide format: one row per (quartile, x), one column per
    percentile - the 'percentile Huff curves' table (paper Section 7.1)."""
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.writer(f)
        if not analysis.curve_sets:
            w.writerow(
                ["No quartile curve sets were produced - see diagnostics/warnings."]
            )
            return
        percentiles = sorted(analysis.curve_sets[0].percentiles.keys())
        header = [
            "Quartile",
            "X",
            "NEvents",
            "InsufficientSample",
            "Mean",
            "Min",
            "Max",
        ]
        header += [f"P{int(p*100)}" for p in percentiles]
        _write_key(f, HUFF_CURVES_KEY, header)
        w.writerow(header)
        for cs in analysis.curve_sets:
            for i, x in enumerate(cs.x_grid):
                row = [
                    cs.quartile,
                    round(x, 4),
                    cs.n_events,
                    cs.insufficient_sample,
                    round(cs.mean_curve[i], 5),
                    round(cs.min_curve[i], 5),
                    round(cs.max_curve[i], 5),
                ]
                row += [round(cs.percentiles[p][i], 5) for p in percentiles]
                w.writerow(row)


def write_metadata(analysis: HuffAnalysis, path: str | Path) -> None:
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.writer(f)
        w.writerow(["Key", "Value"])
        for k, v in analysis.metadata.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["Warnings"])
        for warn in analysis.warnings:
            w.writerow([warn])
