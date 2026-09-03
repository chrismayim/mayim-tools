"""
CSV table exporters (paper Section 7.1, scoped to what v1 computes).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .schemas import HuffAnalysis


def write_event_inventory(analysis: HuffAnalysis, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EventID", "Start", "End", "WetDurationHours", "TotalDepthMM",
                    "MaxIntervalDepthMM", "NWetIntervals", "NDryGaps", "MITHoursUsed",
                    "Quartile", "QuartileTied", "Q1DepthMM", "Q2DepthMM", "Q3DepthMM", "Q4DepthMM",
                    "ContainsMissing", "Excluded", "ExclusionReason"])
        for e in analysis.events:
            qd = e.quartile_depths or (None, None, None, None)
            w.writerow([
                e.event_id, e.start, e.end, round(e.wet_duration_s / 3600, 3), round(e.total_depth_mm, 3),
                round(e.max_interval_depth_mm, 3), e.n_wet_intervals, e.n_dry_gaps, round(e.mit_used_s / 3600, 3),
                e.quartile if e.quartile is not None else "", e.quartile_tied,
                *(round(v, 3) if v is not None else "" for v in qd),
                e.contains_missing, e.excluded, e.exclusion_reason or "",
            ])


def write_event_curves(analysis: HuffAnalysis, path: str | Path) -> None:
    """Long format: one row per event per grid point - the 'normalized
    event curves' table (paper Section 7.1)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EventID", "Quartile", "X", "Y", "NEff", "ResolutionWarning"])
        for c in analysis.curves:
            for x, y in zip(c.x_grid, c.y_grid):
                w.writerow([c.event_id, c.quartile if c.quartile is not None else "",
                            round(x, 4), round(y, 5), round(c.n_eff, 2), c.resolution_warning])


def write_huff_curves(analysis: HuffAnalysis, path: str | Path) -> None:
    """Wide format: one row per (quartile, x), one column per
    percentile - the 'percentile Huff curves' table (paper Section 7.1)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if not analysis.curve_sets:
            w.writerow(["No quartile curve sets were produced - see diagnostics/warnings."])
            return
        percentiles = sorted(analysis.curve_sets[0].percentiles.keys())
        header = ["Quartile", "X", "NEvents", "InsufficientSample", "Mean", "Min", "Max"]
        header += [f"P{int(p*100)}" for p in percentiles]
        w.writerow(header)
        for cs in analysis.curve_sets:
            for i, x in enumerate(cs.x_grid):
                row = [cs.quartile, round(x, 4), cs.n_events, cs.insufficient_sample,
                       round(cs.mean_curve[i], 5), round(cs.min_curve[i], 5), round(cs.max_curve[i], 5)]
                row += [round(cs.percentiles[p][i], 5) for p in percentiles]
                w.writerow(row)


def write_metadata(analysis: HuffAnalysis, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Key", "Value"])
        for k, v in analysis.metadata.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["Warnings"])
        for warn in analysis.warnings:
            w.writerow([warn])
