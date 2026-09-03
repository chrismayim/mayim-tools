"""
Top-level pipeline orchestrator - mirrors the paper's Section 5.3
build_huff_curves() pseudocode, scoped to what v1 implements.

Deferred to later phases (present in the paper, not yet built):
bootstrap uncertainty (Section 3.11), stratification by duration/
season/decade (Section 3.12), duration-magnitude dependence modelling
(Section 3.13), non-stationarity/trend analysis (Section 3.14),
regional pooling (Section 4.4), clustering-based storm taxonomy
(Section 4.3), design hyetograph generation (Section 4.6), and
comparison against SCS/Chicago/alternating-block/other methods
(Section 4.7). See MISSING_FEATURES.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .classification import classify_quartile
from .events import delineate_events, screen_events
from .normalization import normalize_event, resample_to_grid
from .schemas import HuffAnalysis
from .statistics import DEFAULT_PERCENTILES, MIN_SAMPLE_DEFAULT, summarize_curves
from .timebase import normalize_interval_representation, recommend_mit
from .validation import parse_and_validate


def build_huff_curves(
    df: "pd.DataFrame",
    timestamp_col: str,
    depth_col: str,
    timestamp_format: str | None = None,
    timestamp_semantics: str = "interval_end",
    mit_hours: float | None = None,
    wet_threshold_mm: float = 0.0,
    event_depth_threshold_mm: float | None = None,
    min_duration_hours: float | None = None,
    quality_mode: str = "strict",
    normalized_step: float = 0.05,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    min_sample: int = MIN_SAMPLE_DEFAULT,
) -> HuffAnalysis:
    """Runs the full v1 pipeline. If mit_hours is None, a recommended
    value is computed (paper Section 2.2) and used, with the scan
    results recorded in diagnostics so the choice is auditable rather
    than silently imposed (paper item 40: "recommend, but do not
    silently impose, MIT" - v1's compromise, since a Processing
    algorithm runs to completion in one pass rather than interactively,
    is to always record what was scanned and what was picked)."""
    warnings: list[str] = []
    metadata: dict = {
        "timestamp_col": timestamp_col, "depth_col": depth_col,
        "timestamp_semantics": timestamp_semantics, "wet_threshold_mm": wet_threshold_mm,
        "event_depth_threshold_mm": event_depth_threshold_mm, "quality_mode": quality_mode,
        "normalized_step": normalized_step, "percentiles": percentiles,
        "quantile_method": "median_unbiased (Hyndman-Fan type 8)",
        "processing_timestamp": str(pd.Timestamp.now()),
    }

    parsed, val_diag = parse_and_validate(df, timestamp_col, depth_col, timestamp_format)
    warnings += val_diag["warnings"]
    metadata["n_rows_in"] = val_diag["n_rows_in"]
    metadata["n_rows_valid"] = val_diag["n_rows_valid"]
    metadata["timestamp_range"] = val_diag["timestamp_range"]

    if len(parsed) < 2:
        raise ValueError("Fewer than 2 valid rows after parsing - cannot determine a time interval.")

    intervals, nominal_interval_s = normalize_interval_representation(parsed, timestamp_semantics)
    metadata["nominal_interval_s"] = nominal_interval_s
    metadata["nominal_interval_label"] = _format_duration(nominal_interval_s)

    mit_diag = recommend_mit(intervals, nominal_interval_s, wet_threshold_mm)
    if mit_hours is None:
        mit_s = mit_diag["recommended_s"]
        warnings.append(
            f"MIT not specified - using recommended default {_format_duration(mit_s)}. "
            "See the MIT sensitivity table in diagnostics; specify mit_hours explicitly to override."
        )
    else:
        mit_s = mit_hours * 3600.0
    metadata["mit_hours_used"] = round(mit_s / 3600, 3)

    raw_events = delineate_events(intervals, mit_s=mit_s, wet_threshold_mm=wet_threshold_mm)
    if not raw_events:
        warnings.append("No events were delineated - check the wet threshold and input data.")

    min_duration_s = min_duration_hours * 3600 if min_duration_hours is not None else None
    events = screen_events(raw_events, depth_threshold_mm=event_depth_threshold_mm,
                            min_duration_s=min_duration_s, quality_mode=quality_mode)

    n_excluded = sum(1 for e in events if e.excluded)
    if n_excluded:
        warnings.append(f"{n_excluded} of {len(events)} events excluded by screening (see event inventory).")

    for e in events:
        classify_quartile(e, intervals)

    n_missing_curve = sum(1 for e in events if not e.excluded and e.contains_missing)
    if n_missing_curve:
        warnings.append(
            f"{n_missing_curve} retained event(s) contain missing interior data and were excluded "
            "from curve normalization specifically (kept in the event inventory for auditability)."
        )

    grid = np.arange(0.0, 1.0 + normalized_step / 2, normalized_step)
    grid = np.clip(grid, 0.0, 1.0)

    curves = []
    n_resolution_warning = 0
    for e in events:
        if e.excluded:
            continue
        raw_curve = normalize_event(e, intervals)
        if raw_curve is None:
            continue
        curve = resample_to_grid(raw_curve, grid)
        if curve.resolution_warning:
            n_resolution_warning += 1
        curves.append(curve)

    if n_resolution_warning:
        warnings.append(
            f"{n_resolution_warning} event(s) flagged with a resolution warning - the normalized "
            f"grid (step={normalized_step}) is finer than the event's native data can support."
        )

    tied_count = sum(1 for c in curves if c.quartile is None)
    if tied_count:
        warnings.append(f"{tied_count} event curve(s) have a tied/ambiguous quartile and are excluded from percentile curve sets.")

    curve_sets = summarize_curves(curves, grid, percentiles=percentiles, min_sample=min_sample)
    for cs in curve_sets:
        if cs.insufficient_sample:
            warnings.append(
                f"Quartile {cs.quartile}: only {cs.n_events} event(s), below the minimum sample "
                f"threshold ({cs.min_sample_threshold}). Treat this quartile's percentile curve with caution."
            )

    diagnostics = {
        "mit_scan": mit_diag["candidates"],
        "mit_recommended_hours": round(mit_diag["recommended_s"] / 3600, 3),
        "n_events_raw": len(raw_events),
        "n_events_retained": len(events) - n_excluded,
        "n_events_excluded": n_excluded,
        "n_curves": len(curves),
        "n_resolution_warnings": n_resolution_warning,
    }

    return HuffAnalysis(
        events=events, curves=curves, curve_sets=curve_sets,
        diagnostics=diagnostics, warnings=warnings, metadata=metadata,
    )


def _format_duration(seconds: float) -> str:
    hours = seconds / 3600
    if hours == int(hours):
        return f"{int(hours)}h"
    return f"{hours:.2f}h"
