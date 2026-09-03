"""
Core data objects for the Huff curve pipeline.

Named to match the architecture proposed in the source methodology
paper (Section 5.2: RainfallInterval, RainfallEvent, EventCurve,
HuffAnalysis, HuffCurveSet), scoped down to what v1 actually
implements. Fields not yet populated by v1 (quality_score components,
cluster membership, uncertainty bounds) are present but may be None -
this keeps the shape stable for later phases (bootstrap uncertainty,
stratification, clustering) without a breaking schema change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RainfallInterval:
    """One row of the canonical internal representation (paper Section
    3.3). timestamp_start/timestamp_end bound the interval the depth
    is assigned to; duration_s is the ACTUAL interval length (not
    assumed uniform), which is what lets irregular/gappy data be
    handled correctly rather than silently assumed regular."""
    timestamp_start: "pd.Timestamp"
    timestamp_end: "pd.Timestamp"
    depth_mm: float  # NaN = missing, never silently coerced to 0
    duration_s: float
    quality_code: str  # 'valid' | 'zero' | 'missing' | 'suspect' | 'invalid'
    source_row: int


@dataclass
class RainfallEvent:
    """One delineated storm event (paper Section 3.5)."""
    event_id: int
    start: "pd.Timestamp"
    end: "pd.Timestamp"
    wet_duration_s: float
    analysis_duration_s: float
    total_depth_mm: float
    max_interval_depth_mm: float
    n_wet_intervals: int
    n_dry_gaps: int
    mit_used_s: float
    source_rows: tuple[int, ...]
    contains_missing: bool
    interval_positions: tuple[int, ...] = ()
    quartile: int | None = None  # 1-4, or None if tied/ambiguous
    quartile_tied: bool = False
    quartile_depths: tuple[float, float, float, float] | None = None
    excluded: bool = False
    exclusion_reason: str | None = None


@dataclass
class EventCurve:
    """Dimensionless normalized curve for one retained event (paper
    Section 3.7-3.8): x = elapsed time / duration, y = cumulative
    depth / total depth, resampled onto a common grid."""
    event_id: int
    quartile: int | None
    x_raw: tuple[float, ...]
    y_raw: tuple[float, ...]
    x_grid: tuple[float, ...]
    y_grid: tuple[float, ...]
    n_eff: float  # effective observation count = duration / native interval
    resolution_warning: bool  # True if the grid is finer than the data supports


@dataclass
class HuffCurveSet:
    """Percentile summary for one quartile class (paper Section 3.10)."""
    quartile: int
    x_grid: tuple[float, ...]
    percentiles: dict  # {p: tuple[float, ...]} e.g. {0.1: (...), 0.5: (...)}
    n_events: int
    mean_curve: tuple[float, ...]
    min_curve: tuple[float, ...]
    max_curve: tuple[float, ...]
    insufficient_sample: bool
    min_sample_threshold: int


@dataclass
class HuffAnalysis:
    """Top-level result of the pipeline."""
    events: list  # list[RainfallEvent]
    curves: list  # list[EventCurve]
    curve_sets: list  # list[HuffCurveSet], one per quartile (+ optional 'tied'/'all')
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
