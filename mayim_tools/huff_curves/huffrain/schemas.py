"""
Core data objects for the Huff curve pipeline.

These dataclasses are deliberately independent of QGIS. They represent the
canonical rainfall intervals, delineated rainfall events, normalised event
curves and percentile Huff-curve analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RainfallInterval:
    """
    One canonical rainfall interval.

    Timestamp semantics are normalised before event delineation. Missing
    rainfall is represented by NaN and is never silently converted to zero.
    """

    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    depth_mm: float
    duration_s: float
    quality_code: str = "valid"
    source_row: int = -1

    @property
    def contains_missing(self) -> bool:
        """Return True when the interval has no valid rainfall depth."""
        return pd.isna(self.depth_mm)


@dataclass
class RainfallEvent:
    """
    One delineated storm event.

    Events remain available for inventory reporting even when they are
    excluded from curve generation.
    """

    event_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    wet_duration_s: float
    analysis_duration_s: float
    total_depth_mm: float
    max_interval_depth_mm: float
    n_wet_intervals: int
    n_dry_gaps: int
    source_rows: tuple[int, ...] = field(default_factory=tuple)

    contains_missing: bool = False
    interval_positions: tuple[int, ...] = field(default_factory=tuple)

    quartile: int | None = None
    quartile_tied: bool = False
    quartile_depths: tuple[float, ...] | None = None

    excluded: bool = False
    exclusion_reason: str | None = None


@dataclass
class EventCurve:
    """
    Dimensionless normalised curve for one retained rainfall event.

    x and y values are represented on the common normalised grid, where:

        x = elapsed time / event duration
        y = cumulative rainfall depth / total event depth
    """

    event_id: int
    quartile: int
    x: tuple[float, ...]
    y: tuple[float, ...]
    effective_observation_count: int
    resolution_warning: bool = False


@dataclass
class HuffCurveSet:
    """
    Percentile summary for one Huff quartile.

    Percentile keys are normally 10, 25, 50, 75 and 90. The values are
    dimensionless cumulative-depth curves evaluated on the common x-grid.
    """

    quartile: int
    x_grid: tuple[float, ...]
    percentiles: dict[int, tuple[float, ...]] = field(default_factory=dict)
    mean_curve: tuple[float, ...] = field(default_factory=tuple)
    min_curve: tuple[float, ...] = field(default_factory=tuple)
    max_curve: tuple[float, ...] = field(default_factory=tuple)
    n_events: int = 0
    insufficient_sample: bool = False
    minimum_sample_threshold: int = 5


@dataclass
class HuffAnalysis:
    """
    Top-level result returned by the Huff-curve analysis pipeline.
    """

    events: list[RainfallEvent] = field(default_factory=list)
    curves: list[EventCurve] = field(default_factory=list)
    curve_sets: list[HuffCurveSet] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
