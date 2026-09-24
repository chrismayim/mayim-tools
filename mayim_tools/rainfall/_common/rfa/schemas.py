"""
Core data objects for the rainfall frequency analysis pipeline.

Terminology matches the user's own request directly: Location (Xi),
Scale (Alpha), Shape (Kappa) - the Hosking L-moments convention for
GEV, which is also what the WRC K5/1060 methodology underlying the
Design Rainfall plugin uses. See distributions.py for the sign-
convention verification against scipy's MLE fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class YearRecord:
    """Completeness bookkeeping for one calendar year (Stage 1.1)."""

    year: int
    n_expected: int
    n_valid: int
    completeness: float  # n_valid / n_expected
    included: bool
    exclusion_reason: str | None = None


@dataclass
class DurationSeries:
    """Annual maximum series for one duration (Stage 1.1)."""

    duration_label: str
    duration_minutes: float
    years: tuple  # tuple[int, ...] - years actually contributing a valid maximum
    values_mm: (
        tuple  # tuple[float, ...] - the annual maxima themselves, same order as years
    )
    year_records: list  # list[YearRecord] - every year considered, included or not
    n_native_intervals_per_window: int


@dataclass
class DistributionFit:
    """Fitted distribution for one duration (Stage 1.2/1.3)."""

    duration_label: str
    distribution: str  # 'GEV' | 'Gumbel' | 'GLO' | 'LP3'
    method: str  # 'L-moments' | 'MLE'
    location_xi: float
    scale_alpha: float
    shape_kappa: float | None  # None for Gumbel (kappa == 0 by construction)
    n_years: int
    l_moments: dict  # {'l1':..., 'l2':..., 't3':..., 't4':...} for transparency/audit
    extra: dict = field(default_factory=dict)  # LP3's log-space mu/sigma/gamma
    warnings: list = field(default_factory=list)


@dataclass
class QuantileEstimate:
    duration_label: str
    distribution: str
    exceedance_probability: float
    return_period_years: float
    depth_mm: float


@dataclass
class DurationRecommendation:
    """L-moment ratio diagram result for one duration - which
    distribution's shape best matches the observed data, and by how
    much it beat the alternatives."""

    duration_label: str
    tau3_sample: float
    tau4_sample: float
    recommended_distribution: str
    ranking: list  # list of (distribution_name, distance), best first


@dataclass
class FrequencyAnalysisResult:
    duration_series: list  # list[DurationSeries]
    fits: list  # list[DistributionFit] - ALL distributions x ALL durations
    quantiles: (
        list  # list[QuantileEstimate] - ALL distributions x ALL durations x ALL AEPs
    )
    recommendations: list = field(default_factory=list)  # list[DurationRecommendation]
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
