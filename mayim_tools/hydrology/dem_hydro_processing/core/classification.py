"""
Multi-line-of-evidence depression/ridge classification.

Combines the persistence probability from stochastic_classification.py
with two further, deliberately secondary lines of evidence into a
single confidence score and a three-way label. This directly follows
the design agreed in the original project scoping discussion
(2026-09-23) and the conclusion of Lindsay & Creed (2006) itself: "no
single method is reliable alone... combination approaches may be used
to overcome some of their individual limitations."

WHY PERSISTENCE IS WEIGHTED MOST HEAVILY, AND SIZE/DEPTH ARE KEPT
DELIBERATELY SECONDARY: Lindsay & Creed's own controlled comparison
found simple size/depth/volume thresholds - the most common approach
in practice - performed WORST of the methods tested, wrongly removing
several genuine small depressions. Stochastic persistence performed
best (75% correct). This module does not repeat that mistake by
letting size/depth act as a hard threshold or veto; it contributes a
modest, bounded adjustment to the persistence-based score, not an
independent gate a depression must also pass. See combine_evidence()
for the exact weighting and reasoning.

DEM_SOURCE_VERTICAL_ERROR_M: default vertical accuracy (as a standard
deviation, the form this tool's own Monte Carlo noise model needs) by
DEM source type.

RE-RESEARCHED AND REVISED (2026-10-03), after Chris reported that
depressions he independently knows to be genuine were not coming back
Likely Natural, and that the Copernicus GLO-30 default in particular
looked too generous. Investigated properly rather than adjusting the
number by feel - the original 2026-09-23 figures were closer to
published SPECIFICATION ceilings (often a global, all-terrain, worst-
case figure) than to what independent validation studies actually
measure, and spec sheets themselves are frequently reported as LE90
(a 90th-percentile linear error), not as a standard deviation - using
an LE90 figure directly as this tool's own std parameter silently
treats it as about 1.65x too generous (LE90 = 1.6449 x RMSE for a
normal error distribution, confirmed directly against a DLR/TanDEM-X
accuracy-assessment paper's own stated conversion, and independently
cross-checked against an SRTM accuracy paper using the identical
1.6449 factor).

A genuinely important, source-specific complication, not just a
units issue: Copernicus GLO-30, SRTM and ASTER are all Digital
SURFACE Models (they return the top of whatever is there - canopy,
buildings - not bare earth), so their accuracy over VEGETATED or
BUILT-UP land is dominated by canopy/building height, not by random
measurement noise - a real, large, systematic offset, not the kind of
noise this tool's Monte Carlo model is simulating. Over BARE/OPEN
ground specifically, independent validation against airborne LiDAR
finds these products are considerably more accurate than their global
specification figures suggest. Since hydrological depression analysis
is most often applied to drainage lines, floodplains and valleys -
which are open or lightly vegetated far more often than they are
dense forest or urban canopy - the revised defaults below lean toward
the open-ground figures rather than the worst-case global ones, while
still sitting above the very best reported figures to leave headroom
for some vegetation/structure being present. This is still a starting
default, not a substitute for knowing your own terrain's land cover -
a DEM over dense forest or steep mountains should use a notably
larger custom value; the sources below report that explicitly.

- survey (total station / GNSS RTK): 0.02 m, unchanged - well-
  established, typically 1-3 cm vertical for RTK, sub-cm to 1 cm for a
  total station.
- lidar: 0.10 m, unchanged - confirmed directly against the current
  (2022 rev. A) USGS 3DEP Lidar Base Specification: QL1 and QL2 both
  specify 10 cm RMSEz (non-vegetated). USGS's own 2023 program-wide
  monitoring reports the SEAMLESS, multi-project 1/3 arc-second
  product (which blends in older, lower-quality legacy data) at about
  0.82 m RMSE - the 0.10 m default here assumes a single, modern
  QL1/QL2 project specifically, which is what "lidar" means as a
  distinct option from the generic seamless/legacy case.
- uav_photogrammetry: 0.08 m, unchanged - well-supported by multiple
  recent ground-control-point studies converging on roughly 3-12 cm
  vertical RMSE for adequately-controlled surveys (direct
  georeferencing with no ground control at all measured notably worse,
  around 9-14 cm in two studies - a reminder that this default assumes
  real ground control, as the parameter name has always said).
- copernicus_glo30: REVISED 2.5 -> 1.5 m. The vendor's own global LE90
  specification is <4 m (-> std ~2.4 m as a worst case), but the same
  documentation states that over 60% of the world (about 50% of
  Europe) already has LE90 better than 2 m (-> std ~1.2 m), and
  independent validation against airborne LiDAR over BARE GROUND
  specifically finds considerably better performance still - a 2024
  Journal of Geophysical Research study found RMSE of 0.55-0.92 m
  across multiple LiDAR reference datasets (never exceeding 2 m for
  99% of tiles), and a separate Madagascar case study reported RMSE
  1.68 m. 1.5 m sits inside this open-ground range while leaving
  headroom for imperfectly-open terrain. Over genuinely forested or
  mountainous extents, independent studies instead found RMSE of
  5-14 m - use a custom value well above 1.5 m for those.
- srtm: REVISED 6.0 -> 5.0 m. The mission's own target is 16 m LE90
  (~9.7 m std) globally, but independent comparisons in relatively
  open/moderate terrain repeatedly land in the 3-6 m RMSE range (e.g.
  2.9-5.4 m in several regional studies), while mountainous or densely
  vegetated terrain pushes well past 10 m RMSE in others. Like
  Copernicus, SRTM is a radar-derived surface model, so the same
  canopy/structure caveat applies - use a custom value well above
  5.0 m for forested or mountainous extents.
- aster: REVISED 12.0 -> 10.0 m. The most recent, large-sample
  official validation (GDEM v3, USGS/METI, >23,000 geodetic ground
  control points) reports 8.52 m RMSE; independent regional studies
  consistently fall in the 8-15 m band. 10.0 m sits near the better-
  supported end of that range, reflecting the more authoritative
  recent validation, while ASTER's documented stereo-matching noise
  and cloud-contamination artefacts mean it should still be treated as
  the least reliable of the commonly-available global options.

Sources (all retrieved 2026-10-03): DLR TanDEM-X/Copernicus accuracy-
assessment literature (LE90 = 1.6449 x RMSE conversion, Copernicus
LE90 <4 m / >60% world <2 m figures); Simard et al. 2024, JGR
Biogeosciences (GLO-30 bare-ground RMSE against LiDAR/ICESat/GEDI);
ESurf 2022 Madagascar Copernicus/TanDEM-X/UAV-SfM comparison (2.17 m
absolute accuracy / 1.68 m RMSE); multiple SRTM regional accuracy
studies (bare/open terrain 2.9-5.4 m RMSE vs. mountainous/vegetated
7-27 m); USGS/METI ASTER GDEM v3 CONUS validation report (8.52 m
RMSE); USGS 3DEP Lidar Base Specification 2022 rev. A and 2023 3DEP
program status briefing; multiple UAV-SfM ground-control-point studies
(Remote Sensing, Measurement, and related venues, 2016-2025).

These remain STARTING DEFAULTS the caller should override with
project-specific QA data where available (see this project's own
design discussion) - a generic category default should never outrank
a user's own survey accuracy report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depression_hierarchy import DepressionHierarchy
from .stream_evidence import STREAM_NONE, STREAM_PASSES_THROUGH, STREAM_TERMINATES_IN

DEM_SOURCE_VERTICAL_ERROR_M = {
    "survey": 0.02,  # total station / GNSS RTK - unchanged, see module docstring
    "lidar": 0.10,  # airborne LiDAR, USGS 3DEP QL1/QL2 spec (10 cm RMSEz) - unchanged
    "uav_photogrammetry": 0.08,  # UAV/drone structure-from-motion with adequate ground control - unchanged
    "copernicus_glo30": 1.5,  # REVISED 2026-10-03 from 2.5 - see module docstring for the full research
    "srtm": 5.0,  # REVISED 2026-10-03 from 6.0 - see module docstring
    "aster": 10.0,  # REVISED 2026-10-03 from 12.0 - see module docstring
}

DEFAULT_CONFIDENCE_THRESHOLDS = (
    0.35,
    0.65,
)  # below first -> Likely Artificial; above second -> Likely Natural; between -> Uncertain

LABEL_LIKELY_ARTIFICIAL = "Likely Artificial"
LABEL_UNCERTAIN = "Uncertain - Review"
LABEL_LIKELY_NATURAL = "Likely Natural"


@dataclass
class DepressionMetrics:
    leaf_id: int
    num_cells: int
    volume: float  # sum over all cells of (spill_elevation - original_elevation) - a proxy for how much water/material the feature represents
    max_depth: float
    mean_depth: float


@dataclass
class ClassificationResult:
    leaf_id: int
    metrics: DepressionMetrics
    persistence_probability: float
    size_adjustment: float  # the bounded secondary contribution actually applied, for transparency/audit
    stream_crossing: bool | None  # None = no stream data supplied for this feature
    confidence: float  # combined score in [0, 1]: 0 = confidently artificial, 1 = confidently natural
    label: str
    stream_relation: str | None = (
        None  # see stream_evidence.compute_stream_relation; None = not supplied
    )


def compute_depression_metrics(
    dem: np.ndarray, hierarchy: DepressionHierarchy, leaf_id: int
) -> DepressionMetrics:
    """Basic geometric metrics for a single depression, using its own
    direct spill elevation (see impact_reduction.py's own docstring
    for why this is the right scope - one specific depression's own
    extent, not everything further up the hierarchy).

    Despite the parameter's name (kept for minimal disruption to
    existing callers), this now accepts ANY non-ocean node - a leaf OR
    a meta-depression. Needed for hierarchical classification (see
    core/hierarchical_classification.py): a real basin in noisy data
    fragments into many noise-scale leaves that merge into it, so only
    the merged meta-depression has the basin's own true depth - see
    that module's own docstring for the full finding. footprint_of()
    (used below) already handles both cases correctly, since it works
    from cells_of()'s own recursive descendant-leaf union - only this
    function's validation was leaf-only."""
    node = hierarchy.nodes[leaf_id]
    if node.is_ocean:
        raise ValueError(
            f"compute_depression_metrics expects a non-ocean node, got the ocean node {leaf_id}"
        )
    if node.spill_elevation is None:
        return DepressionMetrics(
            leaf_id=leaf_id, num_cells=0, volume=0.0, max_depth=0.0, mean_depth=0.0
        )

    dem64 = dem.astype(np.float64)
    # the FOOTPRINT (cells below the spill elevation), not cells_of(): the
    # latter is the leaf's catchment and includes uphill cells that are not
    # part of the depression at all (see DepressionHierarchy.footprint_of)
    mask = hierarchy.footprint_of(leaf_id, dem64)
    rows, cols = np.nonzero(mask)
    depths = node.spill_elevation - dem64[rows, cols]
    depths = np.clip(
        depths, 0, None
    )  # guard against float noise producing a tiny negative

    return DepressionMetrics(
        leaf_id=leaf_id,
        num_cells=len(rows),
        volume=float(depths.sum()),
        max_depth=float(depths.max()) if len(depths) else 0.0,
        mean_depth=float(depths.mean()) if len(depths) else 0.0,
    )


def _size_adjustment(metrics: DepressionMetrics, vertical_error_std: float) -> float:
    """A modest, BOUNDED secondary contribution in [-0.15, +0.15] -
    deliberately small relative to persistence's own [0, 1] range, so
    it nudges rather than dominates or gates the final score (see
    module docstring for why size/depth must not act as a hard
    threshold here).

    Two components, each contributing up to half the total bound:
    - depth relative to the DEM's own assumed noise level (a
      depression many standard deviations deeper than plausible noise
      is more likely genuine) - using a smooth logistic transition
      rather than a hard cutoff at some specific multiple.
    - cell count (a larger, spatially coherent depression is
      statistically less likely to have arisen from independent random
      noise than a 1-2 cell pit) - also a smooth logistic transition,
      not a hard minimum-size rule.
    """
    if vertical_error_std <= 0:
        depth_component = 0.0
    else:
        depth_in_sigmas = metrics.max_depth / vertical_error_std
        # logistic centred at 2 sigma - roughly where a depth starts being
        # implausible to attribute to noise alone, chosen as a reasoned,
        # not published, default
        depth_component = (
            2.0 / (1.0 + np.exp(-(depth_in_sigmas - 2.0))) - 1.0
        )  # in [-1, 1]

    # logistic centred at 4 cells - a single-cell or pair-of-cells pit is
    # the classic noise artefact shape; anything notably larger starts
    # looking more like real, coherent terrain
    size_component = (
        2.0 / (1.0 + np.exp(-(metrics.num_cells - 4) / 3.0)) - 1.0
    )  # in [-1, 1]

    return 0.075 * depth_component + 0.075 * size_component


def combine_evidence(
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    leaf_id: int,
    persistence_probability: float,
    vertical_error_std: float,
    stream_crossing: bool | None = None,
    stream_relation: str | None = None,
    confidence_thresholds: tuple[float, float] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> ClassificationResult:
    """Combines the Monte Carlo persistence probability (primary
    evidence) with a bounded size/depth adjustment (secondary) and, if
    supplied, stream-crossing evidence (a strong override) into a
    single confidence score and three-way label.

    STREAM-CROSSING EVIDENCE: for a depression, a mapped stream
    crossing into or through it is treated as strong evidence it is a
    genuine, fed water body, not an artefact - this project's own
    modern equivalent of Lindsay & Creed's "examine the source data"
    approach (2006), using whatever reference hydrography is
    available on a given project rather than requiring it. Applied as
    a strong pull toward "natural" (not an absolute override, so a
    single mapped stream crossing a huge, obviously-artificial flat
    area does not blindly force a "natural" verdict) - implemented as
    blending the persistence-based confidence toward 0.9 with a high
    weight when stream_crossing is True. `stream_crossing=None` means
    no stream data was supplied for this feature at all (the common
    case until stream-centreline rasterization, still to be built, is
    wired in) - treated as neutral, not as evidence of absence.
    """
    metrics = compute_depression_metrics(dem, hierarchy, leaf_id)
    size_adj = _size_adjustment(metrics, vertical_error_std)

    base_confidence = float(np.clip(persistence_probability + size_adj, 0.0, 1.0))

    # stream_relation (from stream_evidence.compute_stream_relation) is the
    # richer signal and takes precedence; the plain boolean stream_crossing is
    # kept only for callers that cannot tell a stream ENDING in a depression
    # from one passing through it, and is treated as "terminates_in" when True.
    if stream_relation is None and stream_crossing is not None:
        stream_relation = STREAM_TERMINATES_IN if stream_crossing else STREAM_NONE

    if stream_relation == STREAM_TERMINATES_IN:
        # a stream feeds into (and ends in) the depression: strong pull toward
        # a genuine waterbody, not an absolute override
        confidence = 0.3 * base_confidence + 0.7 * 0.9
    elif stream_relation == STREAM_PASSES_THROUGH:
        # a stream is mapped flowing straight across a supposedly closed basin:
        # water following it would not be trapped, so the barrier is very
        # likely an uncaptured culvert/bridge/embankment. Strong pull toward
        # ARTIFICIAL (deliberately the mirror of the terminates_in pull) - the
        # opposite conclusion from a stream that merely ends in the depression.
        confidence = 0.3 * base_confidence + 0.7 * 0.1
    elif stream_relation == STREAM_NONE:
        # a stream layer WAS supplied and nothing was found nearby - a much
        # milder signal than a confirmed relation (absence of evidence in a
        # sparse vector dataset is weak), so only a small pull toward artificial
        confidence = 0.85 * base_confidence + 0.15 * 0.2
    else:
        confidence = base_confidence

    confidence = float(np.clip(confidence, 0.0, 1.0))

    low, high = confidence_thresholds
    if confidence < low:
        label = LABEL_LIKELY_ARTIFICIAL
    elif confidence > high:
        label = LABEL_LIKELY_NATURAL
    else:
        label = LABEL_UNCERTAIN

    return ClassificationResult(
        leaf_id=leaf_id,
        metrics=metrics,
        persistence_probability=persistence_probability,
        size_adjustment=size_adj,
        stream_crossing=stream_crossing,
        confidence=confidence,
        label=label,
        stream_relation=stream_relation,
    )
