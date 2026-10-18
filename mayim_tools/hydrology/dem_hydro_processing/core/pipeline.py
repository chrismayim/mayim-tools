"""
End-to-end pipeline: analyse a DEM, then apply the corrections the
analysis selects. This is the pure-Python core of the two-stage
design agreed for this tool (Stage 1 "Analyze", Stage 2 "Apply") -
the QGIS wrapper, vector/CSV output and styling still to be built sit
on top of exactly these two functions.

analyze_dem():
    build the depression hierarchy and classify EVERY node
    (core/hierarchical_classification.py's classify_all_nodes) - every
    leaf AND every meta-depression, not leaves alone. This replaced an
    earlier, leaf-only version of this function after testing found
    leaf-only classification cannot recognise a real basin in noisy
    data (see hierarchical_classification.py's own module docstring
    for the full finding) - a genuine lake's noisy floor fragments
    into many small leaves, and only the merged meta-depression that
    covers the whole floor has the lake's true depth. Deliberately
    does NOT also decide which leaves to resolve - see apply_corrections.

apply_corrections():
    decides which leaves to resolve (hierarchical_classification's own
    decide_resolution - cheap, no new Monte Carlo work, so this can be
    called again with a different `resolve_labels` after a human
    review step without repeating Stage 1) and resolves only those,
    using the fill-or-breach-whichever-disturbs-less decision
    (impact_reduction), with breaching guided along the mapped stream
    where the stream passes through the depression. Leaves protected
    by a confidently-natural ancestor are never resolved, regardless
    of resolve_labels or how they would look in isolation - the whole
    point of hierarchical classification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .classification import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    LABEL_LIKELY_ARTIFICIAL,
    ClassificationResult,
)
from .depression_hierarchy import DepressionHierarchy, build_depression_hierarchy
from .drainage_enforcement import enforce_after_resolution
from .hierarchical_classification import (
    classify_all_nodes,
    decide_resolution,
    group_into_resolution_units,
)
from .impact_reduction import IRAResult, apply_ira_to_all_leaves
from .stochastic_classification import (
    DEFAULT_CORRELATION_LENGTH_CELLS,
    DEFAULT_N_ITERATIONS,
)


@dataclass
class AnalysisResult:
    hierarchy: DepressionHierarchy
    classifications: dict[
        int, ClassificationResult
    ]  # every classified node, leaf or meta - keyed by node id
    stream_mask: np.ndarray | None = None
    skipped_non_depression_leaves: int = (
        0  # flat plateau "leaves" with no volume - reported, not silently dropped
    )


def analyze_dem(
    dem: np.ndarray,
    vertical_error_std: float,
    nodata: float | None = None,
    stream_mask: np.ndarray | None = None,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    correlation_length_cells: float = DEFAULT_CORRELATION_LENGTH_CELLS,
    window_margin: int | None = None,
    confidence_thresholds: tuple[float, float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    stream_buffer_cells: int = 1,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    """Stage 1. Builds the hierarchy and classifies every node (see
    module docstring) - leaves and meta-depressions alike. Does not
    yet decide which leaves to resolve; see apply_corrections().

    `progress`, if given, is called once with
    (total_classified, total_classified) after classification
    completes - a genuine per-node progress callback is not yet built
    (classify_all_nodes does not itself call back mid-run; see
    dev/README.md's Still To Build).
    """
    hierarchy = build_depression_hierarchy(dem, nodata=nodata)

    classifications = classify_all_nodes(
        dem,
        hierarchy,
        vertical_error_std,
        nodata=nodata,
        stream_mask=stream_mask,
        n_iterations=n_iterations,
        correlation_length_cells=correlation_length_cells,
        window_margin=window_margin,
        confidence_thresholds=confidence_thresholds,
        stream_buffer_cells=stream_buffer_cells,
        seed=seed,
    )

    skipped = sum(
        1
        for nid, n in hierarchy.nodes.items()
        if not n.is_ocean and not n.children and nid not in classifications
    )

    if progress is not None:
        progress(len(classifications), len(classifications))

    return AnalysisResult(
        hierarchy=hierarchy,
        classifications=classifications,
        stream_mask=stream_mask,
        skipped_non_depression_leaves=skipped,
    )


def apply_corrections(
    dem: np.ndarray,
    analysis: AnalysisResult,
    resolve_labels: tuple[str, ...] = (LABEL_LIKELY_ARTIFICIAL,),
    nodata: float | None = None,
    max_breach_length: int | None = None,
    max_breach_depth: float | None = None,
    enforce: bool = True,
) -> tuple[np.ndarray, list[IRAResult]]:
    """Stage 2. Decides which leaves to resolve (cheap - no new Monte
    Carlo work, see hierarchical_classification.decide_resolution) and
    returns (corrected DEM, per-leaf/unit IRA results). Leaves
    protected by a confidently-natural ancestor are left exactly as
    they were, regardless of resolve_labels or how they would look in
    isolation - the whole point of hierarchical classification. Safe
    to call again on the SAME `analysis` with a different
    `resolve_labels` (e.g. widening to include Uncertain after a human
    review step) without re-running Stage 1.

    Resolved leaves are GROUPED into resolution units
    (hierarchical_classification.group_into_resolution_units) before
    being handed to IRA - fixing a real bug found from a real
    corrected DEM, not a hypothetical one (see that function's own
    docstring for the full account): resolving each artificial leaf in
    a long chain individually, each only to its own direct parent,
    left a whole staircase of residual depressions behind, breaking
    flow accumulation into short, disconnected pieces. Grouping a
    fully-artificial chain or subtree under its own highest common
    ancestor first, then resolving that whole group in one operation,
    fixes this for the common case (a uniformly-classified connected
    region) - confirmed directly: a 29-leaf synthetic chain, every
    leaf artificial, went from 28 residual depressions to zero.
    A scattered exclusion (one leaf mid-chain classified differently
    from its neighbours on both sides) is a known, not yet fully
    solved, remaining limitation - see hierarchical_classification.py's
    own section in dev/README.md for the honest account of why, and
    its own test suite for a test that documents the gap directly
    rather than hiding it.
    """
    resolution = decide_resolution(
        analysis.hierarchy, analysis.classifications, resolve_labels=resolve_labels
    )
    resolution_units = group_into_resolution_units(
        analysis.hierarchy, resolution.resolve_leaf_ids
    )
    corrected, results = apply_ira_to_all_leaves(
        dem,
        analysis.hierarchy,
        nodata=nodata,
        max_breach_length=max_breach_length,
        max_breach_depth=max_breach_depth,
        leaf_ids=resolution_units,
        stream_mask=analysis.stream_mask,
        stream_relations={
            nid: c.stream_relation
            for nid, c in analysis.classifications.items()
            if c.stream_relation is not None
        },
    )
    if enforce:
        corrected, _ = enforce_after_resolution(
            corrected, dem, analysis.hierarchy, resolution.resolve_leaf_ids, nodata
        )
    return corrected, results
