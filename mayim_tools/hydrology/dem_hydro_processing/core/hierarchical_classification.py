"""
Hierarchical depression classification.

THE PROBLEM THIS SOLVES (found by testing, not anticipated in
advance - see dev/README.md's "Findings this session" for the full
account): leaf-level classification cannot see a real basin in noisy
data. A genuine lake's floor is never perfectly smooth - in noisy
terrain it fragments into many noise-scale sub-pits (leaves) that
merge with each other at tiny elevation differences well below the
lake's true depth. A leaf's own depth is measured only to its FIRST
merge with a sibling, i.e. at noise scale - so no single leaf "is"
the lake; only the MERGED meta-depression that eventually contains
the whole lake floor has the lake's actual depth. Scoring a noisy
synthetic benchmark with known ground truth against "a real feature
is a leaf that IS the basin" found ZERO correctly-recognised real
leaves among roughly 100 candidates - not a rare edge case, the
typical case for any real depression on realistic, noisy terrain.

THE FIX: classify every node in the hierarchy - leaves AND
meta-depressions alike, each using its own footprint and its own
direct spill elevation (classification.compute_depression_metrics,
stochastic_classification.compute_persistence_with_baseline, and
stream_evidence.compute_stream_relation all now accept either, see
each function's own updated docstring) - then walk the hierarchy
TOP-DOWN, starting from each top-level feature (whichever node's
parent is the ocean): the first CONFIDENTLY NATURAL node encountered
on a given branch protects its entire subtree from resolution - its
descendant leaves are left untouched, even if some of them would
individually look artificial in isolation (that is expected: a real
lake floor's own micro-relief is exactly the kind of small, noise-
scale feature a leaf-only view would wrongly flag). Below a node that
is NOT confidently natural (Likely Artificial, or Uncertain), the
walk continues into its children, since a coarse verdict at that
level does not rule out a genuine feature nested somewhere inside it.

RESOLUTION, NOT JUST CLASSIFICATION, ALSO NEEDS TO WORK AT WHATEVER
LEVEL IS ACTUALLY ARTIFICIAL - a second real bug, found from a real
corrected DEM producing short, disconnected flow-accumulation after
running through this plugin, not anticipated by any test until then.
An earlier version of this module assumed (see "Findings this
session" in dev/README.md) that resolving each individual ARTIFICIAL
LEAF beneath a non-natural branch, each only to its own DIRECT
parent's spill elevation, was "roughly right" on its own. It is not,
for a long chain of artificial depressions: resolving a leaf only to
its immediate parent leaves every node FURTHER UP the same artificial
chain as its own, still-unresolved, residual depression - confirmed
directly on a 29-leaf synthetic chain (every leaf classified
artificial, none protected by anything natural), where "resolving"
every single leaf left 28 residual depressions behind, each one a
small local sink that breaks flow accumulation into short,
disconnected pieces exactly as reported. See
group_into_resolution_units() below for the fix: group a whole fully-
artificial chain or subtree under its own highest common ancestor
first, then resolve that WHOLE group as one unit - which
impact_reduction.resolve_leaf_with_least_impact can now do directly,
since it (like compute_depression_metrics and compute_stream_relation
before it) was already built generically on cells_of()/footprint_of()
and spill_elevation, and only its own validation was leaf-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classification import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    LABEL_LIKELY_ARTIFICIAL,
    LABEL_LIKELY_NATURAL,
    ClassificationResult,
    combine_evidence,
    compute_depression_metrics,
)
from .depression_hierarchy import DepressionHierarchy
from .stochastic_classification import (
    DEFAULT_CORRELATION_LENGTH_CELLS,
    DEFAULT_N_ITERATIONS,
    compute_persistence_with_baseline,
)
from .stream_evidence import compute_stream_relation


@dataclass
class HierarchicalClassificationResult:
    # every non-ocean node with positive volume gets its own classification,
    # leaf or meta alike - the full picture for REPORTING purposes
    all_classifications: dict[int, ClassificationResult] = field(default_factory=dict)
    # leaves selected for actual resolution (fill/breach) under the
    # resolve_labels this result was computed with - only reached on a
    # branch with no confidently-natural ancestor, see module docstring.
    # Cheap to recompute for a DIFFERENT resolve_labels choice (e.g. after
    # a human review step) via decide_resolution() below, without repeating
    # the expensive Monte Carlo work in classify_all_nodes().
    resolve_leaf_ids: list[int] = field(default_factory=list)
    # leaf id -> the natural ancestor node id that protects it from
    # resolution regardless of resolve_labels (may be the leaf's own id, if
    # the leaf itself was the natural node) - kept for audit/reporting, so
    # it is visible WHY a given leaf was spared
    protected_by: dict[int, int] = field(default_factory=dict)


def classify_all_nodes(
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    vertical_error_std: float,
    nodata: float | None = None,
    stream_mask: np.ndarray | None = None,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    correlation_length_cells: float = DEFAULT_CORRELATION_LENGTH_CELLS,
    window_margin: int | None = None,
    confidence_thresholds: tuple[float, float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    stream_buffer_cells: int = 1,
    seed: int = 0,
) -> dict[int, ClassificationResult]:
    """The expensive half of hierarchical classification: runs Monte
    Carlo persistence (and, where supplied, stream-relation evidence)
    for EVERY non-ocean node with positive volume - leaf or meta-
    depression alike - and combines each into its own
    ClassificationResult. Deliberately separated from the (cheap)
    resolution decision in decide_resolution() below, so a caller who
    wants to change WHICH labels get resolved (e.g. widening to
    include Uncertain after a human review step) can do so instantly,
    without repeating this expensive pass.

    Every node's own Monte Carlo run uses a generator seeded from
    (seed, node_id) - reproducible, and independent of how many other
    nodes are classified in the same run or in what order.
    """
    all_classifications: dict[int, ClassificationResult] = {}

    for node_id, node in hierarchy.nodes.items():
        if node.is_ocean:
            continue
        metrics = compute_depression_metrics(dem, hierarchy, node_id)
        if metrics.volume <= 0.0:
            continue  # a flat plateau "leaf" (or, in principle, a degenerate meta) with no real depression

        p = compute_persistence_with_baseline(
            dem,
            hierarchy,
            node_id,
            vertical_error_std,
            correlation_length_cells=correlation_length_cells,
            n_iterations=n_iterations,
            window_margin=window_margin,
            nodata=nodata,
            rng=np.random.default_rng([seed, node_id]),
        )

        relation = None
        if stream_mask is not None:
            relation = compute_stream_relation(
                hierarchy,
                node_id,
                stream_mask,
                buffer_cells=stream_buffer_cells,
                dem=dem,
            )

        all_classifications[node_id] = combine_evidence(
            dem,
            hierarchy,
            node_id,
            persistence_probability=p.adjusted,
            vertical_error_std=vertical_error_std,
            stream_relation=relation,
            confidence_thresholds=confidence_thresholds,
        )

    return all_classifications


def _walk_and_decide(
    hierarchy: DepressionHierarchy,
    labels: dict[int, str],
    resolve_labels: tuple[str, ...],
) -> tuple[list[int], dict[int, int]]:
    """The actual top-down walk (see module docstring for the
    algorithm), working from plain labels alone - the walk itself
    never needed anything from a ClassificationResult beyond its own
    `.label`, so this is the one true implementation both
    decide_resolution() (same-session, from freshly-computed
    ClassificationResult objects) and decide_resolution_from_labels()
    (cross-session, from a - possibly human-edited - CSV report read
    back via report.read_classification_report) build on, rather than
    two separate copies of the same tree-walking logic.

    Iterative (explicit stack), NOT recursive - the SAME fix, for the
    SAME reason, as depression_hierarchy.DepressionHierarchy's own
    _descendant_leaf_ids (see that function's own docstring for the
    full account of the real crash this was found from): real, noisy,
    basin-scale terrain typically produces a long CHAIN of sequential
    binary merges, not a balanced tree, so a recursive top-down walk
    from a top-level node can be O(number of leaves) stack frames
    deep. This walk has the identical shape (descend one child at a
    time through a binary-merge hierarchy) as the one that actually
    crashed in production, so it carries the same risk even though it
    had not yet been reported to crash itself - fixed proactively
    once the first instance of this exact bug pattern was found and
    understood, rather than waiting for a second crash report to go
    looking for it."""
    resolve_leaf_ids: list[int] = []
    protected_by: dict[int, int] = {}

    ocean = hierarchy.nodes[0]
    # each stack entry is (node_id, natural_ancestor | None)
    stack: list[tuple[int, int | None]] = [
        (top_level_id, None) for top_level_id in ocean.children
    ]

    while stack:
        node_id, natural_ancestor = stack.pop()
        node = hierarchy.nodes[node_id]
        label = labels.get(node_id)

        if natural_ancestor is not None:
            # already protected by a confidently-natural ancestor further up
            if node.children:
                stack.extend((child_id, natural_ancestor) for child_id in node.children)
            else:
                protected_by[node_id] = natural_ancestor
            continue

        if label == LABEL_LIKELY_NATURAL:
            if node.children:
                stack.extend(
                    (child_id, node_id) for child_id in node.children
                )  # this node becomes the protecting ancestor for its whole subtree
            else:
                protected_by[node_id] = node_id  # a natural LEAF protects itself
            continue

        # not confidently natural (Likely Artificial, Uncertain, or filtered
        # out entirely as zero-volume) and no natural ancestor - descend
        # further if there are children, since a coarse non-natural verdict
        # does not rule out a genuine feature nested inside; resolve only an
        # actual leaf, and only if its OWN verdict is in resolve_labels
        if node.children:
            stack.extend((child_id, None) for child_id in node.children)
        elif label in resolve_labels:
            resolve_leaf_ids.append(node_id)

    return resolve_leaf_ids, protected_by


def decide_resolution(
    hierarchy: DepressionHierarchy,
    all_classifications: dict[int, ClassificationResult],
    resolve_labels: tuple[str, ...] = (LABEL_LIKELY_ARTIFICIAL,),
) -> HierarchicalClassificationResult:
    """The cheap half: walks the hierarchy top-down from each top-
    level feature (see module docstring for the full algorithm) using
    already-computed classifications, deciding which leaves to resolve
    and which are protected by a confidently-natural ancestor. Safe
    to call repeatedly with a different `resolve_labels` against the
    SAME all_classifications - e.g. re-deciding after a human review
    step without re-running any Monte Carlo analysis.

    `resolve_labels` controls only the LEAF-level decision at the
    bottom of a non-natural branch (default: Likely Artificial only -
    widen to include Uncertain for a post-review run); it has no
    effect on which nodes count as a protecting natural ancestor,
    which always requires Likely Natural specifically, regardless of
    resolve_labels.
    """
    labels = {nid: c.label for nid, c in all_classifications.items()}
    resolve_leaf_ids, protected_by = _walk_and_decide(hierarchy, labels, resolve_labels)
    return HierarchicalClassificationResult(
        all_classifications=all_classifications,
        resolve_leaf_ids=resolve_leaf_ids,
        protected_by=protected_by,
    )


def decide_resolution_from_labels(
    hierarchy: DepressionHierarchy,
    labels: dict[int, str],
    resolve_labels: tuple[str, ...] = (LABEL_LIKELY_ARTIFICIAL,),
) -> HierarchicalClassificationResult:
    """The cross-session counterpart to decide_resolution(): decides
    resolution from plain {node_id: label} pairs rather than full
    ClassificationResult objects - built specifically for the real
    QGIS workflow, where Stage 1 and Stage 2 run as SEPARATE
    Processing algorithm invocations with no shared Python objects
    between them. `hierarchy` should be freshly rebuilt (via
    depression_hierarchy.build_depression_hierarchy) from the SAME DEM
    Stage 1 analysed - this is fast and fully deterministic, so Stage
    2 never needs to repeat Stage 1's expensive Monte Carlo work, only
    the cheap hierarchy construction. `labels` is typically read back
    from Stage 1's own CSV report via
    report.read_classification_report() - which may have been edited
    by hand during human review (the whole point of writing it to a
    plain, editable CSV in the first place) before Stage 2 runs, so a
    reviewer's corrections are honoured exactly as if they had been
    the tool's own original verdict.

    `all_classifications` on the returned HierarchicalClassificationResult
    is left empty here (there are no ClassificationResult objects to
    report - only bare labels) - callers needing the full evidence
    trail should read the same CSV directly rather than relying on this
    field in the cross-session path.
    """
    resolve_leaf_ids, protected_by = _walk_and_decide(hierarchy, labels, resolve_labels)
    return HierarchicalClassificationResult(
        all_classifications={},
        resolve_leaf_ids=resolve_leaf_ids,
        protected_by=protected_by,
    )


def classify_hierarchy(
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    vertical_error_std: float,
    nodata: float | None = None,
    stream_mask: np.ndarray | None = None,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    correlation_length_cells: float = DEFAULT_CORRELATION_LENGTH_CELLS,
    window_margin: int | None = None,
    confidence_thresholds: tuple[float, float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    stream_buffer_cells: int = 1,
    seed: int = 0,
    resolve_labels: tuple[str, ...] = (LABEL_LIKELY_ARTIFICIAL,),
) -> HierarchicalClassificationResult:
    """Convenience wrapper combining classify_all_nodes() and
    decide_resolution() for the common one-shot case. Prefer calling
    the two separately when `resolve_labels` might change later
    without new terrain data (e.g. a human review step) - see each
    function's own docstring."""
    all_classifications = classify_all_nodes(
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
    return decide_resolution(
        hierarchy, all_classifications, resolve_labels=resolve_labels
    )


def _compute_fully_resolved(
    hierarchy: DepressionHierarchy, resolve_set: set[int]
) -> dict[int, bool]:
    """For every non-ocean node, whether EVERY descendant leaf is in
    resolve_set - the test group_into_resolution_units() below needs
    to find how far up a chain of resolved leaves can be merged into
    one resolution unit. Iterative post-order traversal (no recursion
    - see depression_hierarchy.py's own _descendant_leaf_ids docstring
    for why that matters at scale) and linear in the number of nodes:
    each node's own result is computed once, from its own children's
    already-computed results, not by re-walking each subtree from
    scratch for every node (which would be quadratic on a long chain -
    precisely the shape real, noisy terrain produces, and precisely
    where this fix is needed most)."""
    fully_resolved: dict[int, bool] = {}
    ocean = hierarchy.nodes[0]
    stack: list[tuple[int, bool]] = [(nid, False) for nid in ocean.children]
    while stack:
        node_id, children_done = stack.pop()
        node = hierarchy.nodes[node_id]
        if not node.children:
            fully_resolved[node_id] = node_id in resolve_set
            continue
        if children_done:
            fully_resolved[node_id] = all(fully_resolved[c] for c in node.children)
        else:
            stack.append((node_id, True))
            stack.extend((c, False) for c in node.children)
    return fully_resolved


def group_into_resolution_units(
    hierarchy: DepressionHierarchy, resolve_leaf_ids: list[int]
) -> list[int]:
    """Groups a flat list of leaves selected for resolution (from
    decide_resolution() or decide_resolution_from_labels() above) into
    the actual units that should be resolved - fixing a real bug found
    from a real corrected DEM, not a hypothetical one (see this
    module's own docstring for the full account): resolving every leaf
    individually, each only to its own DIRECT parent's spill elevation,
    leaves every node FURTHER UP a chain of artificial depressions as
    its own residual, unresolved depression. Confirmed directly: a
    29-leaf synthetic chain, every leaf resolved, left 28 residual
    depressions behind - each one breaking flow accumulation into
    short, disconnected pieces.

    For each resolved leaf, walks UP the hierarchy as far as every
    node's FULL set of descendant leaves is also in resolve_leaf_ids -
    i.e. as far as the chain stays entirely artificial - stopping at
    the first ancestor that either (a) has a descendant leaf NOT
    selected for resolution (an unresolved sibling - a Likely Natural
    or Uncertain feature nested in the same branch, which must not be
    affected by resolving this chain), or (b) is itself a top-level
    node (its own parent is ocean). The result is the HIGHEST common
    ancestor of every maximal, fully-artificial run - a leaf id if it
    has no fully-resolved parent, otherwise a meta-depression id.

    Returned ids are fed directly to
    impact_reduction.apply_ira_to_all_leaves's own `leaf_ids` parameter
    (which, despite its name, now accepts either - see that function's
    own updated docstring): each resolution unit is resolved as ONE
    fill-or-breach operation against its own cells_of()/spill_elevation
    (both already defined generically for leaf or meta nodes), correctly
    draining the whole unit to where it actually becomes protected or
    reaches ocean, rather than one small step at a time.
    """
    resolve_set = set(resolve_leaf_ids)
    fully_resolved = _compute_fully_resolved(hierarchy, resolve_set)

    units: set[int] = set()
    for leaf_id in resolve_leaf_ids:
        current = leaf_id
        while True:
            parent_id = hierarchy.nodes[current].parent
            if parent_id == 0 or not fully_resolved.get(parent_id, False):
                break
            current = parent_id
        units.add(current)
    return sorted(units)
