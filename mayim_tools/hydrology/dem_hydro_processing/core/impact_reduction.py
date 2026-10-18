"""
Impact Reduction Approach (IRA).

Lindsay, J.B. and Creed, I.F. (2005). "Removal of artifact
depressions from digital elevation models: towards a minimum impact
approach." Hydrological Processes, 19(16), 3113-3126.

For each depression, IRA resolves it using whichever of filling or
breaching disturbs the DEM less - rather than committing to one
method for the whole DEM (as a plain "always fill" or "always breach"
workflow does). The paper's own stated impact metric combines two
figures: the number of modified cells and the mean absolute
elevation difference. This implementation uses TOTAL absolute
elevation change (the sum of |change| across every modified cell) as
the single decision criterion - since num_cells x mean_change =
total_change, this single number already incorporates both of the
paper's reported metrics multiplicatively, giving one clean quantity
("how much earth moved, in total") to compare rather than needing an
arbitrary weighting between two separate figures. Both component
metrics (cell count, mean change) are still reported separately in
ImpactMetrics for diagnostic purposes, matching the paper's own
practice of reporting both.

WHICH SCOPE OF "FILLING" IS COMPARED: a depression's fill cost here
is the cost to raise it to its own DIRECT spill elevation (the point
where it first merges with its immediate parent in the hierarchy) -
not all the way to wherever it ultimately reaches the ocean. This
matches the intent of resolving one specific depression at a time:
if a small pit turns out to be an artefact, only IT is resolved here:
whatever larger feature it belongs to further up the hierarchy is a
separate decision, made separately (by classification logic built on
top of this module, not assumed automatically here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .breaching import BreachResult, least_cost_breach_from_pit
from .depression_hierarchy import DepressionHierarchy
from .flat_resolution import resolve_flat_gradient
from .windowed import NodeWindows, edge_flags


@dataclass
class ImpactMetrics:
    num_cells: int
    total_abs_change: float

    @property
    def mean_abs_change(self) -> float:
        return self.total_abs_change / self.num_cells if self.num_cells else 0.0


@dataclass
class IRAResult:
    leaf_id: int
    method: str  # "fill" or "breach"
    fill_impact: ImpactMetrics
    breach_impact: ImpactMetrics | None  # None if no valid breach was found
    breach_result: BreachResult | None
    elevation_changes: dict[
        tuple[int, int], float
    ]  # cell -> NEW elevation, for cells actually modified by the chosen method
    touched_window_edge: bool = False


def _fill_impact_for_leaf(
    dem64: np.ndarray, hierarchy: DepressionHierarchy, leaf_id: int
) -> tuple[ImpactMetrics, dict]:
    node = hierarchy.nodes[leaf_id]
    if node.spill_elevation is None:
        # a leaf with no parent yet (shouldn't occur in a well-formed
        # hierarchy from build_depression_hierarchy, but handled
        # explicitly rather than silently producing a wrong answer)
        return ImpactMetrics(0, 0.0), {}
    mask = hierarchy.cells_of(leaf_id)
    rows, cols = np.nonzero(mask)
    changes = {}
    total = 0.0
    count = 0
    for r, c in zip(rows.tolist(), cols.tolist()):
        original = dem64[r, c]
        diff = node.spill_elevation - original
        if diff > 0:
            changes[(r, c)] = node.spill_elevation
            total += diff
            count += 1
    return ImpactMetrics(count, total), changes


def _breach_impact_from_result(result: BreachResult) -> ImpactMetrics:
    total = sum(result.excavation.values())
    count = sum(1 for v in result.excavation.values() if v > 0)
    return ImpactMetrics(count, total)


def resolve_leaf_with_least_impact(
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    leaf_id: int,
    nodata: float | None = None,
    max_breach_length: int | None = None,
    max_breach_depth: float | None = None,
    breach_allowed_mask: np.ndarray | None = None,
    prefer_breach: bool = False,
    breach_edge_flags: tuple[bool, bool, bool, bool] | None = None,
) -> IRAResult:
    """Compares filling vs breaching for a single depression and
    returns whichever has the lower total impact (see module
    docstring). Fill is always available (every non-root node in a
    well-formed hierarchy has a spill elevation); breach may fail
    (blocked by NoData, or exceeding the length/depth bounds), in
    which case fill is used regardless of its own cost, since it is
    the only option available.

    Despite the parameter's name (kept for minimal disruption to
    existing callers), this now accepts ANY non-ocean node - a leaf OR
    a meta-depression - matching classification.compute_depression_metrics
    and stream_evidence.compute_stream_relation, which were broadened
    the same way for the same underlying reason. The reason here is a
    real bug, not a hypothetical one (see
    hierarchical_classification.group_into_resolution_units's own
    docstring for the full account): resolving every leaf in a long
    chain of artificial depressions individually, each only to its own
    DIRECT parent's spill elevation, leaves a whole staircase of
    residual small depressions behind - confirmed directly on a 29-leaf
    synthetic chain, where 28 residual depressions remained after
    "resolving" every single leaf. group_into_resolution_units groups
    a fully-artificial chain or subtree under its own highest common
    ancestor first; this function (and _fill_impact_for_leaf, which it
    calls, already built generically on cells_of()/spill_elevation)
    then resolves that WHOLE group in one operation, correctly
    draining it all the way to where the chain actually becomes
    protected or reaches ocean, rather than one small step at a time.
    """
    dem64 = dem.astype(np.float64)
    node = hierarchy.nodes[leaf_id]
    if node.is_ocean:
        raise ValueError(
            f"resolve_leaf_with_least_impact expects a non-ocean node, got the ocean node {leaf_id}"
        )

    fill_impact, fill_changes = _fill_impact_for_leaf(dem64, hierarchy, leaf_id)
    if fill_changes:
        # a flat-area gradient is superimposed on the fill BEFORE any breach
        # comparison, so both return paths below benefit automatically - a
        # real bug found from real data, not a hypothetical one: filling
        # raises every cell to one uniform elevation, which a standard D8
        # flow-direction algorithm cannot resolve a direction across (every
        # neighbour is at the same elevation), breaking flow accumulation
        # into short, disconnected pieces exactly as reported. See
        # flat_resolution.py's own module docstring for the full account,
        # the Garbrecht & Martz (1997) / Barnes et al. (2014) method used,
        # and why the added gradient is tiny enough to leave the DEM's real
        # hydrology unaffected.
        group_mask = np.zeros_like(dem64, dtype=bool)
        for cell in fill_changes:
            group_mask[cell] = True
        gradient_changes = resolve_flat_gradient(
            dem64, group_mask, node.spill_elevation, nodata=nodata
        )
        fill_changes.update(gradient_changes)

    leaf_mask = hierarchy.cells_of(leaf_id)
    leaf_rows, leaf_cols = np.nonzero(leaf_mask)
    leaf_elevs = dem64[leaf_rows, leaf_cols]
    min_idx = int(np.argmin(leaf_elevs))
    pit_row, pit_col = int(leaf_rows[min_idx]), int(leaf_cols[min_idx])

    breach_result = None
    touched = False
    if breach_allowed_mask is not None:
        # stream-guided attempt first (see breaching.py's allowed_mask) - cut
        # where a mapped stream actually crosses the barrier
        breach_result = least_cost_breach_from_pit(
            dem64,
            pit_row,
            pit_col,
            nodata=nodata,
            max_breach_length=max_breach_length,
            max_breach_depth=max_breach_depth,
            allowed_mask=breach_allowed_mask,
            edge_flags=breach_edge_flags,
        )
    if breach_result is not None:
        touched = breach_result.touched_window_edge
    if breach_result is None or not breach_result.success:
        breach_result = least_cost_breach_from_pit(
            dem64,
            pit_row,
            pit_col,
            nodata=nodata,
            max_breach_length=max_breach_length,
            max_breach_depth=max_breach_depth,
            edge_flags=breach_edge_flags,
        )

    touched = touched or breach_result.touched_window_edge
    if not breach_result.success:
        return IRAResult(
            leaf_id=leaf_id,
            method="fill",
            fill_impact=fill_impact,
            breach_impact=None,
            breach_result=breach_result,
            elevation_changes=fill_changes,
            touched_window_edge=touched,
        )

    breach_impact = _breach_impact_from_result(breach_result)

    # prefer_breach: for a depression a mapped stream flows straight through, the
    # stream defines the real drainage path, so filling the pond up to the barrier
    # would be physically wrong even where it happens to move less material
    if prefer_breach or breach_impact.total_abs_change < fill_impact.total_abs_change:
        breach_changes = {
            cell: dem64[cell] - amount
            for cell, amount in breach_result.excavation.items()
            if amount > 0
        }
        return IRAResult(
            leaf_id=leaf_id,
            method="breach",
            fill_impact=fill_impact,
            breach_impact=breach_impact,
            breach_result=breach_result,
            elevation_changes=breach_changes,
            touched_window_edge=touched,
        )
    else:
        return IRAResult(
            leaf_id=leaf_id,
            method="fill",
            fill_impact=fill_impact,
            breach_impact=breach_impact,
            breach_result=breach_result,
            elevation_changes=fill_changes,
            touched_window_edge=touched,
        )


def apply_ira_to_all_leaves(
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    nodata: float | None = None,
    max_breach_length: int | None = None,
    max_breach_depth: float | None = None,
    leaf_ids: list[int] | None = None,
    stream_mask: np.ndarray | None = None,
    stream_relations: dict[int, str] | None = None,
    progress=None,
    is_canceled=None,
) -> tuple[np.ndarray, list[IRAResult]]:
    """Runs resolve_leaf_with_least_impact for every leaf depression
    in the hierarchy and applies the chosen changes to produce a
    single corrected DEM.

    A real conflict, not just a hypothetical one - found via testing
    on random terrain, not anticipated in advance: a breach search is
    NOT confined to its own leaf's territory. Its cheapest path can
    legitimately cross through cells belonging to a different, nearby
    leaf on the way to an escape (breaching doesn't know or care about
    leaf boundaries, only elevations) - so two leaves resolved
    independently, from the same original DEM, can disagree about
    what a shared cell's final elevation should be.

    Resolved by processing leaves in a deterministic order (ascending
    pit elevation - lower, generally more clearly artifactual pits are
    resolved first) and tracking every cell already committed by an
    earlier leaf's resolution. A later leaf's own resolution is
    computed independently as before, but any of its proposed changes
    that land on an already-committed cell are dropped before
    applying - the earlier leaf's decision for that cell stands. This
    is a deliberately simple, deterministic tie-break (first-resolved-
    wins) rather than re-searching for an alternative, non-conflicting
    path for the later leaf, which would add real complexity for what
    should be a rare situation once classification (not yet built)
    limits resolution to genuinely artifactual depressions rather than
    every leaf in the DEM.

    `leaf_ids` restricts resolution to a chosen subset (e.g. only
    those classification.py labelled Likely Artificial); None keeps
    the original resolve-everything behaviour. For any leaf whose
    `stream_relations` entry is "passes_through" (see
    stream_evidence.py), the breach is guided along the mapped stream
    and breaching is preferred over filling.
    """
    dem64 = dem.astype(np.float64)
    output = dem64.copy()
    committed_cells: set[tuple[int, int]] = set()
    results = []

    if leaf_ids is None:
        leaf_ids = [
            node_id
            for node_id, node in hierarchy.nodes.items()
            if not node.is_ocean and not node.children
        ]
    else:
        leaf_ids = list(leaf_ids)

    windows = NodeWindows(hierarchy)
    pit_elev = _pit_elevations(hierarchy, dem64, leaf_ids)
    leaf_ids.sort(key=lambda i: pit_elev[i])
    shape = dem64.shape

    guide_mask = None
    if stream_mask is not None:
        from .stream_evidence import _dilate

        guide_mask = _dilate(stream_mask, 1)

    import time

    t_start = last_report = time.time()
    total = len(leaf_ids)
    for n_done, node_id in enumerate(leaf_ids):
        if is_canceled is not None and is_canceled():
            break
        now = time.time()
        if progress is not None and now - last_report >= 5.0:
            last_report = now
            progress(
                n_done,
                total,
                f"{n_done} of {total} units resolved "
                f"({now - t_start:.0f} s elapsed)",
            )
        t_unit = time.time()
        through = (stream_relations or {}).get(
            node_id
        ) == "passes_through" and guide_mask is not None
        margin = windows.default_margin(node_id)
        while True:
            win = windows.window(node_id, margin)
            r0, r1, c0, c1 = win
            flags = edge_flags(win, shape)
            result = resolve_leaf_with_least_impact(
                output[r0:r1, c0:c1],
                windows.sub_hierarchy(win),
                node_id,
                nodata=nodata,
                max_breach_length=max_breach_length,
                max_breach_depth=max_breach_depth,
                breach_allowed_mask=guide_mask[r0:r1, c0:c1] if through else None,
                prefer_breach=through,
                breach_edge_flags=flags,
            )
            if not result.touched_window_edge or not any(flags):
                break
            margin = max(margin * 4, 32)
        if progress is not None and time.time() - t_unit >= 10.0:
            progress(
                n_done,
                total,
                f"Slow unit {node_id}: {time.time() - t_unit:.0f} s "
                f"(search window {r1 - r0} x {c1 - c0} cells)",
            )
        if r0 or c0:
            _shift_result(result, r0, c0)
        applied_changes = {}
        for cell, new_elev in result.elevation_changes.items():
            if cell in committed_cells:
                continue  # an earlier leaf already committed this cell - its decision stands
            output[cell] = new_elev
            committed_cells.add(cell)
            applied_changes[cell] = new_elev
        result.elevation_changes = applied_changes
        results.append(result)
    return output, results


def _pit_elevations(hierarchy, dem64, node_ids):
    """Lowest cell elevation of each node (over its descendant leaves)."""
    labels = hierarchy.leaf_label
    n = int(labels.max()) + 1 if labels.size else 0
    leaf_min = np.full(n, np.inf)
    ok = labels >= 0
    np.minimum.at(leaf_min, labels[ok], dem64[ok])
    out = {}
    for nid in node_ids:
        ids = [i for i in hierarchy._descendant_leaf_ids(nid) if i < n]
        out[nid] = float(leaf_min[ids].min()) if ids else float("inf")
    return out


def _shift_result(result, dr, dc):
    """Crop coordinates -> grid coordinates."""
    result.elevation_changes = {
        (r + dr, c + dc): v for (r, c), v in result.elevation_changes.items()
    }
    br = result.breach_result
    if br is not None:
        br.path_cells = [(r + dr, c + dc) for r, c in br.path_cells]
        br.excavation = {(r + dr, c + dc): v for (r, c), v in br.excavation.items()}
        if br.escape_cell is not None:
            br.escape_cell = (br.escape_cell[0] + dr, br.escape_cell[1] + dc)
