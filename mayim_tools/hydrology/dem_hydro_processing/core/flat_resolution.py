"""
Flat-area gradient resolution for filled depressions.

THE PROBLEM THIS SOLVES: a real, independently reported problem, not a
hypothetical one. Chris ran the plugin end-to-end: it filled and
breached depressions correctly (confirmed on a profile comparison),
but flow-direction/accumulation afterward was still short and
disconnected. The actual cause: filling raises every cell of a
depression to one uniform elevation (its spill elevation) - correct
for removing the depression's VOLUME, but it leaves a perfectly flat
plateau with no local elevation gradient anywhere inside it. A
standard D8 (steepest-descent) flow-direction algorithm has no way to
choose a direction on a flat cell - every neighbour is at the same
elevation - so flow accumulation simply cannot pass through it,
breaking the drainage network into short, disconnected pieces exactly
as reported. This is a well-known, well-studied problem in
operational hydrological DEM conditioning (every production tool -
ArcGIS Fill, TauDEM, WhiteboxTools, GRASS r.fill.dir - has a step for
it), not something specific to this project, but this project's own
IRA fill operation had never produced one before this (every synthetic
test DEM used small, few-cell depressions where the resulting flat,
if any, happened not to matter for the tests written) - found from
real data, not anticipated.

THE FIX: Garbrecht, J. and Martz, L.W. (1997), "The assignment of
drainage direction over flat surfaces in raster digital elevation
models," Journal of Hydrology 193(1-4), 204-213 - the standard,
widely-adopted solution, used (in one form or another) by every
production tool listed above. A flat is always bordered by two kinds
of terrain: higher ground (where water flows IN from) and lower
ground (where water flows OUT to). G&M impose two tiny, superimposed
elevation gradients on the flat: one increasing with distance FROM
the higher (inflow) boundary, and one increasing with distance TO the
lower (outflow) boundary. Combined, every flat cell ends up with a
well-defined, monotonically-decreasing path toward the outflow, so
D8 (or any steepest-descent method) can resolve a direction
everywhere - without ever changing the DEM's real hydrology, since
the added increments are tiny.

Implemented here following Barnes, R., Lehman, C. and Mulla, D.
(2014), "An efficient assignment of drainage direction over flat
surfaces in raster digital elevation models," Computers & Geosciences
62, 128-135 - an improved, nonrecursive version of G&M by the SAME
author already cited throughout this project for priority-flood
filling (core/priority_flood.py). Barnes et al.'s own key
improvement, carried over here: the gradient toward lower terrain is
given TWICE the weight of the gradient away from higher terrain,
which the paper shows eliminates the ambiguous ties G&M's original,
equally-weighted combination could still produce - meaning this
reaches a fully-resolved gradient in a single pass, with no need for
G&M's own iterative re-application at ever-smaller increments.

SCOPE: this module is deliberately narrower than Barnes et al.'s own
general-purpose tool, which discovers arbitrary flats anywhere in a
DEM from unknown sources. Here, the "flat" is always exactly known in
advance - it is the set of cells impact_reduction.py's own fill
operation just raised to one uniform elevation - so there is no need
for the general flat-discovery step (their Algorithm 1, Step 1); this
module picks up directly at identifying each flat cell's own distance
to its own group's higher (inflow) and lower (outflow) boundaries.

A GENUINE SIMPLIFICATION, STATED PLAINLY: Barnes et al.'s own
algorithm computes BFS distance using the full 8-connected queue
mechanics described in their paper (with some incidental, harmless
double-processing of boundary cells themselves, visible in their own
pseudocode). This implementation computes the same two underlying
quantities - distance from the inflow boundary, distance to the
outflow boundary - via a direct, standard multi-source BFS, which is
mathematically equivalent for the purpose this module needs (a
consistent, monotonically-ordered gradient) without reproducing their
exact queue/marker bookkeeping. The combination formula itself
(FlatHeight - away_from_higher + 2 x toward_lower) is taken directly
from their own Step 3 (Section 2.4).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .priority_flood import _NEIGHBOR_OFFSETS

# The largest TOTAL elevation distortion this module will ever add to a
# single group, regardless of how many BFS steps its gradient spans -
# see _choose_epsilon's own docstring for why this is computed
# adaptively per group rather than using one fixed per-step increment.
DEFAULT_MAX_TOTAL_DISTORTION_M = 0.01


def _bfs_distance_to_boundary(
    group_mask: np.ndarray, is_boundary_adjacent: np.ndarray
) -> np.ndarray:
    """Multi-source BFS (8-connected, within group_mask only): for
    every group cell, the number of steps to the nearest cell that is
    itself adjacent to the given boundary condition. Boundary-adjacent
    cells themselves get distance 1 (not 0), so there is always a
    genuine, non-zero step between a boundary-adjacent cell and
    whatever lies beyond the group - matching Barnes et al.'s own
    convention (their Figs. 1-2 show the first ring of cells numbered
    1, not 0). Cells never reached (no path within the group to any
    boundary-adjacent cell - only possible for a group with no
    instances of that boundary type at all) are left at 0.
    """
    rows, cols = group_mask.shape
    distance = np.zeros((rows, cols), dtype=np.int64)
    visited = np.zeros((rows, cols), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    seed_rows, seed_cols = np.nonzero(is_boundary_adjacent & group_mask)
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist()):
        distance[r, c] = 1
        visited[r, c] = True
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        d = distance[r, c]
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and group_mask[nr, nc]
                and not visited[nr, nc]
            ):
                visited[nr, nc] = True
                distance[nr, nc] = d + 1
                queue.append((nr, nc))

    return distance


def float32_safe_min_epsilon(
    reference_elevation: float, safety_factor: float = 4.0
) -> float:
    """The smallest per-step increment guaranteed to still be
    distinguishable from `reference_elevation` after the DEM is
    written out as float32 (the precision this project's own output
    uses), with a safety margin.

    A REAL BUG, found from Chris reporting "same result" a fourth
    time after three previous, verified fixes - not a hypothetical
    edge case. The adaptive scaling in _choose_epsilon below was
    ALWAYS intended to respect float32's rounding floor (its own
    docstring said so), but the code never actually checked or
    enforced it - a real gap between the stated reasoning and what was
    implemented. Checked directly: at a realistic elevation (1500 m,
    plausible for South African/Western Australian terrain) and a
    channel/flat spanning a few hundred cells (easily reached once the
    grouped-resolution fix merges a long artificial chain into one
    unit), the un-floored adaptive epsilon shrinks far below float32's
    own representable precision at that magnitude - silently
    collapsing many cells onto the identical rounded value and
    recreating exactly the flat stretches the gradient fix exists to
    remove, with no visible error anywhere. This explains "disconnected
    within flowpaths" persisting unchanged after the gradient fixes
    were already in place: the fix was computing a real gradient in
    float64, then losing it at the point the DEM is actually written.
    Uses numpy's own np.spacing (the authoritative gap to the next
    representable float32 value at a given magnitude) rather than a
    manually reconstructed power-of-two formula, with a safety margin
    to survive further arithmetic a downstream tool might perform.
    """
    step = float(np.spacing(np.float32(reference_elevation)))
    return step * safety_factor


def _choose_epsilon(
    max_combined_increment: int,
    max_total_distortion_m: float,
    reference_elevation: float,
) -> float:
    """The per-step elevation increment, chosen so the LARGEST possible
    total distortion for this specific group (max_combined_increment
    steps) never exceeds max_total_distortion_m - UNLESS that would
    require an increment too small to survive float32 rounding at
    `reference_elevation`'s own magnitude (see
    float32_safe_min_epsilon's own docstring for why this floor is
    enforced, not just reasoned about), in which case the float32-safe
    floor wins instead, even though the resulting total distortion
    then exceeds the originally requested target for a sufficiently
    large group. A working gradient that uses slightly more of the
    DEM's own precision budget is always preferable to a precisely-
    bounded one that silently does nothing.
    """
    if max_combined_increment <= 0:
        return 0.0
    target = max_total_distortion_m / max_combined_increment
    floor = float32_safe_min_epsilon(reference_elevation)
    return max(target, floor)


def resolve_flat_gradient(
    dem: np.ndarray,
    group_mask: np.ndarray,
    fill_elevation: float,
    nodata: float | None = None,
    max_total_distortion_m: float = DEFAULT_MAX_TOTAL_DISTORTION_M,
) -> dict[tuple[int, int], float]:
    """Given a DEM where `group_mask` cells have just been filled to a
    single uniform `fill_elevation`, returns {cell: new_elevation} with
    a tiny gradient superimposed (Garbrecht & Martz 1997, via Barnes
    et al. 2014 - see module docstring) so a steepest-descent flow-
    direction algorithm can resolve a continuous, convergent path from
    the group's inflow boundary to its outflow boundary. Only cells
    whose elevation actually changes are included, matching this
    project's own elevation_changes convention elsewhere
    (impact_reduction.IRAResult).

    Returns an empty dict if the group has no outflow boundary at all
    (every neighbouring cell is higher than fill_elevation) - this
    should not happen for a group that came from a well-formed
    depression hierarchy (every non-ocean node has a real
    spill_elevation reachable from its own footprint by construction),
    but is handled explicitly rather than silently producing a
    gradient that cannot actually drain, consistent with Barnes et
    al.'s own requirement that only flats with a genuine outlet are
    resolved.
    """
    dem64 = dem.astype(np.float64)
    rows, cols = dem64.shape

    is_higher = np.zeros(
        (rows, cols), dtype=bool
    )  # group cells adjacent to a higher (inflow) neighbour
    is_lower_or_equal = np.zeros(
        (rows, cols), dtype=bool
    )  # group cells adjacent to an outflow neighbour

    group_rows, group_cols = np.nonzero(group_mask)
    for r, c in zip(group_rows.tolist(), group_cols.tolist()):
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or group_mask[nr, nc]:
                continue
            neighbor_elev = dem64[nr, nc]
            if nodata is not None and neighbor_elev == nodata:
                continue
            if neighbor_elev > fill_elevation:
                is_higher[r, c] = True
            else:
                is_lower_or_equal[r, c] = True

    if not is_lower_or_equal.any():
        return (
            {}
        )  # no outflow boundary - cannot be resolved into a draining gradient, see docstring

    away_from_higher = _bfs_distance_to_boundary(group_mask, is_higher)
    toward_lower = _bfs_distance_to_boundary(group_mask, is_lower_or_equal)
    flat_height = int(away_from_higher.max())

    combined = np.where(
        away_from_higher > 0,
        (flat_height - away_from_higher) + 2 * toward_lower,
        2 * toward_lower,
    )
    combined = np.where(group_mask, combined, 0)

    max_combined = int(combined.max())
    epsilon = _choose_epsilon(max_combined, max_total_distortion_m, fill_elevation)
    if epsilon == 0.0:
        return {}

    changes: dict[tuple[int, int], float] = {}
    for r, c in zip(group_rows.tolist(), group_cols.tolist()):
        increment = combined[r, c]
        if increment > 0:
            changes[(r, c)] = fill_elevation + epsilon * increment
    return changes
