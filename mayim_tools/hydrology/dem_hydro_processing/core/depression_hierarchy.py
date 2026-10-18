"""
Depression hierarchy construction.

Concept follows the published, peer-reviewed approach described in:
    Barnes, R., Callaghan, K.L., Wickert, A.D. (2020). "Computing
    water flow through complex landscapes - Part 2: Finding
    hierarchies in depressions and morphological segmentations."
    Earth Surface Dynamics, 8, 431-445.
    https://esurf.copernicus.org/articles/8/431/2020/

    Also closely related to earlier work on the same underlying
    problem: Wang & Liu (2006), Barnes, Lehman & Mulla (2014)
    (priority-flood, see core/priority_flood.py), and the general
    "watershed transform" concept (Vincent & Soille, 1991) from image
    processing, which this shares its core mechanism with.

Implemented independently, from the published algorithmic description
of what a depression hierarchy IS and the general watershed-transform/
union-find flood-merge mechanism that constructs one - not from any
existing tool's source code. The specific implementation choices below
(the exact union-find bookkeeping, the same-elevation local-minima
grouping) are this project's own, not copied from Barnes et al.'s own
implementation, which was not consulted.

WHAT A DEPRESSION HIERARCHY IS: every local minimum in a DEM is a
"leaf" depression. As water notionally rises, adjacent depressions
eventually merge into a combined "meta-depression" at the elevation
where they first spill into each other (their shared "spill point").
This continues until everything has merged into "the ocean" - the
DEM's own edge (or NoData boundary - see priority_flood.py for why
that's treated the same way). The result is a forest of trees: leaves
are simple depressions, internal nodes are meta-depressions, and the
root is the ocean. This structure is what lets later processing make
a SEPARATE decision for every individual depression (or meta-
depression) - e.g. "is this one likely an artefact?" - rather than
only being able to apply one blanket rule to the whole DEM.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .priority_flood import _NEIGHBOR_OFFSETS


class _UnionFind:
    """Standard union-find (disjoint set) with path compression and
    union by rank - tracks, at any point during the flood, which
    top-level merged region a given original leaf id currently
    belongs to."""

    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return ra


@dataclass
class DepressionNode:
    """One node in the depression hierarchy - either a leaf (a single
    simple depression, i.e. one local minimum and the cells that
    drain to it before any merge), a meta-depression formed by a
    binary merge of exactly two children, or the single special
    "ocean" node (id 0) representing the DEM's own edge/NoData
    boundary.

    spill_elevation/spill_cell describe the elevation and location at
    which THIS node merges into its own parent - not the parent's own
    formation. This is a deliberately uniform choice: an ordinary
    binary meta-depression merge sets it on BOTH children at once
    (they spill into each other at the same moment by definition), and
    a depression draining directly to the ocean sets it the same way,
    without needing the ocean to be restricted to exactly two children
    like an ordinary meta-depression - the ocean is the DEM's edge, so
    an arbitrary number of separate depressions can each independently
    drain to it at their own elevations, and forcing that through
    successive binary merges would misrepresent the geometry (it would
    imply those depressions merge with EACH OTHER before draining,
    which usually is not what happens)."""

    id: int
    is_ocean: bool = False
    children: list[int] = field(default_factory=list)
    parent: int | None = None
    spill_elevation: float | None = None
    spill_cell: tuple[int, int] | None = None


@dataclass
class DepressionHierarchy:
    nodes: dict[int, DepressionNode]
    # per-cell record of which LEAF node originally claimed that cell
    # (assigned once, at first-visit time during the flood, and never
    # reassigned afterward - a cell's leaf membership is a fixed
    # geometric fact, even though leaves go on to merge into larger
    # meta-depressions above them in the tree)
    leaf_label: np.ndarray
    filled_dem: np.ndarray

    def cells_of(self, node_id: int) -> np.ndarray:
        """Boolean mask of every cell belonging to `node_id` - for a
        leaf, cells whose leaf_label matches directly; for a
        meta-depression, the union of all its descendant leaves'
        cells (computed on demand by tree traversal, rather than
        stored redundantly per node)."""
        leaf_ids = self._descendant_leaf_ids(node_id)
        if not leaf_ids:
            return np.zeros(self.leaf_label.shape, dtype=bool)
        return np.isin(self.leaf_label, list(leaf_ids))

    def footprint_of(self, node_id: int, dem: np.ndarray) -> np.ndarray:
        """Boolean mask of the depression's actual FOOTPRINT: the cells
        of `node_id` lying strictly below its own spill elevation - the
        extent that would hold water.

        This is deliberately NOT the same as cells_of(). cells_of()
        returns every cell first CLAIMED by the node's flood front,
        which behaves like the depression's catchment: it includes
        cells above the spill elevation (a hillside draining toward
        the depression, or the whole rising valley upstream of a
        pond). Found the hard way: the size signal, the stream
        relation and the persistence control surface all initially used
        cells_of() as if it were the footprint, so a 9-cell pond
        counted as 36 cells and a stream on the hillside above it
        counted as being inside it. Anything asking "how big is this
        depression" or "is this thing inside it" should use this.
        """
        node = self.nodes[node_id]
        if node.is_ocean or node.spill_elevation is None:
            return np.zeros(self.leaf_label.shape, dtype=bool)
        return self.cells_of(node_id) & (
            dem.astype(np.float64) < node.spill_elevation - 1e-12
        )

    def _descendant_leaf_ids(self, node_id: int) -> set[int]:
        """Iterative (explicit stack), NOT recursive - found the hard
        way (a real crash report from real QGIS, not anticipated by
        any test here): on real, noisy, basin-scale terrain, the
        hierarchy is typically a long CHAIN of sequential binary
        merges (pits merge with their immediate neighbour roughly one
        at a time as water rises), not a balanced tree - so a
        recursive descent from a high-level node can be O(number of
        leaves) stack frames deep, not O(log leaves). Every synthetic
        test DEM built for this project, before that crash report, had
        at most a few hundred leaves - nowhere near enough to expose
        this; a direct test with 3000 leaves in a deliberately chain-
        shaped hierarchy reproduces a clean RecursionError in this
        sandbox's own Python (well below the thousands of leaves a
        real noisy DEM can easily produce - see the project's own
        performance notes, which already mention DEMs with over 4000
        candidate leaves), and on a QGIS worker thread - plausibly
        running with a smaller C stack than this sandbox's main
        thread - the same excessive recursion exhausted the actual
        stack before Python's own recursion-depth guard could even
        trigger, producing a hard access-violation crash rather than a
        catchable exception. An iterative implementation has no
        recursion depth at all, regardless of hierarchy shape."""
        result: set[int] = set()
        stack = [node_id]
        while stack:
            current_id = stack.pop()
            node = self.nodes[current_id]
            if not node.children:
                if not node.is_ocean:
                    result.add(current_id)
            else:
                stack.extend(node.children)
        return result


def _find_leaf_seeds(dem64: np.ndarray, is_nodata: np.ndarray) -> np.ndarray:
    """Returns an integer label array (0 where unlabeled) identifying
    every local minimum, with connected same-elevation local-minimum
    cells grouped into a single shared leaf seed (a flat basin floor
    is one depression, not one per cell). Leaf ids start at 1; id 0 is
    reserved for the ocean and for not-yet-labeled cells."""
    rows, cols = dem64.shape
    is_local_min = np.ones((rows, cols), dtype=bool)
    is_local_min[is_nodata] = False

    padded = np.pad(dem64, 1, mode="constant", constant_values=np.inf)
    padded_nodata = np.pad(is_nodata, 1, mode="constant", constant_values=True)
    for dr, dc in _NEIGHBOR_OFFSETS:
        neighbor_elev = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        neighbor_is_nodata = padded_nodata[
            1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols
        ]
        # a NoData neighbour never disqualifies a cell from being a local minimum -
        # only valid, LOWER neighbours do
        violates = (~neighbor_is_nodata) & (neighbor_elev < dem64)
        is_local_min &= ~violates

    labels = np.zeros((rows, cols), dtype=np.int64)
    next_id = 1
    visited_candidate = np.zeros((rows, cols), dtype=bool)
    cand_rows, cand_cols = np.nonzero(is_local_min)
    for r0, c0 in zip(cand_rows.tolist(), cand_cols.tolist()):
        if visited_candidate[r0, c0]:
            continue
        # flood-fill this connected, same-elevation group of local-minimum candidates
        elev0 = dem64[r0, c0]
        stack = [(r0, c0)]
        visited_candidate[r0, c0] = True
        group_cells = []
        while stack:
            r, c = stack.pop()
            group_cells.append((r, c))
            for dr, dc in _NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and is_local_min[nr, nc]
                    and not visited_candidate[nr, nc]
                ):
                    if dem64[nr, nc] == elev0:
                        visited_candidate[nr, nc] = True
                        stack.append((nr, nc))
        for r, c in group_cells:
            labels[r, c] = next_id
        next_id += 1

    return labels


def build_depression_hierarchy(
    dem: np.ndarray, nodata: float | None = None
) -> DepressionHierarchy:
    """Builds the full depression hierarchy for `dem`. See module
    docstring for the concept and its published basis."""
    rows, cols = dem.shape
    dem64 = dem.astype(np.float64)
    is_nodata = (
        (dem64 == nodata) if nodata is not None else np.zeros((rows, cols), dtype=bool)
    )

    leaf_seed_labels = _find_leaf_seeds(dem64, is_nodata)

    # ocean seeding: identical rule to priority_flood_fill - border
    # cells and any cell adjacent to NoData
    ocean_seed_mask = np.zeros((rows, cols), dtype=bool)
    if rows > 0 and cols > 0:
        ocean_seed_mask[0, :] = True
        ocean_seed_mask[-1, :] = True
        ocean_seed_mask[:, 0] = True
        ocean_seed_mask[:, -1] = True
    if nodata is not None:
        padded = np.pad(is_nodata, 1, mode="constant", constant_values=False)
        for dr, dc in _NEIGHBOR_OFFSETS:
            ocean_seed_mask |= padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    ocean_seed_mask &= ~is_nodata

    OCEAN_ID = 0
    leaf_seed_labels = leaf_seed_labels.copy()
    leaf_seed_labels[ocean_seed_mask] = (
        OCEAN_ID  # ocean membership takes priority over an edge-adjacent local min
    )

    max_leaf_id = int(leaf_seed_labels.max())
    # a local-minimum id allocated by _find_leaf_seeds can end up with
    # ZERO remaining cells if every one of its cells was overwritten
    # to OCEAN_ID above (a local minimum sitting directly on the
    # border, or entirely surrounded by NoData-adjacent cells) - such
    # an id was never really a leaf at all and must not get a
    # DepressionNode of its own, or it would sit in the hierarchy
    # forever unvisited and parentless.
    ids_with_cells = set(np.unique(leaf_seed_labels).tolist())
    uf = _UnionFind(max_leaf_id + 1)  # ids 0..max_leaf_id

    origin_leaf = np.full((rows, cols), -1, dtype=np.int64)  # -1 = not yet visited
    filled = dem64.copy()
    visited = np.zeros((rows, cols), dtype=bool)
    # `visited` means "claimed by some region" (set the moment a cell is
    # seeded or discovered - prevents it being reclaimed by another
    # region). `resolved` means "actually processed in true elevation
    # order" (set only when a cell is popped from the queue). These
    # must be distinguished for correct merge detection - see this
    # function's own note below at the point that distinction matters.
    resolved = np.zeros((rows, cols), dtype=bool)

    open_heap: list[tuple[float, int, int, int]] = []
    pit_queue: deque[tuple[int, int]] = deque()
    counter = 0

    seed_rows, seed_cols = np.nonzero(leaf_seed_labels > 0)
    ocean_rows, ocean_cols = np.nonzero(ocean_seed_mask)
    all_seed_r = np.concatenate([seed_rows, ocean_rows])
    all_seed_c = np.concatenate([seed_cols, ocean_cols])
    seen = set()
    for r, c in zip(all_seed_r.tolist(), all_seed_c.tolist()):
        if (r, c) in seen:
            continue
        seen.add((r, c))
        visited[r, c] = True
        origin_leaf[r, c] = leaf_seed_labels[r, c]
        heapq.heappush(open_heap, (float(filled[r, c]), counter, r, c))
        counter += 1

    next_node_id = (
        max_leaf_id + 1
    )  # meta-depression ids continue after every leaf id (and after the ocean's id 0)
    nodes: dict[int, DepressionNode] = {
        i: DepressionNode(id=i, is_ocean=(i == OCEAN_ID)) for i in ids_with_cells
    }
    # current "live" representative for each original leaf/ocean id -> the
    # hierarchy node id currently representing it (starts as itself;
    # updated to the newest merged parent's id each time it merges)
    current_repr = {i: i for i in ids_with_cells}

    while open_heap or pit_queue:
        if pit_queue:
            r, c = pit_queue.popleft()
        else:
            _elev, _tie, r, c = heapq.heappop(open_heap)
        resolved[r, c] = True

        base_elev = filled[r, c]
        this_origin = origin_leaf[r, c]

        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or is_nodata[nr, nc]:
                continue
            if not visited[nr, nc]:
                visited[nr, nc] = True
                origin_leaf[nr, nc] = this_origin
                neighbor_elev = (
                    dem64[nr, nc] if dem64[nr, nc] > base_elev else base_elev
                )
                filled[nr, nc] = neighbor_elev
                if neighbor_elev == base_elev:
                    pit_queue.append((nr, nc))
                else:
                    counter += 1
                    heapq.heappush(open_heap, (float(neighbor_elev), counter, nr, nc))
            elif resolved[nr, nc]:
                # A merge is only valid against a neighbour that has
                # genuinely been processed in true elevation order, not
                # merely seeded/claimed. A seed cell (e.g. a high-
                # elevation border cell) is marked `visited` immediately
                # at setup so it can't be reclaimed, but it may still be
                # sitting unprocessed in the queue - treating that as an
                # already-reached boundary would let a low-elevation
                # region "merge" with it prematurely, before genuinely
                # closer, lower paths have had a chance to be explored
                # first (a real bug this exact check was added to fix -
                # confirmed by comparing against priority_flood_fill's
                # own, separately-verified output on real terrain).
                # An unresolved neighbour is simply skipped here; the
                # merge will still be detected correctly later, from the
                # other side, once that neighbour's own region reaches
                # this boundary in its proper turn.
                other_origin = origin_leaf[nr, nc]
                rep_a = uf.find(this_origin)
                rep_b = uf.find(other_origin)
                if rep_a != rep_b:
                    merged_uf_id = uf.union(rep_a, rep_b)
                    child_a_node = current_repr[rep_a]
                    child_b_node = current_repr[rep_b]
                    spill_elev = float(max(filled[r, c], filled[nr, nc]))
                    spill_loc = (r, c)

                    if child_a_node == OCEAN_ID or child_b_node == OCEAN_ID:
                        # the ocean is not an ordinary binary parent - see
                        # DepressionNode's own docstring for why. The
                        # non-ocean side becomes a direct child of the
                        # existing ocean node; no new node is created.
                        draining_node = (
                            child_b_node if child_a_node == OCEAN_ID else child_a_node
                        )
                        nodes[draining_node].parent = OCEAN_ID
                        nodes[draining_node].spill_elevation = spill_elev
                        nodes[draining_node].spill_cell = spill_loc
                        nodes[OCEAN_ID].children.append(draining_node)
                        current_repr[rep_a] = OCEAN_ID
                        current_repr[rep_b] = OCEAN_ID
                        current_repr[merged_uf_id] = OCEAN_ID
                    else:
                        new_id = next_node_id
                        next_node_id += 1
                        nodes[new_id] = DepressionNode(
                            id=new_id,
                            is_ocean=False,
                            children=[child_a_node, child_b_node],
                        )
                        nodes[child_a_node].parent = new_id
                        nodes[child_a_node].spill_elevation = spill_elev
                        nodes[child_a_node].spill_cell = spill_loc
                        nodes[child_b_node].parent = new_id
                        nodes[child_b_node].spill_elevation = spill_elev
                        nodes[child_b_node].spill_cell = spill_loc
                        current_repr[rep_a] = new_id
                        current_repr[rep_b] = new_id
                        current_repr[merged_uf_id] = new_id

    filled = _correct_filled_elevations_for_seeded_minima(
        filled, origin_leaf, nodes, OCEAN_ID
    )

    return DepressionHierarchy(nodes=nodes, leaf_label=origin_leaf, filled_dem=filled)


def _correct_filled_elevations_for_seeded_minima(
    filled: np.ndarray,
    origin_leaf: np.ndarray,
    nodes: dict[int, DepressionNode],
    ocean_id: int,
) -> np.ndarray:
    """Fixes a real gap left by seeding local minima directly into the
    flood (necessary so their flood-fronts can be tracked separately
    for merge detection - see build_depression_hierarchy): a cell's
    filled elevation, once first visited, is never revisited even
    after its region goes on to merge with something higher. For a
    normal (non-seed) cell this is fine, exactly matching
    priority_flood_fill's own behaviour - by the time the flood
    reaches it, the elevation arriving from its neighbour already
    reflects everything relevant. But a seeded local-minimum cell's
    own value is locked in at its raw, unfilled elevation from the
    very start, before any of that region's later merges are known.

    The correct final elevation for a leaf's cells is the elevation at
    which that leaf's region ultimately reaches the ocean - i.e. the
    spill_elevation of the top-level ancestor immediately below the
    ocean in its parent chain. This must be at least as high as any
    lower spill in that chain, since elevations only increase moving
    up the hierarchy (each merge happens no earlier, in flood order,
    than the ones before it) - a cell without a non-descending path at
    that final elevation would not actually have one to the edge at
    all, which is the entire defining property of a filled DEM (see
    priority_flood.py's own module docstring).

    Applied as one vectorised correction after the flood completes,
    rather than reworked into the flood's own per-cell logic, which
    would have made an already-intricate merge-tracking loop harder to
    follow for a benefit only needed by a small minority of cells (the
    seeded local minima and whatever was visited before their region's
    later merges were known)."""
    # walk each leaf's parent chain once (far cheaper than doing it per-cell)
    # to find the elevation at which it ultimately reaches the ocean
    ultimate_elevation: dict[int, float] = {}
    for leaf_id, node in nodes.items():
        if node.is_ocean:
            continue
        current = node
        while current.parent is not None and not nodes[current.parent].is_ocean:
            current = nodes[current.parent]
        # `current` is now the top-level node whose parent IS the ocean (or,
        # if this leaf itself already drains directly to ocean, `current` is
        # the leaf itself) - either way its own spill_elevation is correct
        ultimate_elevation[leaf_id] = (
            current.spill_elevation
            if current.spill_elevation is not None
            else float(filled.max())
        )

    max_leaf_id = int(origin_leaf.max())
    lookup = np.zeros(max_leaf_id + 1, dtype=np.float64)
    for leaf_id, elev in ultimate_elevation.items():
        if 0 <= leaf_id <= max_leaf_id:
            lookup[leaf_id] = elev

    non_ocean_mask = origin_leaf != ocean_id
    clipped_origin = np.clip(origin_leaf, 0, max_leaf_id)
    ultimate_per_cell = lookup[clipped_origin]
    corrected = filled.copy()
    corrected[non_ocean_mask] = np.maximum(
        filled[non_ocean_mask], ultimate_per_cell[non_ocean_mask]
    )
    return corrected
