"""
Priority-Flood depression filling.

Implements the algorithm described in:
    Barnes, R., Lehman, C., Mulla, D. (2014). "Priority-Flood: An
    Optimal Depression-Filling and Watershed-Labeling Algorithm for
    Digital Elevation Models." Computers & Geosciences, 62, 117-127.
    https://arxiv.org/pdf/1511.04463

Implemented independently from this published algorithm description,
not from any existing tool's source code - see dev/README.md for the
full rationale (this project deliberately avoids depending on or
adapting any other GIS tool's implementation).

ALGORITHM (the "improved" variant from the paper, using a secondary
plain queue to avoid unnecessary O(log n) heap operations for flat/
plateau cells): flood the DEM inward from its edges (and from the
edges of any NoData region, which are treated the same way - a valid
cell bordering NoData is assumed able to drain off the known data,
the standard, conservative convention also used by other hydrological
tools) using a priority queue ordered by elevation. Each cell, once
reached, is assigned a "filled" elevation equal to the greater of its
own original elevation and the filled elevation of whichever
already-resolved neighbour first reached it. This guarantees every
cell a non-descending path to the domain's edge - the defining
property of a depression-free DEM.

The result is an OPTIMAL fill (Barnes et al. prove this): no cell is
raised any higher than the minimum necessary to guarantee drainage.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np

# 8-connected (Moore) neighbourhood - standard for D8-style hydrological
# terrain analysis throughout this whole plugin suite
_NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def priority_flood_fill(dem: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Returns a filled copy of `dem` (same shape, dtype float64) in
    which every cell has a non-descending path to the domain's edge.

    `dem` is a 2D array. `nodata` cells (if given) are never visited
    or modified, and a valid cell adjacent to a NoData cell is treated
    as a domain edge for seeding purposes (see module docstring).

    A cell that cannot reach any edge at all (e.g. a valid region
    fully enclosed by NoData with no other outlet) is left at its
    original elevation, unfilled and unvisited - this is a genuine
    edge case (a landlocked island of valid data) rather than
    something this function should guess how to resolve; the caller
    can detect it via the returned `visited` mask if needed (see
    priority_flood_fill_verbose).

    In practice, given the seeding rule above (a valid cell adjacent
    to NoData is itself treated as a drainage point), this case
    cannot actually arise: any connected region of valid cells
    necessarily has its own boundary touching either NoData or the
    array edge, so it is always reachable. The `visited` mask and this
    docstring note are kept because that seeding rule is a specific,
    documented design choice (matching how a clipped DEM's boundary is
    conventionally treated elsewhere), not an accident of the
    implementation - if that convention is ever revisited, this is
    exactly the behaviour that would change.
    """
    filled, _visited = priority_flood_fill_verbose(dem, nodata)
    return filled


def priority_flood_fill_verbose(
    dem: np.ndarray, nodata: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Same as priority_flood_fill, but also returns the boolean
    `visited` mask (True for every cell that was reached and
    processed) - a cell that is valid (not NoData) but was never
    visited indicates a landlocked region with no path to any edge,
    worth being able to detect explicitly rather than silently.
    """
    rows, cols = dem.shape
    dem64 = dem.astype(np.float64)
    filled = dem64.copy()
    visited = np.zeros((rows, cols), dtype=bool)

    is_nodata = (
        (dem64 == nodata) if nodata is not None else np.zeros((rows, cols), dtype=bool)
    )

    open_heap: list[tuple[float, int, int, int]] = []
    pit_queue: deque[tuple[int, int]] = deque()
    counter = 0  # heap tie-breaker - keeps heapq's comparisons well-defined and stable

    # Seed: every valid cell on the array border, plus every valid
    # cell adjacent to a NoData cell (see module docstring for why).
    seed_mask = np.zeros((rows, cols), dtype=bool)
    if rows > 0 and cols > 0:
        seed_mask[0, :] = True
        seed_mask[-1, :] = True
        seed_mask[:, 0] = True
        seed_mask[:, -1] = True
    if nodata is not None:
        # a valid cell is a NoData-adjacent seed if any of its 8 neighbours is NoData
        padded = np.pad(is_nodata, 1, mode="constant", constant_values=False)
        for dr, dc in _NEIGHBOR_OFFSETS:
            seed_mask |= padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    seed_mask &= ~is_nodata

    seed_rows, seed_cols = np.nonzero(seed_mask)
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist()):
        visited[r, c] = True
        heapq.heappush(open_heap, (float(filled[r, c]), counter, r, c))
        counter += 1

    while open_heap or pit_queue:
        if pit_queue:
            r, c = pit_queue.popleft()
        else:
            _elev, _tie, r, c = heapq.heappop(open_heap)

        base_elev = filled[r, c]
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc] or is_nodata[nr, nc]:
                continue
            visited[nr, nc] = True
            neighbor_elev = dem64[nr, nc] if dem64[nr, nc] > base_elev else base_elev
            filled[nr, nc] = neighbor_elev
            if neighbor_elev == base_elev:
                pit_queue.append((nr, nc))
            else:
                counter += 1
                heapq.heappush(open_heap, (float(neighbor_elev), counter, nr, nc))

    return filled, visited
