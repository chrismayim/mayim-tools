"""
Mayim Tools - Depression Detection
====================================

Implements the Priority-Flood depression-labelling traversal for
Stage 3 of the Mayim Tools hydrological-conditioning pipeline.

This module identifies every depression in a DEM and assigns each
cell a depression ID. The traversal follows the Priority-Flood
algorithm published in:

    Barnes, R., Lehman, C., and Mulla, D. (2014).
    Priority-flood: An optimal depression-filling and
    watershed-labeling algorithm for digital elevation models.
    Computers & Geosciences, 62, 117-127.

IP Status
---------
Original Mayim implementation.
Written solely from the published paper listed above.
No WhiteboxTools, RichDEM or any other third-party hydrological
source was consulted by the implementing engineer.
Runtime dependencies: numpy only (BSD licence).

Clean-room protocol
-------------------
This module was implemented without opening any external
implementation of Priority-Flood. The algorithm source is the
published paper only. No third-party source was open in the
implementing engineer's environment during development.
"""

from __future__ import annotations

import heapq

import numpy as np

# Eight-connected neighbour offsets (D8).
_D8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def detect_depressions(
    dem: np.ndarray,
    nodata: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Identify every depression in a DEM and label each depressed cell
    with a unique integer depression ID.

    Uses a Priority-Flood-style boundary-inward traversal following
    Barnes, Lehman and Mulla (2014). Every cell is visited in
    non-decreasing elevation order. A cell whose elevation is lower
    than the current flood level is a depression cell; it is assigned
    the ID of the depression that contains it.

    A separate pit-cell array records the single lowest cell of each
    depression (the pit), used by the hierarchy builder to determine
    spill relationships.

    Parameters
    ----------
    dem : np.ndarray
        2-D float64 array of elevation values.
    nodata : float
        Sentinel value for invalid cells. These cells are skipped.

    Returns
    -------
    depression_ids : np.ndarray
        Integer array of the same shape as dem.
        0  = valid non-depressed cell.
        >0 = depression ID (1-indexed).
        -1 = NoData cell.
    pit_cells : np.ndarray
        Boolean array. True where a cell is the lowest point
        (pit) of its depression.
    n_depressions : int
        Total number of distinct depressions detected.

    References
    ----------
    Barnes, R., Lehman, C., and Mulla, D. (2014).
    Priority-flood: An optimal depression-filling and
    watershed-labeling algorithm for digital elevation models.
    Computers & Geosciences, 62, 117-127.
    """
    rows, cols = dem.shape
    valid = (dem != nodata) & np.isfinite(dem)

    depression_ids = np.zeros((rows, cols), dtype=np.int32)
    depression_ids[~valid] = -1

    pit_cells = np.zeros((rows, cols), dtype=bool)
    visited = np.zeros((rows, cols), dtype=bool)

    # flood_level tracks the water surface elevation at each cell
    # as the Priority-Flood traversal proceeds.
    flood_level = np.full((rows, cols), np.inf, dtype=np.float64)

    heap = []

    # ── Seed the heap with all boundary cells ─────────────────────── #
    for r in range(rows):
        for c in [0, cols - 1]:
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True
                flood_level[r, c] = dem[r, c]
                heapq.heappush(heap, (dem[r, c], r, c))

    for c in range(cols):
        for r in [0, rows - 1]:
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True
                flood_level[r, c] = dem[r, c]
                heapq.heappush(heap, (dem[r, c], r, c))

    next_id = 1

    # ── Priority-Flood traversal ───────────────────────────────────── #
    while heap:
        z, r, c = heapq.heappop(heap)

        for dr, dc in _D8:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if visited[nr, nc] or not valid[nr, nc]:
                continue

            visited[nr, nc] = True

            if dem[nr, nc] >= z:
                # Neighbour is at or above the current flood level.
                # It drains normally — not a depression cell.
                flood_level[nr, nc] = dem[nr, nc]
                heapq.heappush(heap, (dem[nr, nc], nr, nc))
            else:
                # Neighbour is below the current flood level.
                # It is a depression cell — it cannot drain without
                # rising to at least z.
                flood_level[nr, nc] = z

                # Assign a new depression ID if this is a new pit.
                if depression_ids[nr, nc] == 0:
                    # Check whether this cell is a local pit
                    # (all valid neighbours are higher or equal).
                    is_pit = True
                    for dr2, dc2 in _D8:
                        nr2, nc2 = nr + dr2, nc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and valid[nr2, nc2]
                            and dem[nr2, nc2] < dem[nr, nc]
                        ):
                            is_pit = False
                            break

                    if is_pit:
                        depression_ids[nr, nc] = next_id
                        pit_cells[nr, nc] = True
                        next_id += 1
                    else:
                        # Propagate the ID from the upstream neighbour
                        # that led here.
                        depression_ids[nr, nc] = depression_ids[r, c]

                heapq.heappush(heap, (flood_level[nr, nc], nr, nc))

    n_depressions = next_id - 1
    return depression_ids, pit_cells, n_depressions


def identify_spill_points(
    dem: np.ndarray,
    depression_ids: np.ndarray,
    nodata: float,
) -> dict[int, float]:
    """
    Determine the spill elevation for each depression.

    The spill elevation is the lowest elevation at which water can
    escape from a depression into a non-depressed or differently
    labelled neighbouring cell.

    Parameters
    ----------
    dem : np.ndarray
        2-D float64 elevation array.
    depression_ids : np.ndarray
        Depression ID array from detect_depressions().
    nodata : float
        Sentinel value for invalid cells.

    Returns
    -------
    dict[int, float]
        Mapping from depression_id to spill elevation.
    """
    rows, cols = dem.shape
    spill: dict[int, float] = {}

    for r in range(rows):
        for c in range(cols):
            did = depression_ids[r, c]
            if did <= 0:
                continue

            for dr, dc in _D8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if dem[nr, nc] == nodata or not np.isfinite(dem[nr, nc]):
                    continue
                if depression_ids[nr, nc] == did:
                    continue

                # This neighbour is outside the depression.
                # The spill elevation is the higher of the two cells.
                candidate = max(dem[r, c], dem[nr, nc])
                if did not in spill or candidate < spill[did]:
                    spill[did] = candidate

    return spill
