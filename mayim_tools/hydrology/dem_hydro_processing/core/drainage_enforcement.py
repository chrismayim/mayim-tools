"""Final drainage-enforcement pass (epsilon Priority-Flood, Barnes et al. 2014).

WHY: diagnostics on the corrected DEM (core/diagnostics.py) showed that
per-unit fill/breach can leave cells with no strictly lower 8-neighbour
even when every feature was selected: (a) a breach on a multi-leaf unit
drains only its lowest pit; (b) the fill gradient lifts a cell above a
neighbour that sat at spill level. Both break D8 flow accumulation.

WHAT: flood inwards from legitimate outlets (raster edge, NoData
neighbours, and the pit cell of every PROTECTED depression, which is
allowed to stay a sink). Any cell whose neighbour was reached at an
equal or higher level is raised to nextafter-style epsilon above the
cell it drains to. Cells that already drain are not changed, so this
only touches residual pits and flats.
"""

from __future__ import annotations

import heapq

import numpy as np

from .flat_resolution import float32_safe_min_epsilon
from .priority_flood import _NEIGHBOR_OFFSETS


def enforce_drainage(
    dem: np.ndarray,
    nodata: float | None = None,
    sink_cells: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, int]:
    """Returns (dem with every non-sink cell having a strictly lower
    neighbour path to an outlet, number of cells raised)."""
    z = np.asarray(dem, dtype=np.float64).copy()
    rows, cols = z.shape
    valid = np.isfinite(z) if nodata is None else (np.isfinite(z) & (z != nodata))
    done = np.zeros((rows, cols), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    def seed(r, c):
        if not done[r, c]:
            done[r, c] = True
            heapq.heappush(heap, (float(z[r, c]), r, c))

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                done[r, c] = True
                continue
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            if r in (0, rows - 1) or c in (0, cols - 1):
                seed(r, c)
                continue
            for dr, dc in _NEIGHBOR_OFFSETS:
                if not valid[r + dr, c + dc]:
                    seed(r, c)
                    break
    for r, c in sink_cells or []:
        seed(r, c)

    raised = 0
    # Compare/raise at float32 precision: the corrected DEM is written as float32.
    while heap:
        level, r, c = heapq.heappop(heap)
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or done[nr, nc]:
                continue
            done[nr, nc] = True
            eps = float32_safe_min_epsilon(level)
            if z[nr, nc] <= level + eps * 0.5 or np.float32(z[nr, nc]) <= np.float32(
                level
            ):
                new = level + eps
                if np.float32(new) <= np.float32(level):
                    new = (
                        float(np.nextafter(np.float32(level), np.float32(np.inf))) * 1.0
                        + eps
                    )
                z[nr, nc] = new
                raised += 1
            heapq.heappush(heap, (float(z[nr, nc]), nr, nc))
    return z, raised


def enforce_after_resolution(
    corrected, original_dem, hierarchy, resolve_leaf_ids, nodata=None
):
    """Runs enforce_drainage with the pit cell of every leaf that was NOT
    selected for resolution (i.e. protected / left natural) kept as a
    legitimate sink. Returns (dem, number of cells raised)."""
    selected = set(resolve_leaf_ids)
    orig = np.asarray(original_dem, dtype=np.float64)
    sinks = []
    for nid, node in hierarchy.nodes.items():
        if node.is_ocean or node.children or nid in selected:
            continue
        rr, cc = np.nonzero(hierarchy.cells_of(nid))
        if len(rr):
            k = int(np.argmin(orig[rr, cc]))
            sinks.append((int(rr[k]), int(cc[k])))
    return enforce_drainage(corrected, nodata=nodata, sink_cells=sinks)
