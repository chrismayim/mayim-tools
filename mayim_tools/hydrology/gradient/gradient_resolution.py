"""
Mayim Tools - Garbrecht-Martz Flat Resolution
==============================================

Implements Stage 6 flat-area gradient resolution.

The Garbrecht-Martz method imposes two superimposed synthetic gradients
across each flat surface:

    Gradient A:
        Pushes flow away from higher terrain at the flat inflow boundary.
        Computed as BFS distance from higher-boundary cells.

    Gradient B:
        Pulls flow toward the flat outflow point.
        Computed as BFS distance from lower-boundary cells.

    Combined correction:
        correction = (2 * gradient_B + gradient_A) * step

    where step is scaled so the imposed gradient stays below the DEM's
    vertical resolution.

This produces an unambiguous downhill gradient across the flat without
modifying terrain outside the flat footprint.

Methodology basis
-----------------
Garbrecht, J. and Martz, L. W. (1997).
The assignment of drainage direction over flat surfaces in raster
digital elevation models.
Journal of Hydrology, 193(1-4), 204-213.

IP status
---------
Original Mayim implementation based on the published paper.
No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
implementation is used at runtime.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def resolve_flats(
    dem: np.ndarray,
    flat_mask: np.ndarray,
    higher_boundary: np.ndarray,
    lower_boundary: np.ndarray,
    cell_size: float,
    vertical_accuracy: float,
    nodata: float,
) -> tuple[np.ndarray, dict]:
    """
    Apply the Garbrecht-Martz dual-gradient method to flat cells.

    Only cells inside flat_mask are modified. All other cells are
    copied unchanged from the input DEM.

    The imposed gradient step is:

        step = min(vertical_accuracy, cell_size) / (
            2.0 * max(gradient_away.max(), gradient_toward.max(), 1)
        )

    This ensures the imposed signal stays below the DEM's vertical
    resolution and does not introduce a slope larger than the data
    can support.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    flat_mask:
        Boolean mask identifying flat cells.
    higher_boundary:
        Boolean mask identifying flat cells adjacent to higher terrain.
    lower_boundary:
        Boolean mask identifying flat cells adjacent to lower terrain.
    cell_size:
        Mean cell size in map units.
    vertical_accuracy:
        DEM vertical accuracy in metres.
    nodata:
        NoData sentinel value.

    Returns
    -------
    tuple[np.ndarray, dict]
        Gradient-resolved DEM and audit dictionary.

    Raises
    ------
    ValueError
        If inputs are invalid or the flat has no valid outlet.
    """
    _validate_inputs(
        dem=dem,
        flat_mask=flat_mask,
        higher_boundary=higher_boundary,
        lower_boundary=lower_boundary,
        cell_size=cell_size,
        vertical_accuracy=vertical_accuracy,
    )

    result = dem.astype(np.float64, copy=True)

    if not np.any(flat_mask):
        return result, _empty_audit()

    if not np.any(lower_boundary):
        raise ValueError(
            "The flat surface has no valid lower boundary. "
            "It cannot be resolved without a drainage outlet."
        )

    gradient_away = _bfs_distance(
        flat_mask=flat_mask,
        seeds=higher_boundary,
    )

    gradient_toward = _bfs_distance(
        flat_mask=flat_mask,
        seeds=lower_boundary,
    )

    max_away = float(gradient_away[flat_mask].max()) if np.any(flat_mask) else 1.0
    max_toward = float(gradient_toward[flat_mask].max()) if np.any(flat_mask) else 1.0
    max_distance = max(max_away, max_toward, 1.0)

    step = min(
        float(vertical_accuracy),
        float(cell_size),
    ) / (2.0 * max_distance)

    correction = (
        2.0 * gradient_toward
        + gradient_away
    ) * step

    result[flat_mask] = (
        dem[flat_mask].astype(np.float64)
        + correction[flat_mask]
    )

    modified_cells = int(np.sum(flat_mask))
    total_change = float(
        np.sum(result[flat_mask] - dem[flat_mask])
    )
    maximum_change = float(
        np.max(result[flat_mask] - dem[flat_mask])
    )

    audit = {
        "method": "garbrecht_martz_flat_resolution",
        "flat_cells": modified_cells,
        "higher_boundary_cells": int(np.sum(higher_boundary)),
        "lower_boundary_cells": int(np.sum(lower_boundary)),
        "step": float(step),
        "max_gradient_away": float(max_away),
        "max_gradient_toward": float(max_toward),
        "total_elevation_change": total_change,
        "maximum_elevation_change": maximum_change,
        "modified_cells": modified_cells,
    }

    return result, audit


def _bfs_distance(
    flat_mask: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """
    Compute BFS distances from seed cells within the flat mask.

    Non-flat cells and unvisited flat cells receive a distance of zero.

    :param flat_mask: Boolean flat-cell mask.
    :param seeds: Boolean seed-cell mask.
    :returns: Integer distance array.
    """
    rows, cols = flat_mask.shape
    distance = np.zeros((rows, cols), dtype=np.int32)
    visited = np.zeros((rows, cols), dtype=bool)

    queue = deque()

    seed_positions = np.argwhere(seeds & flat_mask)

    for position in seed_positions:
        row, col = int(position[0]), int(position[1])
        visited[row, col] = True
        distance[row, col] = 1
        queue.append((row, col))

    while queue:
        row, col = queue.popleft()

        for row_offset, col_offset in [
            (-1, 0), (1, 0), (0, -1), (0, 1),
        ]:
            nr = row + row_offset
            nc = col + col_offset

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if visited[nr, nc] or not flat_mask[nr, nc]:
                continue

            visited[nr, nc] = True
            distance[nr, nc] = distance[row, col] + 1
            queue.append((nr, nc))

    return distance

def _validate_inputs(
    dem: np.ndarray,
    flat_mask: np.ndarray,
    higher_boundary: np.ndarray,
    lower_boundary: np.ndarray,
    cell_size: float,
    vertical_accuracy: float,
) -> None:
    """
    Validate gradient-resolution inputs.

    :raises ValueError: If any input is invalid.
    """
    if not isinstance(dem, np.ndarray) or dem.ndim != 2:
        raise ValueError(
            "dem must be a two-dimensional NumPy array."
        )

    if not isinstance(flat_mask, np.ndarray):
        raise ValueError(
            "flat_mask must be a NumPy array."
        )

    if flat_mask.shape != dem.shape:
        raise ValueError(
            "flat_mask must have the same shape as dem."
        )

    if flat_mask.dtype != np.bool_:
        raise ValueError(
            "flat_mask must have Boolean dtype."
        )

    if not isinstance(higher_boundary, np.ndarray):
        raise ValueError(
            "higher_boundary must be a NumPy array."
        )

    if higher_boundary.shape != dem.shape:
        raise ValueError(
            "higher_boundary must have the same shape as dem."
        )

    if not isinstance(lower_boundary, np.ndarray):
        raise ValueError(
            "lower_boundary must be a NumPy array."
        )

    if lower_boundary.shape != dem.shape:
        raise ValueError(
            "lower_boundary must have the same shape as dem."
        )

    if not np.isfinite(float(cell_size)) or cell_size <= 0:
        raise ValueError(
            "cell_size must be finite and greater than zero."
        )

    if not np.isfinite(float(vertical_accuracy)) or vertical_accuracy <= 0:
        raise ValueError(
            "vertical_accuracy must be finite and greater than zero."
        )


def _empty_audit() -> dict:
    """
    Return an audit record for a no-op flat resolution.

    Used when the flat mask is empty and no modification is required.

    :returns: Audit dictionary.
    """
    return {
        "method": "garbrecht_martz_flat_resolution",
        "flat_cells": 0,
        "higher_boundary_cells": 0,
        "lower_boundary_cells": 0,
        "step": 0.0,
        "max_gradient_away": 0.0,
        "max_gradient_toward": 0.0,
        "total_elevation_change": 0.0,
        "maximum_elevation_change": 0.0,
        "modified_cells": 0,
    }
