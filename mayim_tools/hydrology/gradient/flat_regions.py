"""
Mayim Tools - Connected Flat-Region Identification
===================================================

Groups flat candidate cells into deterministic connected regions for
Stage 6 gradient resolution.

This module does not modify elevations. It only identifies connected
flat regions and calculates basic region metadata.

IP status
---------
Original Mayim implementation using NumPy and Python standard-library
components only. No WhiteboxTools, RichDEM or TauDEM runtime code is used.

Methodology basis
-----------------
Garbrecht, J. and Martz, L. W. (1997).
The assignment of drainage direction over flat surfaces in raster
digital elevation models.
Journal of Hydrology, 193(1-4), 204-213.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_D4 = [
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
]

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


def label_flat_regions(
    flat_mask: np.ndarray,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict[int, dict]]:
    """
    Label connected flat regions.

    Region IDs are assigned deterministically in row-major order.
    Non-flat cells receive region ID zero.

    Parameters
    ----------
    flat_mask:
        Two-dimensional Boolean array identifying flat candidate cells.
    connectivity:
        Neighbourhood connectivity. Must be 4 or 8.

    Returns
    -------
    tuple[np.ndarray, dict[int, dict]]
        Region ID raster and region metadata.

        Region metadata contains:
            region_id
            cell_count
            row_min
            row_max
            col_min
            col_max

    Raises
    ------
    ValueError
        If the input mask or connectivity is invalid.
    """
    _validate_inputs(flat_mask, connectivity)

    rows, cols = flat_mask.shape
    region_ids = np.zeros(
        flat_mask.shape,
        dtype=np.int32,
    )
    regions: dict[int, dict] = {}

    offsets = _D4 if connectivity == 4 else _D8
    next_region_id = 1

    for row in range(rows):
        for col in range(cols):
            if not flat_mask[row, col]:
                continue

            if region_ids[row, col] != 0:
                continue

            queue = deque([(row, col)])
            region_ids[row, col] = next_region_id
            cells = []

            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))

                for row_offset, col_offset in offsets:
                    neighbour_row = current_row + row_offset
                    neighbour_col = current_col + col_offset

                    if not (
                        0 <= neighbour_row < rows
                        and 0 <= neighbour_col < cols
                    ):
                        continue

                    if not flat_mask[neighbour_row, neighbour_col]:
                        continue

                    if region_ids[neighbour_row, neighbour_col] != 0:
                        continue

                    region_ids[neighbour_row, neighbour_col] = (
                        next_region_id
                    )
                    queue.append((neighbour_row, neighbour_col))

            region_rows = [cell[0] for cell in cells]
            region_cols = [cell[1] for cell in cells]

            regions[next_region_id] = {
                "region_id": next_region_id,
                "cell_count": len(cells),
                "row_min": min(region_rows),
                "row_max": max(region_rows),
                "col_min": min(region_cols),
                "col_max": max(region_cols),
            }

            next_region_id += 1

    return region_ids, regions


def _validate_inputs(
    flat_mask: np.ndarray,
    connectivity: int,
) -> None:
    """
    Validate flat-region inputs.
    """
    if not isinstance(flat_mask, np.ndarray):
        raise ValueError(
            "flat_mask must be a NumPy array."
        )

    if flat_mask.ndim != 2:
        raise ValueError(
            "flat_mask must be a two-dimensional array."
        )

    if flat_mask.dtype != np.bool_:
        raise ValueError(
            "flat_mask must have Boolean dtype."
        )

    if connectivity not in (4, 8):
        raise ValueError(
            "connectivity must be either 4 or 8."
        )
