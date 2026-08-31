"""
Mayim Tools - Flat Surface Detection
=====================================

Identifies equal-elevation plateau cells and their higher and lower
boundaries for Stage 6 gradient resolution.

A flat candidate is a valid cell with at least one valid neighbour at
the same elevation. A candidate remains part of the flat even when it
also borders a lower or higher cell. This is necessary because the
boundary cells of a flat are precisely the cells through which the flat
receives or releases flow.

The boundary masks are therefore computed separately:

    higher_boundary:
        Flat cells adjacent to valid terrain higher than the flat.

    lower_boundary:
        Flat cells adjacent to valid terrain lower than the flat.

This module performs candidate-cell detection. Connected-component
grouping and per-flat gradient resolution are handled by later stages.

Methodology basis
-----------------
Garbrecht, J. and Martz, L. W. (1997).
The assignment of drainage direction over flat surfaces in raster
digital elevation models.
Journal of Hydrology, 193(1-4), 204-213.

IP status
---------
Original Mayim implementation using NumPy only.
No third-party hydrological package is used at runtime.
"""

from __future__ import annotations

import numpy as np

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


def detect_flats(
    dem: np.ndarray,
    nodata: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify flat candidate cells and their boundaries.

    A flat candidate is a valid cell with at least one valid
    eight-connected neighbour at the same elevation.

    A flat candidate may also have a higher or lower neighbour. Such
    neighbours define the inflow and outflow boundaries and must not
    exclude the cell from the flat candidate mask.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    nodata:
        NoData sentinel value.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Three Boolean arrays:

            flat_mask
                Equal-elevation flat candidate cells.

            higher_boundary
                Flat cells adjacent to higher valid terrain.

            lower_boundary
                Flat cells adjacent to lower valid terrain.

    Raises
    ------
    ValueError
        If dem is not a two-dimensional NumPy array.
    """
    if not isinstance(dem, np.ndarray) or dem.ndim != 2:
        raise ValueError(
            "dem must be a two-dimensional NumPy array."
        )

    rows, cols = dem.shape
    valid = np.isfinite(dem) & (dem != nodata)

    has_equal_neighbour = np.zeros(
        (rows, cols),
        dtype=bool,
    )

    higher_boundary = np.zeros(
        (rows, cols),
        dtype=bool,
    )

    lower_boundary = np.zeros(
        (rows, cols),
        dtype=bool,
    )

    for row_offset, col_offset in _D8:
        for row in range(rows):
            for col in range(cols):
                if not valid[row, col]:
                    continue

                neighbour_row = row + row_offset
                neighbour_col = col + col_offset

                if not (
                    0 <= neighbour_row < rows
                    and 0 <= neighbour_col < cols
                ):
                    continue

                if not valid[neighbour_row, neighbour_col]:
                    continue

                cell_elevation = float(dem[row, col])
                neighbour_elevation = float(
                    dem[neighbour_row, neighbour_col]
                )

                if neighbour_elevation == cell_elevation:
                    has_equal_neighbour[row, col] = True

    # A flat candidate is an equal-elevation plateau cell.
    flat_mask = valid & has_equal_neighbour

    # Identify the higher and lower boundaries separately. Boundary
    # cells remain in flat_mask; their neighbouring terrain determines
    # whether they are inflow or outflow boundary cells.
    for row_offset, col_offset in _D8:
        for row in range(rows):
            for col in range(cols):
                if not flat_mask[row, col]:
                    continue

                neighbour_row = row + row_offset
                neighbour_col = col + col_offset

                if not (
                    0 <= neighbour_row < rows
                    and 0 <= neighbour_col < cols
                ):
                    continue

                if not valid[neighbour_row, neighbour_col]:
                    continue

                cell_elevation = float(dem[row, col])
                neighbour_elevation = float(
                    dem[neighbour_row, neighbour_col]
                )

                if neighbour_elevation > cell_elevation:
                    higher_boundary[row, col] = True

                elif neighbour_elevation < cell_elevation:
                    lower_boundary[row, col] = True

    return flat_mask, higher_boundary, lower_boundary
