"""
Mayim Tools - Native Single-Cell De-Pitting
============================================

Implements the isolated single-cell de-pitting operation for Stage 5
Selective Flow Enforcement.

A single-cell pit is raised to the elevation of its lowest valid
neighbour. This is deliberately separate from breach and fill operations
because isolated pits do not require a full path-search or flood-fill
operation.

Methodology basis
-----------------
Lindsay, J. B., and Dhun, K. (2015). Modelling surface drainage
patterns in altered landscapes using LiDAR. International Journal of
Geographical Information Science, 29(3), 397-411.

IP status
---------
Original Mayim implementation based on the published methodological
description. No WhiteboxTools, RichDEM, TauDEM or other third-party
hydrological source code is used.

This function returns an audit record for every call. It never modifies
the input array in place and never lowers an elevation.
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


def depit_single_cell(
    dem: np.ndarray,
    pit: tuple[int, int],
    nodata: float,
) -> tuple[np.ndarray, dict]:
    """
    Raise an isolated pit to its lowest valid neighbour.

    The input array is copied before modification. The input is never
    changed in place.

    If the pit elevation is already equal to or greater than the lowest
    valid neighbour, no elevation change is made. This prevents the
    function from lowering terrain.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    pit:
        Tuple containing ``(row, column)`` of the candidate pit.
    nodata:
        NoData sentinel value.

    Returns
    -------
    tuple[np.ndarray, dict]
        The new DEM array and an audit record.

    Raises
    ------
    TypeError
        If the DEM is not a NumPy array or the coordinate values are not
        integers.
    ValueError
        If the DEM is not two-dimensional, the pit coordinate is
        invalid, the pit is NoData, or no valid neighbours exist.
    """
    _validate_dem(dem)
    row, col = _validate_pit_coordinate(dem, pit)

    if not np.isfinite(dem[row, col]) or dem[row, col] == nodata:
        raise ValueError("The pit cell is NoData or non-finite.")

    neighbours = []

    for row_offset, col_offset in _D8:
        neighbour_row = row + row_offset
        neighbour_col = col + col_offset

        if not (
            0 <= neighbour_row < dem.shape[0] and 0 <= neighbour_col < dem.shape[1]
        ):
            continue

        value = dem[neighbour_row, neighbour_col]

        if value == nodata or not np.isfinite(value):
            continue

        neighbours.append(float(value))

    if not neighbours:
        raise ValueError("The pit has no valid neighbours.")

    original_elevation = float(dem[row, col])
    lowest_neighbour = min(neighbours)
    new_elevation = max(original_elevation, lowest_neighbour)
    elevation_change = new_elevation - original_elevation
    modified = elevation_change > 0.0

    result = dem.astype(np.float64, copy=True)
    result[row, col] = new_elevation

    audit = {
        "row": row,
        "column": col,
        "original_elevation": original_elevation,
        "new_elevation": float(new_elevation),
        "elevation_change": float(elevation_change),
        "lowest_neighbour_elevation": float(lowest_neighbour),
        "modified": modified,
        "method": "single_cell_depitting",
    }

    return result, audit


def _validate_dem(dem: np.ndarray) -> None:
    """
    Validate the DEM array.

    Parameters
    ----------
    dem : np.ndarray
        DEM array.

    Raises
    ------
    TypeError
        If the DEM is not a NumPy array.
    ValueError
        If the DEM is not two-dimensional.
    """
    if not isinstance(dem, np.ndarray):
        raise TypeError("dem must be a NumPy array.")

    if dem.ndim != 2:
        raise ValueError("dem must be a two-dimensional array.")


def _validate_pit_coordinate(
    dem: np.ndarray,
    pit: tuple[int, int],
) -> tuple[int, int]:
    """
    Validate and normalise a pit coordinate.

    Parameters
    ----------
    dem : np.ndarray
        DEM array.
    pit : tuple[int, int]
        Row and column coordinate.

    Returns
    -------
    tuple[int, int]
        Integer row and column.

    Raises
    ------
    ValueError
        If the coordinate is invalid or outside the DEM.
    TypeError
        If the coordinate values are not integers.
    """
    if not isinstance(pit, tuple) or len(pit) != 2:
        raise ValueError("pit must be a (row, column) tuple.")

    row, col = pit

    if not isinstance(row, (int, np.integer)) or not isinstance(
        col,
        (int, np.integer),
    ):
        raise TypeError("Pit coordinates must be integers.")

    row = int(row)
    col = int(col)

    if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
        raise ValueError(f"Pit coordinate ({row}, {col}) is outside the DEM.")

    return row, col
