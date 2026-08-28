"""
Mayim Tools - Native Constrained Least-Cost Breaching
======================================================

Implements the Stage 5 constrained least-cost breach search.

The function searches from a depression pit to a valid outlet cell at
or below the depression spill elevation. Candidate paths are evaluated
using a deterministic least-cost search. The path cost is the total
amount of elevation lowering required to bring each path cell to the
spill elevation.

The search is bounded by:

    - Maximum path length in cells.
    - Maximum local excavation depth.
    - NoData barriers.
    - Raster bounds.
    - Selected neighbourhood connectivity.

This module returns a proposed path and audit information. It does not
modify a DEM. Applying the path to a DEM is deliberately separate so
that path selection can be reviewed before terrain modification.

Methodology basis
-----------------
The implementation follows the constrained least-cost breaching
principles described in:

    Lindsay, J. B. and Dhun, K. (2015). Modelling surface drainage
    patterns in altered landscapes using LiDAR. International Journal
    of Geographical Information Science, 29(3), 397-411.

    Lindsay, J. B. (2016). Efficient hybrid breaching-filling sink
    removal methods for flow path enforcement in digital elevation
    models. Hydrological Processes, 30(6), 846-857.

The implementation is also aligned with the Stage 5 specification in
the revised Mayim Tools DEM Hydrological Conditioning Research Paper.

IP status
---------
Original Mayim implementation based on the published methodological
description and the revised Mayim research paper.

No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
implementation is imported or called.

The implementation must remain based on the published methodology and
not on third-party source code.
"""

from __future__ import annotations

import heapq
from math import isfinite

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


def least_cost_breach(
    dem: np.ndarray,
    pit: tuple[int, int],
    spill_elevation: float,
    max_length: int,
    max_depth: float,
    nodata: float,
    connectivity: int = 8,
) -> tuple[list[tuple[int, int]] | None, dict]:
    """
    Find a constrained least-cost breach path.

    The path starts at ``pit`` and terminates at the first valid cell
    at or below ``spill_elevation`` other than the starting pit.

    For each candidate path cell, the required excavation depth is:

        max(0, cell_elevation - spill_elevation)

    A candidate cell is rejected if its required excavation depth exceeds
    ``max_depth``. The cumulative path cost is the sum of the required
    excavation depths along the path.

    The input DEM is never modified.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    pit:
        Candidate pit coordinate as ``(row, column)``.
    spill_elevation:
        Elevation at which the depression can drain.
    max_length:
        Maximum number of path steps from the pit.
    max_depth:
        Maximum excavation depth permitted at any path cell.
    nodata:
        NoData sentinel value.
    connectivity:
        Neighbourhood connectivity. Supported values are 4 and 8.

    Returns
    -------
    tuple[list[tuple[int, int]] | None, dict]
        Ordered path and audit record. The path is ``None`` if no
        admissible path is found.

    Notes
    -----
    The returned path is a proposed breach path. It is not an edited
    elevation profile. The Stage 5 enforcement layer must apply the
    path using an explicit breach-profile rule and record the resulting
    elevation changes separately.
    """
    _validate_inputs(
        dem=dem,
        pit=pit,
        spill_elevation=spill_elevation,
        max_length=max_length,
        max_depth=max_depth,
        connectivity=connectivity,
    )

    pit_row, pit_col = _validate_pit_coordinate(dem, pit)

    if (
        dem[pit_row, pit_col] == nodata
        or not np.isfinite(dem[pit_row, pit_col])
    ):
        raise ValueError("The pit cell is NoData or non-finite.")

    offsets = _D4 if connectivity == 4 else _D8

    # Each heap item is ordered by:
    #
    #   cumulative cost,
    #   maximum local excavation depth,
    #   path length,
    #   row,
    #   column
    #
    # The row and column tie-breakers make the result deterministic.
    queue: list[
        tuple[float, float, int, int, int]
    ] = [
        (0.0, 0.0, 0, pit_row, pit_col),
    ]

    best_cost: dict[tuple[int, int], tuple[float, float, int]] = {
        (pit_row, pit_col): (0.0, 0.0, 0),
    }

    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    nodata_cells_encountered = 0
    visited_cells = 0
    failure_reason = "maximum_constraints_exceeded"

    while queue:
        cost, maximum_depth, length, row, col = heapq.heappop(queue)
        current = (row, col)

        best = best_cost.get(current)

        if best is None:
            continue

        if (
            cost > best[0]
            or maximum_depth > best[1]
            or length > best[2]
        ):
            continue

        visited_cells += 1

        # The starting pit is not considered a successful outlet even
        # when its elevation is already below the spill elevation.
        if current != (pit_row, pit_col):
            if dem[row, col] <= spill_elevation:
                path = _reconstruct_path(
                    came_from=came_from,
                    endpoint=current,
                )

                audit = _success_audit(
                    path=path,
                    total_cost=cost,
                    maximum_excavation_depth=maximum_depth,
                    visited_cells=visited_cells,
                    nodata_cells_encountered=nodata_cells_encountered,
                    spill_elevation=spill_elevation,
                    max_length=max_length,
                    max_depth=max_depth,
                    connectivity=connectivity,
                )

                return path, audit

        if length >= max_length:
            continue

        for row_offset, col_offset in offsets:
            neighbour_row = row + row_offset
            neighbour_col = col + col_offset
            neighbour = (neighbour_row, neighbour_col)

            if not (
                0 <= neighbour_row < dem.shape[0]
                and 0 <= neighbour_col < dem.shape[1]
            ):
                continue

            neighbour_value = dem[neighbour_row, neighbour_col]

            if (
                neighbour_value == nodata
                or not np.isfinite(neighbour_value)
            ):
                nodata_cells_encountered += 1
                continue

            excavation_depth = max(
                0.0,
                float(neighbour_value) - float(spill_elevation),
            )

            if excavation_depth > max_depth:
                failure_reason = "maximum_constraints_exceeded"
                continue

            new_length = length + 1
            new_cost = cost + excavation_depth
            new_maximum_depth = max(
                maximum_depth,
                excavation_depth,
            )

            candidate_state = (
                new_cost,
                new_maximum_depth,
                new_length,
            )

            previous_state = best_cost.get(neighbour)

            if previous_state is not None:
                if not _is_better_state(
                    candidate=candidate_state,
                    previous=previous_state,
                ):
                    continue

            best_cost[neighbour] = candidate_state
            came_from[neighbour] = current

            heapq.heappush(
                queue,
                (
                    new_cost,
                    new_maximum_depth,
                    new_length,
                    neighbour_row,
                    neighbour_col,
                ),
            )

    audit = {
        "success": False,
        "method": "constrained_least_cost_breach",
        "failure_reason": failure_reason,
        "path": None,
        "path_length": 0,
        "total_cost": None,
        "maximum_excavation_depth": None,
        "visited_cells": visited_cells,
        "nodata_cells_encountered": nodata_cells_encountered,
        "spill_elevation": float(spill_elevation),
        "max_length": int(max_length),
        "max_depth": float(max_depth),
        "connectivity": int(connectivity),
    }

    return None, audit


def apply_breach_path(
    dem: np.ndarray,
    path: list[tuple[int, int]],
    spill_elevation: float,
    nodata: float,
) -> tuple[np.ndarray, dict]:
    """
    Apply a proposed breach path to a copied DEM.

    Each valid path cell is lowered, where necessary, to the spill
    elevation. Cells already below the spill elevation are unchanged.
    NoData cells are not modified.

    This function is separate from ``least_cost_breach()`` so that the
    selected path can be reviewed before terrain modification.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    path:
        Ordered breach path returned by ``least_cost_breach()``.
    spill_elevation:
        Target breach elevation.
    nodata:
        NoData sentinel value.

    Returns
    -------
    tuple[np.ndarray, dict]
        Modified DEM copy and audit record.

    Raises
    ------
    ValueError
        If the path or inputs are invalid.
    """
    if not isinstance(dem, np.ndarray) or dem.ndim != 2:
        raise ValueError(
            "dem must be a two-dimensional NumPy array."
        )

    if not isinstance(path, list) or not path:
        raise ValueError(
            "path must be a non-empty list of coordinates."
        )

    if not isfinite(float(spill_elevation)):
        raise ValueError(
            "spill_elevation must be finite."
        )

    result = dem.astype(np.float64, copy=True)
    modified_cells = []
    total_excavation = 0.0
    maximum_excavation = 0.0

    for coordinate in path:
        row, col = _validate_pit_coordinate(
            dem,
            coordinate,
        )

        original_value = float(dem[row, col])

        if (
            original_value == nodata
            or not np.isfinite(original_value)
        ):
            continue

        new_value = min(
            original_value,
            float(spill_elevation),
        )
        change = original_value - new_value

        if change <= 0.0:
            continue

        result[row, col] = new_value
        total_excavation += change
        maximum_excavation = max(
            maximum_excavation,
            change,
        )

        modified_cells.append(
            {
                "row": row,
                "column": col,
                "original_elevation": original_value,
                "new_elevation": float(new_value),
                "elevation_change": float(new_value - original_value),
                "excavation_depth": float(change),
            }
        )

    audit = {
        "success": True,
        "method": "apply_constrained_least_cost_breach",
        "path": [
            {
                "row": int(row),
                "column": int(col),
            }
            for row, col in path
        ],
        "path_length": max(0, len(path) - 1),
        "modified_cells": len(modified_cells),
        "total_excavation": float(total_excavation),
        "maximum_excavation": float(maximum_excavation),
        "modified_cell_records": modified_cells,
        "spill_elevation": float(spill_elevation),
    }

    return result, audit


def _validate_inputs(
    dem: np.ndarray,
    pit: tuple[int, int],
    spill_elevation: float,
    max_length: int,
    max_depth: float,
    connectivity: int,
) -> None:
    """
    Validate least-cost breach inputs.
    """
    if not isinstance(dem, np.ndarray):
        raise ValueError("dem must be a NumPy array.")

    if dem.ndim != 2:
        raise ValueError(
            "dem must be a two-dimensional array."
        )

    if not isinstance(pit, tuple) or len(pit) != 2:
        raise ValueError(
            "pit must be a (row, column) tuple."
        )

    if not isfinite(float(spill_elevation)):
        raise ValueError(
            "spill_elevation must be finite."
        )

    if not isinstance(max_length, (int, np.integer)):
        raise ValueError(
            "max_length must be an integer."
        )

    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than zero."
        )

    if not isfinite(float(max_depth)) or max_depth <= 0.0:
        raise ValueError(
            "max_depth must be greater than zero."
        )

    if connectivity not in (4, 8):
        raise ValueError(
            "connectivity must be either 4 or 8."
        )


def _validate_pit_coordinate(
    dem: np.ndarray,
    coordinate: tuple[int, int],
) -> tuple[int, int]:
    """
    Validate and normalise a raster coordinate.
    """
    if not isinstance(coordinate, tuple) or len(coordinate) != 2:
        raise ValueError(
            "Coordinate must be a (row, column) tuple."
        )

    row, col = coordinate

    if not isinstance(row, (int, np.integer)) or not isinstance(
        col,
        (int, np.integer),
    ):
        raise ValueError(
            "Coordinate values must be integers."
        )

    row = int(row)
    col = int(col)

    if not (
        0 <= row < dem.shape[0]
        and 0 <= col < dem.shape[1]
    ):
        raise ValueError(
            f"Coordinate ({row}, {col}) is outside the DEM."
        )

    return row, col


def _is_better_state(
    candidate: tuple[float, float, int],
    previous: tuple[float, float, int],
) -> bool:
    """
    Return whether a candidate search state is better.

    States are compared lexicographically by:

        1. total cost,
        2. maximum local excavation depth,
        3. path length.

    This produces deterministic behaviour for equal-cost paths.
    """
    return candidate < previous


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    endpoint: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Reconstruct an ordered path from the search origin to an endpoint.

    :param came_from: Predecessor mapping generated by the search.
    :param endpoint: Final path coordinate.
    :returns: Ordered list of ``(row, column)`` coordinates.
    """
    path = [endpoint]
    current = endpoint

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def _success_audit(
    path: list[tuple[int, int]],
    total_cost: float,
    maximum_excavation_depth: float,
    visited_cells: int,
    nodata_cells_encountered: int,
    spill_elevation: float,
    max_length: int,
    max_depth: float,
    connectivity: int,
) -> dict:
    """
    Build an audit record for a successful breach search.

    :param path: Ordered proposed breach path.
    :param total_cost: Cumulative excavation cost.
    :param maximum_excavation_depth: Largest excavation depth on path.
    :param visited_cells: Number of search states visited.
    :param nodata_cells_encountered: NoData neighbours encountered.
    :param spill_elevation: Target spill elevation.
    :param max_length: Maximum allowed path length.
    :param max_depth: Maximum allowed local excavation depth.
    :param connectivity: Search connectivity.
    :returns: Serialisable audit dictionary.
    """
    return {
        "success": True,
        "method": "constrained_least_cost_breach",
        "failure_reason": None,
        "path": [
            {
                "row": int(row),
                "column": int(col),
            }
            for row, col in path
        ],
        "path_length": max(0, len(path) - 1),
        "total_cost": float(total_cost),
        "maximum_excavation_depth": float(
            maximum_excavation_depth
        ),
        "visited_cells": int(visited_cells),
        "nodata_cells_encountered": int(
            nodata_cells_encountered
        ),
        "spill_elevation": float(spill_elevation),
        "max_length": int(max_length),
        "max_depth": float(max_depth),
        "connectivity": int(connectivity),
    }

