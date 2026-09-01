"""
Mayim Tools - Region-Aware Gradient Resolution
===============================================

Implements the native Stage 6 flat-area gradient-resolution component.

The implementation applies a region-aware, two-distance synthetic
gradient to connected flat regions:

    - A distance from higher-elevation boundary cells.
    - A distance from lower-elevation outlet cells.

The two distances are combined into a small correction. The correction
magnitude is limited by the supplied vertical accuracy and cell size.

This module does not use WhiteboxTools, RichDEM, TauDEM or NetworkX.
It does not modify cells outside the supplied flat regions and does not
modify the input DEM in place.

Methodology basis
-----------------
Garbrecht, J. and Martz, L. W. (1997). The assignment of drainage
direction over flat surfaces in raster digital elevation models.
Journal of Hydrology, 193(1-4), 204-213.

IP status
---------
Original Mayim implementation based on the published methodology.
No third-party hydrological source code is used at runtime.

Important qualification
-----------------------
The present implementation is a native Mayim Stage 6 component.
Further validation is required for complex flats, multiple outlets,
floating-point tolerance, rectangular pixels and very large rasters.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_CARDINAL = [
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
]

def resolve_flats(
    dem: np.ndarray,
    flat_mask: np.ndarray,
    higher_boundary: np.ndarray,
    lower_boundary: np.ndarray,
    cell_size: float,
    vertical_accuracy: float,
    nodata: float,
    region_ids: np.ndarray | None = None,
    allow_unresolved: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Resolve connected flat regions independently.

    When ``region_ids`` is supplied, each positive region ID is processed
    independently. This prevents BFS distances and audit statistics from
    being mixed between disconnected flat regions.

    When ``region_ids`` is None, the complete flat mask is treated as one
    backward-compatible region.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    flat_mask:
        Boolean mask identifying candidate flat cells.
    higher_boundary:
        Boolean mask identifying flat cells adjacent to higher terrain.
    lower_boundary:
        Boolean mask identifying flat cells adjacent to lower terrain.
    cell_size:
        Cell size in map units.
    vertical_accuracy:
        DEM vertical accuracy or supported vertical-resolution estimate.
    nodata:
        NoData sentinel value.
    region_ids:
        Optional integer raster identifying connected flat regions.
        Zero identifies non-flat cells. Positive integers identify
        individual flat regions.
    allow_unresolved:
        If False, a flat region without a valid lower boundary raises
        ValueError. If True, the region is preserved, recorded as
        unresolved and processing continues for other regions.

    Returns
    -------
    tuple[np.ndarray, dict]
        Resolved DEM and audit dictionary.

    Raises
    ------
    ValueError
        If any input is invalid or a non-empty flat region has no valid
        lower boundary.
    """
    _validate_inputs(
        dem=dem,
        flat_mask=flat_mask,
        higher_boundary=higher_boundary,
        lower_boundary=lower_boundary,
        cell_size=cell_size,
        vertical_accuracy=vertical_accuracy,
        region_ids=region_ids,
    )

    result = dem.astype(np.float64, copy=True)

    valid_mask = (
        np.isfinite(dem)
        & (dem != nodata)
    )

    active_flat_mask = flat_mask & valid_mask

    if not np.any(active_flat_mask):
        return result, _empty_audit()

    if region_ids is None:
        working_region_ids = np.zeros(
            dem.shape,
            dtype=np.int32,
        )
        working_region_ids[active_flat_mask] = 1
    else:
        working_region_ids = region_ids.copy()
        working_region_ids[~active_flat_mask] = 0

    region_values = sorted(
        int(region_id)
        for region_id in np.unique(working_region_ids)
        if region_id > 0
    )

    if not region_values:
        return result, _empty_audit()

    region_audits = []
    unresolved_regions = []

    for region_id in region_values:
        region_mask = (
            active_flat_mask
            & (working_region_ids == region_id)
        )

        if not np.any(region_mask):
            continue

        region_higher_boundary = (
            higher_boundary
            & region_mask
        )

        region_lower_boundary = (
            lower_boundary
            & region_mask
        )

        if not np.any(region_lower_boundary):
            unresolved_record = {
                "region_id": region_id,
                "flat_cells": int(np.sum(region_mask)),
                "higher_boundary_cells": int(
                    np.sum(region_higher_boundary)
                ),
                "lower_boundary_cells": 0,
                "status": "unresolved",
                "reason": "no valid lower boundary",
                "modified_cells": 0,
            }

            if not allow_unresolved:
                raise ValueError(
                    f"Flat region {region_id} has no valid lower boundary."
                )

            unresolved_regions.append(unresolved_record)
            continue

        distance_away = _bfs_distance(
            flat_mask=region_mask,
            seeds=region_higher_boundary,
        )

        distance_toward = _bfs_distance(
            flat_mask=region_mask,
            seeds=region_lower_boundary,
        )

        max_distance_away = float(
            np.max(distance_away[region_mask])
        )

        max_distance_toward = float(
            np.max(distance_toward[region_mask])
        )

        max_distance = max(
            max_distance_away,
            max_distance_toward,
            1.0,
        )

        # Keep the synthetic signal strictly bounded by the smaller
        # of the supplied vertical accuracy and cell size.
        step = min(
            float(vertical_accuracy),
            float(cell_size),
        ) / (2.0 * max_distance)

        correction = (
            2.0 * distance_toward
            + distance_away
        ) * step

        original_region_values = dem[region_mask].astype(
            np.float64,
            copy=True,
        )

        result[region_mask] = (
            original_region_values
            + correction[region_mask]
        )

        region_changes = (
            result[region_mask]
            - original_region_values
        )

        region_modified = np.abs(region_changes) > 0.0

        region_audits.append(
            {
                "region_id": region_id,
                "flat_cells": int(np.sum(region_mask)),
                "higher_boundary_cells": int(
                    np.sum(region_higher_boundary)
                ),
                "lower_boundary_cells": int(
                    np.sum(region_lower_boundary)
                ),
                "step": float(step),
                "max_gradient_away": max_distance_away,
                "max_gradient_toward": max_distance_toward,
                "total_elevation_change": float(
                    np.sum(region_changes)
                ),
                "maximum_elevation_change": float(
                    np.max(region_changes)
                ) if region_changes.size else 0.0,
                "minimum_elevation_change": float(
                    np.min(region_changes)
                ) if region_changes.size else 0.0,
                "modified_cells": int(
                    np.sum(region_modified)
                ),
            }
        )

    all_changes = result[active_flat_mask] - dem[active_flat_mask]

    audit = {
        "method": "garbrecht_martz_flat_resolution",
        "region_count": len(region_audits),
        "resolved_region_count": len(region_audits),
        "unresolved_region_count": len(unresolved_regions),
        "regions": region_audits,
        "unresolved_regions": unresolved_regions,
        "allow_unresolved": allow_unresolved,
        "flat_cells": int(np.sum(active_flat_mask)),
        "higher_boundary_cells": int(
            np.sum(higher_boundary & active_flat_mask)
        ),
        "lower_boundary_cells": int(
            np.sum(lower_boundary & active_flat_mask)
        ),
        "step": float(
            max(
                region["step"]
                for region in region_audits
            )
        ) if region_audits else 0.0,
        "max_gradient_away": float(
            max(
                region["max_gradient_away"]
                for region in region_audits
            )
        ) if region_audits else 0.0,
        "max_gradient_toward": float(
            max(
                region["max_gradient_toward"]
                for region in region_audits
            )
        ) if region_audits else 0.0,
        "total_elevation_change": float(
            np.sum(all_changes)
        ) if all_changes.size else 0.0,
        "maximum_elevation_change": float(
            np.max(all_changes)
        ) if all_changes.size else 0.0,
        "minimum_elevation_change": float(
            np.min(all_changes)
        ) if all_changes.size else 0.0,
        "modified_cells": int(
            np.sum(np.abs(all_changes) > 0.0)
        ) if all_changes.size else 0,
    }

    return result, audit


def _bfs_distance(
    flat_mask: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """
    Calculate cardinal BFS distance from seed cells.

    Distances are calculated only inside ``flat_mask``. Seed cells have
    distance 1. Unreachable cells remain zero.

    Parameters
    ----------
    flat_mask:
        Boolean mask of one connected flat region.
    seeds:
        Boolean mask of seed cells inside the flat region.

    Returns
    -------
    np.ndarray
        Integer distance array.
    """
    rows, cols = flat_mask.shape

    distances = np.zeros(
        flat_mask.shape,
        dtype=np.int32,
    )

    visited = np.zeros(
        flat_mask.shape,
        dtype=bool,
    )

    queue = deque()

    seed_positions = np.argwhere(
        seeds & flat_mask
    )

    for position in seed_positions:
        row = int(position[0])
        col = int(position[1])

        if visited[row, col]:
            continue

        visited[row, col] = True
        distances[row, col] = 1
        queue.append((row, col))

    while queue:
        row, col = queue.popleft()

        for row_offset, col_offset in _CARDINAL:
            neighbour_row = row + row_offset
            neighbour_col = col + col_offset

            if not (
                0 <= neighbour_row < rows
                and 0 <= neighbour_col < cols
            ):
                continue

            if not flat_mask[neighbour_row, neighbour_col]:
                continue

            if visited[neighbour_row, neighbour_col]:
                continue

            visited[neighbour_row, neighbour_col] = True
            distances[neighbour_row, neighbour_col] = (
                distances[row, col] + 1
            )
            queue.append(
                (neighbour_row, neighbour_col)
            )

    return distances


def _validate_inputs(
    dem: np.ndarray,
    flat_mask: np.ndarray,
    higher_boundary: np.ndarray,
    lower_boundary: np.ndarray,
    cell_size: float,
    vertical_accuracy: float,
    region_ids: np.ndarray | None,
) -> None:
    """
    Validate gradient-resolution inputs.
    """
    if not isinstance(dem, np.ndarray):
        raise ValueError("dem must be a NumPy array.")

    if dem.ndim != 2:
        raise ValueError(
            "dem must be a two-dimensional array."
        )

    _validate_boolean_mask(
        flat_mask,
        dem.shape,
        "flat_mask",
    )

    _validate_boolean_mask(
        higher_boundary,
        dem.shape,
        "higher_boundary",
    )

    _validate_boolean_mask(
        lower_boundary,
        dem.shape,
        "lower_boundary",
    )

    if not np.isfinite(float(cell_size)) or cell_size <= 0.0:
        raise ValueError(
            "cell_size must be finite and greater than zero."
        )

    if (
        not np.isfinite(float(vertical_accuracy))
        or vertical_accuracy <= 0.0
    ):
        raise ValueError(
            "vertical_accuracy must be finite and greater than zero."
        )

    if region_ids is not None:
        if not isinstance(region_ids, np.ndarray):
            raise ValueError(
                "region_ids must be a NumPy array or None."
            )

        if region_ids.ndim != 2:
            raise ValueError(
                "region_ids must be a two-dimensional array."
            )

        if region_ids.shape != dem.shape:
            raise ValueError(
                "region_ids must have the same shape as dem."
            )

        if not np.issubdtype(
            region_ids.dtype,
            np.integer,
        ):
            raise ValueError(
                "region_ids must have an integer dtype."
            )


def _validate_boolean_mask(
    mask: np.ndarray,
    expected_shape: tuple[int, int],
    name: str,
) -> None:
    """
    Validate a Boolean mask.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(
            f"{name} must be a NumPy array."
        )

    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional array."
        )

    if mask.shape != expected_shape:
        raise ValueError(
            f"{name} must have the same shape as dem."
        )

    if mask.dtype != np.bool_:
        raise ValueError(
            f"{name} must have Boolean dtype."
        )


def _empty_audit() -> dict:
    """
    Return an audit record for a no-op operation.
    """
    return {
        "method": "garbrecht_martz_flat_resolution",
        "region_count": 0,
        "regions": [],
        "flat_cells": 0,
        "higher_boundary_cells": 0,
        "lower_boundary_cells": 0,
        "step": 0.0,
        "max_gradient_away": 0.0,
        "max_gradient_toward": 0.0,
        "total_elevation_change": 0.0,
        "maximum_elevation_change": 0.0,
        "minimum_elevation_change": 0.0,
        "modified_cells": 0,
    }
