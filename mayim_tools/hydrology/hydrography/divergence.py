"""
Mayim Tools - Hydrography and DEM Divergence Analysis
======================================================

Stage 7C native divergence analysis for optional hydrography enforcement.

This module compares:

    - A Boolean mask representing mapped hydrography.
    - A Boolean mask representing DEM-derived flow evidence.

The comparison allows a configurable positional tolerance in raster cells.
Small spatial differences are classified as tolerated rather than material
divergence.

This module does not:

    - Modify DEM elevations.
    - Burn hydrography.
    - Reproject vector data.
    - Rasterise vector data.
    - Resolve topology conflicts.
    - Calculate burn depth.

Those operations belong to later Stage 7 components.

IP status
---------
Original Mayim implementation using NumPy only.

No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
implementation is used.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def analyse_hydrography_divergence(
    hydrography_mask: np.ndarray,
    dem_flow_mask: np.ndarray,
    positional_tolerance_cells: int,
    nodata_mask: np.ndarray | None = None,
) -> dict:
    """
    Analyse positional divergence between hydrography and DEM flow evidence.

    A hydrography cell is considered aligned or tolerated when it lies
    within the configured cell tolerance of a DEM-flow cell. The same
    symmetric rule is applied to DEM-flow cells relative to hydrography.

    Material divergence consists of valid hydrography or DEM-flow cells
    that have no corresponding cell within the positional tolerance.

    Parameters
    ----------
    hydrography_mask:
        Two-dimensional Boolean raster mask. True indicates mapped
        hydrography.

    dem_flow_mask:
        Two-dimensional Boolean raster mask. True indicates DEM-derived
        flow evidence.

    positional_tolerance_cells:
        Non-negative integer tolerance in raster cells. Zero requires
        exact cell agreement.

    nodata_mask:
        Optional Boolean mask identifying cells that cannot be assessed.
        NoData cells are assigned 255 in the output masks.

    Returns
    -------
    dict
        Dictionary containing output masks, review records and
        statistics.

    Raises
    ------
    ValueError
        If masks, tolerance or NoData mask are invalid.
    """
    _validate_inputs(
        hydrography_mask=hydrography_mask,
        dem_flow_mask=dem_flow_mask,
        positional_tolerance_cells=positional_tolerance_cells,
        nodata_mask=nodata_mask,
    )

    assessable = np.ones(
        hydrography_mask.shape,
        dtype=bool,
    )

    if nodata_mask is not None:
        assessable &= ~nodata_mask

    tolerance = int(positional_tolerance_cells)

    hydrography_assessable = (
        hydrography_mask & assessable
    )
    dem_flow_assessable = (
        dem_flow_mask & assessable
    )

    hydrography_near_dem = _dilate(
        dem_flow_assessable,
        tolerance,
    )
    dem_near_hydrography = _dilate(
        hydrography_assessable,
        tolerance,
    )

    hydrography_only = (
        hydrography_assessable
        & ~hydrography_near_dem
    )

    dem_only = (
        dem_flow_assessable
        & ~dem_near_hydrography
    )

    material_divergence = hydrography_only | dem_only

    exact_alignment = (
        hydrography_assessable
        & dem_flow_assessable
    )

    tolerated_hydrography = (
        hydrography_assessable
        & hydrography_near_dem
        & ~dem_flow_assessable
    )

    tolerated_dem_flow = (
        dem_flow_assessable
        & dem_near_hydrography
        & ~hydrography_assessable
    )

    tolerated = tolerated_hydrography | tolerated_dem_flow

    aligned = exact_alignment | tolerated_hydrography

    # A same-grid Boolean comparison cannot independently determine
    # whether a disagreement is caused by topology, positional error,
    # incomplete hydrography or an incorrect DEM. Such cases are
    # retained as review records rather than automatically enforced.
    conflict = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )

    conflict[
        material_divergence
        & assessable
    ] = 1

    divergence_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )
    divergence_output[
        material_divergence
        & assessable
    ] = 1

    aligned_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )
    aligned_output[
        aligned
        & assessable
    ] = 1

    tolerated_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )
    tolerated_output[
        tolerated
        & assessable
    ] = 1

    conflict_output = conflict

    outside_extent_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )

    for output in (
        divergence_output,
        aligned_output,
        tolerated_output,
        conflict_output,
        outside_extent_output,
    ):
        output[~assessable] = 255

    review_records = _build_review_records(
        material_divergence=material_divergence,
        hydrography_only=hydrography_only,
        dem_only=dem_only,
        assessable=assessable,
    )

    statistics = {
        "assessable_cells": int(np.sum(assessable)),
        "nodata_cells": int(np.sum(~assessable)),
        "hydrography_cells": int(
            np.sum(hydrography_assessable)
        ),
        "dem_flow_cells": int(
            np.sum(dem_flow_assessable)
        ),
        "aligned_cells": int(
            np.sum(exact_alignment)
        ),
        "tolerated_cells": int(
            np.sum(tolerated)
        ),
        "hydrography_only_cells": int(
            np.sum(hydrography_only)
        ),
        "dem_only_cells": int(
            np.sum(dem_only)
        ),
        "material_divergence_cells": int(
            np.sum(material_divergence)
        ),
        "conflict_cells": int(
            np.sum(conflict_output == 1)
        ),
        "outside_extent_cells": int(
            np.sum(outside_extent_output == 1)
        ),
        "positional_tolerance_cells": tolerance,
    }

    return {
        "divergence_mask": divergence_output,
        "aligned_mask": aligned_output,
        "tolerated_mask": tolerated_output,
        "conflict_mask": conflict_output,
        "outside_extent_mask": outside_extent_output,
        "review_records": review_records,
        "statistics": statistics,
    }


def _dilate(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    Dilate a Boolean mask using a square cell-distance neighbourhood.

    The implementation uses breadth-first traversal and therefore does
    not require SciPy. A radius of zero returns a copy of the original
    mask.

    Parameters
    ----------
    mask:
        Two-dimensional Boolean mask.
    radius:
        Maximum cell distance for dilation.

    Returns
    -------
    np.ndarray
        Dilated Boolean mask.
    """
    if radius == 0:
        return mask.copy()

    rows, cols = mask.shape
    result = mask.copy()
    distances = np.full(
        mask.shape,
        -1,
        dtype=np.int32,
    )
    queue = deque()

    for row, col in np.argwhere(mask):
        row = int(row)
        col = int(col)
        distances[row, col] = 0
        queue.append((row, col))

    neighbours = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    while queue:
        row, col = queue.popleft()
        distance = int(distances[row, col])

        if distance >= radius:
            continue

        for row_offset, col_offset in neighbours:
            neighbour_row = row + row_offset
            neighbour_col = col + col_offset

            if not (
                0 <= neighbour_row < rows
                and 0 <= neighbour_col < cols
            ):
                continue

            if distances[neighbour_row, neighbour_col] != -1:
                continue

            distances[neighbour_row, neighbour_col] = distance + 1
            result[neighbour_row, neighbour_col] = True
            queue.append(
                (neighbour_row, neighbour_col)
            )

    return result


def _build_review_records(
    material_divergence: np.ndarray,
    hydrography_only: np.ndarray,
    dem_only: np.ndarray,
    assessable: np.ndarray,
) -> list[dict]:
    """
    Build deterministic review records for divergent cells.

    Contiguous-region aggregation is intentionally deferred to a later
    refinement. For now, each materially divergent cell is recorded so
    that no conflict is hidden.

    Parameters
    ----------
    material_divergence:
        Boolean material-divergence mask.
    hydrography_only:
        Boolean mapped-hydrography-only mask.
    dem_only:
        Boolean DEM-flow-only mask.
    assessable:
        Boolean assessable-cell mask.

    Returns
    -------
    list[dict]
        One deterministic record per divergent cell.
    """
    records = []

    rows, cols = np.where(
        material_divergence & assessable
    )

    for row, col in zip(rows.tolist(), cols.tolist()):
        if hydrography_only[row, col]:
            divergence_type = "hydrography_only"
        elif dem_only[row, col]:
            divergence_type = "dem_flow_only"
        else:
            divergence_type = "unknown"

        records.append(
            {
                "row": int(row),
                "column": int(col),
                "type": divergence_type,
                "requires_review": True,
            }
        )

    return records


def _validate_inputs(
    hydrography_mask: np.ndarray,
    dem_flow_mask: np.ndarray,
    positional_tolerance_cells: int,
    nodata_mask: np.ndarray | None,
) -> None:
    """
    Validate divergence-analysis inputs.
    """
    for mask, name in (
        (hydrography_mask, "hydrography_mask"),
        (dem_flow_mask, "dem_flow_mask"),
    ):
        if not isinstance(mask, np.ndarray):
            raise ValueError(
                f"{name} must be a NumPy array."
            )

        if mask.ndim != 2:
            raise ValueError(
                f"{name} must be two-dimensional."
            )

        if mask.dtype != np.bool_:
            raise ValueError(
                f"{name} must have Boolean dtype."
            )

    if hydrography_mask.shape != dem_flow_mask.shape:
        raise ValueError(
            "hydrography_mask and dem_flow_mask must have "
            "the same shape."
        )

    if not isinstance(
        positional_tolerance_cells,
        (int, np.integer),
    ):
        raise ValueError(
            "positional_tolerance_cells must be an integer."
        )

    if positional_tolerance_cells < 0:
        raise ValueError(
            "positional_tolerance_cells must be non-negative."
        )

    if nodata_mask is not None:
        if not isinstance(nodata_mask, np.ndarray):
            raise ValueError(
                "nodata_mask must be a NumPy array or None."
            )

        if nodata_mask.shape != hydrography_mask.shape:
            raise ValueError(
                "nodata_mask must have the same shape as the "
                "input masks."
            )

        if nodata_mask.dtype != np.bool_:
            raise ValueError(
                "nodata_mask must have Boolean dtype."
            )
