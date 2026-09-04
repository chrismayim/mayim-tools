"""
Mayim Tools - Hydrography and DEM Divergence Analysis
======================================================

Stage 7C native divergence analysis for optional hydrography enforcement.

This module compares:

    - A Boolean mask representing mapped hydrography.
    - A Boolean mask representing DEM-derived flow evidence.

A configurable positional tolerance allows small raster-grid differences
to be classified as tolerated rather than material divergence.

This module does not:

    - Modify DEM elevations.
    - Burn hydrography into a DEM.
    - Reproject vector data.
    - Rasterise vector data.
    - Calculate burn depth.
    - Resolve topology conflicts.

Those operations belong to later Stage 7 components.

IP status
---------
Original Mayim implementation using NumPy and Python standard-library
components only.

No WhiteboxTools, RichDEM, TauDEM or other hydrological implementation
is imported or called.

The divergence result is evidence for later enforcement. It is not, by
itself, permission to modify terrain.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_NEIGHBOURS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def analyse_hydrography_divergence(
    hydrography_mask: np.ndarray,
    dem_flow_mask: np.ndarray,
    positional_tolerance_cells: int,
    nodata_mask: np.ndarray | None = None,
) -> dict:
    """
    Analyse positional divergence between mapped hydrography and
    DEM-derived flow evidence.

    A mapped hydrography cell is considered spatially supported when a
    DEM-flow cell occurs within the specified tolerance. A DEM-flow cell
    is similarly considered supported when mapped hydrography occurs
    within the specified tolerance.

    Exact overlap is reported separately from tolerance-based support.

    Parameters
    ----------
    hydrography_mask:
        Two-dimensional Boolean array. True identifies mapped
        hydrography cells.

    dem_flow_mask:
        Two-dimensional Boolean array. True identifies DEM-derived flow
        evidence cells.

    positional_tolerance_cells:
        Non-negative integer tolerance in raster cells. Zero requires
        exact overlap.

    nodata_mask:
        Optional two-dimensional Boolean array. True identifies cells
        that cannot be assessed.

    Returns
    -------
    dict
        Dictionary containing output masks, review records and
        statistics.

    Raises
    ------
    ValueError
        If an input mask, tolerance or NoData mask is invalid.
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

    hydrography_cells = hydrography_mask & assessable

    dem_flow_cells = dem_flow_mask & assessable

    tolerance = int(positional_tolerance_cells)

    dem_neighbourhood = _dilate(
        dem_flow_cells,
        radius=tolerance,
    )

    hydrography_neighbourhood = _dilate(
        hydrography_cells,
        radius=tolerance,
    )

    exact_alignment = hydrography_cells & dem_flow_cells

    hydrography_supported = hydrography_cells & dem_neighbourhood

    dem_flow_supported = dem_flow_cells & hydrography_neighbourhood

    tolerated_hydrography = hydrography_supported & ~exact_alignment

    tolerated_dem_flow = dem_flow_supported & ~exact_alignment

    tolerated = tolerated_hydrography | tolerated_dem_flow

    hydrography_only = hydrography_cells & ~dem_neighbourhood

    dem_only = dem_flow_cells & ~hydrography_neighbourhood

    material_divergence = hydrography_only | dem_only

    aligned_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )
    aligned_output[exact_alignment] = 1

    tolerated_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )
    tolerated_output[tolerated] = 1

    divergence_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )
    divergence_output[material_divergence] = 1

    conflict_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )

    # Material divergence is not automatically enforcement permission.
    # It is recorded as conflict/review evidence until later topology,
    # confidence and positional-accuracy checks are completed.
    conflict_output[material_divergence] = 1

    outside_extent_output = np.zeros(
        hydrography_mask.shape,
        dtype=np.uint8,
    )

    for output in (
        aligned_output,
        tolerated_output,
        divergence_output,
        conflict_output,
        outside_extent_output,
    ):
        output[~assessable] = 255

    review_records = _build_review_records(
        hydrography_only=hydrography_only,
        dem_only=dem_only,
        assessable=assessable,
    )

    statistics = {
        "rows": int(hydrography_mask.shape[0]),
        "columns": int(hydrography_mask.shape[1]),
        "assessable_cells": int(np.sum(assessable)),
        "nodata_cells": int(np.sum(~assessable)),
        "hydrography_cells": int(np.sum(hydrography_cells)),
        "dem_flow_cells": int(np.sum(dem_flow_cells)),
        "aligned_cells": int(np.sum(exact_alignment)),
        "tolerated_cells": int(np.sum(tolerated)),
        "hydrography_only_cells": int(np.sum(hydrography_only)),
        "dem_only_cells": int(np.sum(dem_only)),
        "material_divergence_cells": int(np.sum(material_divergence)),
        "conflict_cells": int(np.sum(conflict_output == 1)),
        "outside_extent_cells": int(np.sum(outside_extent_output == 1)),
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
    Dilate a Boolean mask using an eight-connected cell neighbourhood.

    The radius is measured in raster-cell steps using Chebyshev distance.
    A radius of zero returns a copy of the original mask.

    Parameters
    ----------
    mask:
        Two-dimensional Boolean array.

    radius:
        Non-negative dilation radius in cells.

    Returns
    -------
    np.ndarray
        Dilated Boolean array.
    """
    if radius == 0:
        return mask.copy()

    rows, cols = mask.shape
    result = mask.copy()

    distance = np.full(
        mask.shape,
        -1,
        dtype=np.int32,
    )

    queue = deque()

    for row, col in np.argwhere(mask):
        row = int(row)
        col = int(col)

        distance[row, col] = 0
        queue.append((row, col))

    while queue:
        row, col = queue.popleft()
        current_distance = int(distance[row, col])

        if current_distance >= radius:
            continue

        for row_offset, col_offset in _NEIGHBOURS_8:
            neighbour_row = row + row_offset
            neighbour_col = col + col_offset

            if not (0 <= neighbour_row < rows and 0 <= neighbour_col < cols):
                continue

            if distance[neighbour_row, neighbour_col] != -1:
                continue

            distance[neighbour_row, neighbour_col] = current_distance + 1
            result[neighbour_row, neighbour_col] = True
            queue.append((neighbour_row, neighbour_col))

    return result


def _build_review_records(
    hydrography_only: np.ndarray,
    dem_only: np.ndarray,
    assessable: np.ndarray,
) -> list[dict]:
    """
    Build deterministic per-cell review records.

    Each materially divergent cell is recorded individually in this
    first implementation. Later versions may aggregate neighbouring
    cells into conflict regions.

    Parameters
    ----------
    hydrography_only:
        Cells containing mapped hydrography without nearby DEM-flow
        evidence.

    dem_only:
        Cells containing DEM-flow evidence without nearby mapped
        hydrography.

    assessable:
        Cells eligible for assessment.

    Returns
    -------
    list[dict]
        Deterministically ordered review records.
    """
    records = []

    material = hydrography_only | dem_only

    rows, cols = np.where(material & assessable)

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
            raise ValueError(f"{name} must be a NumPy array.")

        if mask.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")

        if mask.dtype != np.bool_:
            raise ValueError(f"{name} must have Boolean dtype.")

    if hydrography_mask.shape != dem_flow_mask.shape:
        raise ValueError(
            "hydrography_mask and dem_flow_mask must have " "the same shape."
        )

    if not isinstance(
        positional_tolerance_cells,
        (int, np.integer),
    ):
        raise ValueError("positional_tolerance_cells must be an integer.")

    if positional_tolerance_cells < 0:
        raise ValueError("positional_tolerance_cells must be non-negative.")

    if nodata_mask is not None:
        if not isinstance(nodata_mask, np.ndarray):
            raise ValueError("nodata_mask must be a NumPy array or None.")

        if nodata_mask.ndim != 2:
            raise ValueError("nodata_mask must be two-dimensional.")

        if nodata_mask.shape != hydrography_mask.shape:
            raise ValueError(
                "nodata_mask must have the same shape as " "the input masks."
            )

        if nodata_mask.dtype != np.bool_:
            raise ValueError("nodata_mask must have Boolean dtype.")
