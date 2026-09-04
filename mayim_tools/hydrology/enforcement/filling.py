"""
Mayim Tools - Confined Priority-Flood Filling
==============================================

Implements the confined filling fallback for Stage 5 Selective Flow
Enforcement.

This function fills only the cells contained in a supplied depression
mask. Cells outside the mask are never modified.

Methodology basis
-----------------
The implementation follows the Priority-Flood principle described in:

    Barnes, R., Lehman, C., and Mulla, D. (2014).
    Priority-flood: An optimal depression-filling and watershed-labeling
    algorithm for digital elevation models.
    Computers and Geosciences, 62, 117-127.

The confined Stage 5 use case raises cells within a classified artifact
depression to its explicitly supplied spill elevation after breaching
has failed or has been rejected by the enforcement constraints.

IP status
---------
Original Mayim implementation using Python and NumPy only.

No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
implementation is imported or called.

The implementation must remain based on the published methodology and
the revised Mayim research paper, not third-party source code.

Important
---------
This function is a fallback operation. It must only be called for a
depression that has already been classified for enforcement. It must
not be used to fill REAL_FEATURE or REVIEW_REQUIRED depressions.
"""

from __future__ import annotations

import numpy as np


def confined_priority_flood_fill(
    dem: np.ndarray,
    depression_mask: np.ndarray,
    spill_elevation: float,
    nodata: float,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict]:
    """
    Fill a supplied depression footprint to its spill elevation.

    Only valid cells inside ``depression_mask`` are eligible for
    modification. Cells outside the mask are copied unchanged.

    The input DEM is never modified in place.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    depression_mask:
        Boolean array with the same shape as ``dem``. True cells are
        part of the depression footprint to be filled.
    spill_elevation:
        Elevation to which eligible depression cells are raised.
    nodata:
        NoData sentinel value.
    connectivity:
        Connectivity convention recorded in the audit output.
        Supported values are 4 and 8. Connectivity does not alter the
        result of this confined elevation operation, but it is recorded
        for reproducibility and consistency with other enforcement
        components.

    Returns
    -------
    tuple[np.ndarray, dict]
        A copied filled DEM and an audit dictionary.

    Raises
    ------
    ValueError
        If the DEM or mask is invalid, the mask is empty, the spill
        elevation is not finite, or connectivity is unsupported.
    """
    _validate_inputs(
        dem=dem,
        depression_mask=depression_mask,
        spill_elevation=spill_elevation,
        connectivity=connectivity,
    )

    result = dem.astype(np.float64, copy=True)

    valid_mask = np.isfinite(dem) & (dem != nodata)

    eligible_mask = depression_mask & valid_mask

    if not np.any(eligible_mask):
        raise ValueError("The depression mask contains no valid cells.")

    original_values = dem[eligible_mask].astype(np.float64)

    # Filling may raise cells to the spill elevation, but never lowers
    # terrain and never changes cells above the spill elevation.
    filled_values = np.maximum(
        original_values,
        float(spill_elevation),
    )

    changes = filled_values - original_values
    modified = changes > 0.0

    result[eligible_mask] = filled_values

    modified_rows, modified_cols = np.where(eligible_mask & (result > dem))

    valid_changes = changes[modified]

    audit = {
        "method": "confined_priority_flood_fill",
        "connectivity": connectivity,
        "spill_elevation": float(spill_elevation),
        "mask_cells": int(np.sum(depression_mask)),
        "eligible_cells": int(np.sum(eligible_mask)),
        "nodata_cells": int(np.sum(depression_mask & ~valid_mask)),
        "modified_cells": int(np.sum(modified)),
        "total_elevation_change": (
            float(np.sum(valid_changes)) if valid_changes.size else 0.0
        ),
        "maximum_change": float(np.max(valid_changes)) if valid_changes.size else 0.0,
        "minimum_change": float(np.min(changes)) if changes.size else 0.0,
        "modified_cell_coordinates": [
            {
                "row": int(row),
                "column": int(col),
                "original_elevation": float(dem[row, col]),
                "new_elevation": float(result[row, col]),
                "elevation_change": float(result[row, col] - dem[row, col]),
            }
            for row, col in zip(
                modified_rows.tolist(),
                modified_cols.tolist(),
            )
        ],
    }

    return result, audit


def _validate_inputs(
    dem: np.ndarray,
    depression_mask: np.ndarray,
    spill_elevation: float,
    connectivity: int,
) -> None:
    """
    Validate confined-fill inputs.

    :param dem: DEM array.
    :param depression_mask: Boolean depression footprint.
    :param spill_elevation: Target spill elevation.
    :param connectivity: Connectivity convention.
    :raises ValueError: If any input is invalid.
    """
    if not isinstance(dem, np.ndarray):
        raise ValueError("dem must be a NumPy array.")

    if dem.ndim != 2:
        raise ValueError("dem must be a two-dimensional array.")

    if not isinstance(depression_mask, np.ndarray):
        raise ValueError("depression_mask must be a NumPy array.")

    if depression_mask.ndim != 2:
        raise ValueError("depression_mask must be a two-dimensional array.")

    if dem.shape != depression_mask.shape:
        raise ValueError("dem and depression_mask must have the same shape.")

    if depression_mask.dtype != np.bool_:
        raise ValueError("depression_mask must have Boolean dtype.")

    if not np.any(depression_mask):
        raise ValueError("The depression mask is empty.")

    if not np.isfinite(spill_elevation):
        raise ValueError("spill_elevation must be finite.")

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be either 4 or 8.")
