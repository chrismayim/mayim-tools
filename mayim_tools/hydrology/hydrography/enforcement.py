"""
Mayim Tools - Adaptive Hydrography Enforcement
===============================================

Stage 7D native hydrography enforcement.

This module applies bounded, adaptive elevation lowering to explicitly
authorised hydrography cells.

The implementation is deliberately conservative:

    - Only cells in hydrography_mask AND eligible_mask are modified.
    - NoData cells are never modified.
    - Aligned, tolerated or conflict-excluded cells are not modified.
    - Burn depth cannot exceed the configured maximum.
    - Burn depth cannot exceed the supplied vertical accuracy.
    - Upstream contributing area may scale burn depth continuously.
    - The input DEM is never modified in place.
    - A signed difference raster and audit record are returned.

This module does not:

    - Validate vector geometry.
    - Prepare network topology.
    - Rasterise vector hydrography.
    - Derive flow direction.
    - Derive flow accumulation.
    - Resolve ambiguous hydrography conflicts.

Those responsibilities belong to other Stage 7 components.

Methodology basis
-----------------
The implementation follows the adaptive and topology-aware principles
described in:

    Soille, P., Vogt, J., and Colombo, R. (2003). Carving and
    adaptive drainage enforcement of grid digital elevation models.
    Water Resources Research, 39(12), 1366.

    Lindsay, J. B. (2016). The practice of DEM stream burning revisited.
    Earth Surface Processes and Landforms, 41(5), 658-668.

It also follows the Stage 7 specification in the revised Mayim Tools
DEM Hydrological Conditioning Research Paper.

IP status
---------
Original Mayim implementation.

No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
implementation is imported or called.

Shapely, GeoPandas and raster I/O libraries may be used by other
components as generic infrastructure, but the Mayim enforcement decision,
burn-depth model, bounds and audit logic are implemented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Stage 7 enforcement-mask values.
MASK_UNCHANGED = 0
MASK_ENFORCED = 1
MASK_ELIGIBLE_NOT_MODIFIED = 2
MASK_CONFLICT_EXCLUDED = 3
MASK_NODATA = 255


def enforce_hydrography(
    dem: np.ndarray,
    hydrography_mask: np.ndarray,
    eligible_mask: np.ndarray,
    cell_size: float,
    vertical_accuracy: float,
    maximum_burn_depth: float,
    nodata: float,
    upstream_area: np.ndarray | None = None,
    reference_upstream_area: float | None = None,
    conflict_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Apply bounded adaptive hydrography enforcement.

    The output elevation is calculated as:

        enforced_elevation = original_elevation - burn_depth

    The base burn depth is:

        min(maximum_burn_depth, vertical_accuracy)

    If upstream area is supplied, the depth is scaled using:

        area_factor = min(upstream_area / reference_upstream_area, 1.0)

    Therefore:

        burn_depth = base_burn_depth * area_factor

    Cells with zero upstream area receive zero burn depth when an
    upstream-area raster is supplied.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.

    hydrography_mask:
        Boolean array identifying cells occupied by mapped hydrography.

    eligible_mask:
        Boolean array identifying cells explicitly authorised for
        enforcement. This should normally exclude aligned, tolerated,
        conflicted and review-required cells.

    cell_size:
        Mean DEM cell size in map units. Recorded in the audit output.

    vertical_accuracy:
        DEM vertical accuracy in elevation units, normally RMSE or a
        conservative source-based estimate.

    maximum_burn_depth:
        Absolute maximum permitted lowering in elevation units.

    nodata:
        DEM NoData sentinel.

    upstream_area:
        Optional numeric raster of upstream contributing area. Must
        have the same shape as dem if supplied.

    reference_upstream_area:
        Positive reference area used to normalise upstream_area.
        Required when upstream_area is supplied.

    conflict_mask:
        Optional Boolean mask identifying conflict or review cells.
        Conflict cells are excluded even when eligible_mask is True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
        A tuple containing:

            enforced_dem
                New DEM array.

            difference
                Signed difference:
                enforced_dem - dem.

            enforcement_mask
                uint8 raster using:
                    0   unchanged
                    1   enforced
                    2   hydrography eligible but not modified
                    3   conflict/review excluded
                    255 NoData

            audit
                JSON-serialisable audit record.

    Raises
    ------
    ValueError
        If any input is invalid.

    Important
    ---------
    Material divergence is not automatically sufficient to authorise
    enforcement. The caller must construct eligible_mask using the
    preceding validation, topology and divergence stages.
    """
    _validate_inputs(
        dem=dem,
        hydrography_mask=hydrography_mask,
        eligible_mask=eligible_mask,
        cell_size=cell_size,
        vertical_accuracy=vertical_accuracy,
        maximum_burn_depth=maximum_burn_depth,
        upstream_area=upstream_area,
        reference_upstream_area=reference_upstream_area,
        conflict_mask=conflict_mask,
    )

    result = dem.astype(np.float64, copy=True)

    valid_mask = (
        np.isfinite(dem)
        & (dem != nodata)
    )

    authorised_mask = (
        hydrography_mask
        & eligible_mask
        & valid_mask
    )

    if conflict_mask is not None:
        authorised_mask &= ~conflict_mask

    eligible_not_modified = (
        hydrography_mask
        & valid_mask
        & eligible_mask
        & ~authorised_mask
    )

    valid_hydrography = (
        hydrography_mask
        & valid_mask
    )

    eligible_not_modified = (
        valid_hydrography
        & eligible_mask
        & ~authorised_mask
    )

    not_enforced_hydrography = (
        valid_hydrography
        & ~authorised_mask
    )

    excluded_hydrography = np.zeros_like(
        hydrography_mask,
        dtype=bool,
    )

    if conflict_mask is not None:
        excluded_hydrography = (
            valid_hydrography
            & conflict_mask
        )


    if conflict_mask is not None:
        excluded_hydrography = (
            hydrography_mask
            & valid_mask
            & conflict_mask
        )

    base_burn_depth = min(
        float(maximum_burn_depth),
        float(vertical_accuracy),
    )

    if upstream_area is None:
        burn_depth = np.zeros_like(
            dem,
            dtype=np.float64,
        )
        burn_depth[authorised_mask] = base_burn_depth
        area_factor = None
    else:
        area_factor = np.zeros_like(
            dem,
            dtype=np.float64,
        )

        area_factor[authorised_mask] = np.clip(
            upstream_area[authorised_mask]
            / float(reference_upstream_area),
            0.0,
            1.0,
        )

        burn_depth = (
            base_burn_depth
            * area_factor
        )

    original_authorised = dem[authorised_mask].astype(
        np.float64,
        copy=True,
    )

    authorised_depths = burn_depth[authorised_mask]

    result[authorised_mask] = (
        original_authorised
        - authorised_depths
    )

    difference = np.zeros_like(
        dem,
        dtype=np.float64,
    )
    difference[valid_mask] = (
        result[valid_mask]
        - dem[valid_mask]
    )
    difference[~valid_mask] = 0.0

    modified_mask = (
        authorised_mask
        & (np.abs(difference) > 0.0)
    )

    enforcement_mask = np.zeros(
        dem.shape,
        dtype=np.uint8,
    )

    # Every valid hydrography cell not modified receives code 2.
    # This includes hydrography that was not eligible and eligible
    # hydrography for which no effective change was required.
    enforcement_mask[not_enforced_hydrography] = (
        MASK_ELIGIBLE_NOT_MODIFIED
    )

    # Explicit conflicts override the general non-enforced code.
    enforcement_mask[excluded_hydrography] = (
        MASK_CONFLICT_EXCLUDED
    )

    # Actual modifications take precedence over code 2.
    enforcement_mask[modified_mask] = MASK_ENFORCED

    # NoData always has the highest precedence.
    enforcement_mask[~valid_mask] = MASK_NODATA

    modified_depths = -difference[modified_mask]

    if modified_depths.size:
        total_lowering = float(np.sum(modified_depths))
        maximum_lowering = float(np.max(modified_depths))
        mean_lowering = float(np.mean(modified_depths))
        minimum_lowering = float(np.min(modified_depths))
    else:
        total_lowering = 0.0
        maximum_lowering = 0.0
        mean_lowering = 0.0
        minimum_lowering = 0.0

    modified_coordinates = [
        {
            "row": int(row),
            "column": int(column),
            "original_elevation": float(
                dem[row, column]
            ),
            "new_elevation": float(
                result[row, column]
            ),
            "elevation_change": float(
                difference[row, column]
            ),
            "burn_depth": float(
                -difference[row, column]
            ),
        }
        for row, column in zip(
            *np.where(modified_mask)
        )
    ]

    audit: dict[str, Any] = {
        "method": "adaptive_topology_aware_hydrography_enforcement",
        "success": True,
        "cell_size": float(cell_size),
        "vertical_accuracy": float(vertical_accuracy),
        "maximum_configured_burn_depth": float(
            maximum_burn_depth
        ),
        "maximum_burn_depth": float(base_burn_depth),
        "base_burn_depth": float(base_burn_depth),
        "reference_upstream_area": (
            float(reference_upstream_area)
            if reference_upstream_area is not None
            else None
        ),
        "input_hydrography_cells": int(
            np.sum(hydrography_mask & valid_mask)
        ),
        "eligible_hydrography_cells": int(
            np.sum(
                hydrography_mask
                & valid_mask
                & eligible_mask
            )
        ),
        "authorised_hydrography_cells": int(
            np.sum(authorised_mask)
        ),
        "eligible_not_modified_cells": int(
            np.sum(eligible_not_modified)
        ),
        "excluded_hydrography_cells": int(
            np.sum(excluded_hydrography)
        ),
        "modified_cells": int(
            np.sum(modified_mask)
        ),
        "conflict_excluded_cells": int(
            np.sum(
                enforcement_mask == MASK_CONFLICT_EXCLUDED
            )
        ),
        "nodata_cells": int(
            np.sum(~valid_mask)
        ),
        "total_lowering": total_lowering,
        "maximum_lowering": maximum_lowering,
        "mean_lowering": mean_lowering,
        "minimum_lowering": minimum_lowering,
        "total_signed_change": float(
            np.sum(difference[valid_mask])
        ),
        "area_scaling_used": upstream_area is not None,
        "modified_cell_records": modified_coordinates,
        "mask_values": {
            "0": "unchanged non-hydrography cell",
            "1": "hydrography-enforced",
            "2": "hydrography present but not enforced",
            "3": "conflict/review excluded",
            "255": "NoData",
        },
    }

    if area_factor is not None:
        valid_area_factors = area_factor[authorised_mask]
        audit["area_factor_minimum"] = float(
            np.min(valid_area_factors)
        ) if valid_area_factors.size else 0.0
        audit["area_factor_maximum"] = float(
            np.max(valid_area_factors)
        ) if valid_area_factors.size else 0.0

    return result, difference, enforcement_mask, audit


def _validate_inputs(
    dem: np.ndarray,
    hydrography_mask: np.ndarray,
    eligible_mask: np.ndarray,
    cell_size: float,
    vertical_accuracy: float,
    maximum_burn_depth: float,
    upstream_area: np.ndarray | None,
    reference_upstream_area: float | None,
    conflict_mask: np.ndarray | None,
) -> None:
    """
    Validate adaptive-enforcement inputs.
    """
    if not isinstance(dem, np.ndarray):
        raise ValueError(
            "dem must be a NumPy array."
        )

    if dem.ndim != 2:
        raise ValueError(
            "dem must be a two-dimensional array."
        )

    _validate_boolean_mask(
        hydrography_mask,
        dem.shape,
        "hydrography_mask",
    )

    _validate_boolean_mask(
        eligible_mask,
        dem.shape,
        "eligible_mask",
    )

    _validate_positive_finite(
        cell_size,
        "cell_size",
    )

    _validate_positive_finite(
        vertical_accuracy,
        "vertical_accuracy",
    )

    _validate_positive_finite(
        maximum_burn_depth,
        "maximum_burn_depth",
    )

    if upstream_area is not None:
        if not isinstance(upstream_area, np.ndarray):
            raise ValueError(
                "upstream_area must be a NumPy array or None."
            )

        if upstream_area.ndim != 2:
            raise ValueError(
                "upstream_area must be two-dimensional."
            )

        if upstream_area.shape != dem.shape:
            raise ValueError(
                "upstream_area must have the same shape as dem."
            )

        if not np.all(
            np.isfinite(upstream_area)
            | np.isnan(upstream_area)
        ):
            raise ValueError(
                "upstream_area contains invalid values."
            )

        if reference_upstream_area is None:
            raise ValueError(
                "reference_upstream_area is required when "
                "upstream_area is supplied."
            )

        _validate_positive_finite(
            reference_upstream_area,
            "reference_upstream_area",
        )

        if np.any(
            np.isfinite(upstream_area)
            & (upstream_area < 0.0)
        ):
            raise ValueError(
                "upstream_area cannot contain negative values."
            )

    if conflict_mask is not None:
        _validate_boolean_mask(
            conflict_mask,
            dem.shape,
            "conflict_mask",
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
            f"{name} must be two-dimensional."
        )

    if mask.shape != expected_shape:
        raise ValueError(
            f"{name} must have the same shape as dem."
        )

    if mask.dtype != np.bool_:
        raise ValueError(
            f"{name} must have Boolean dtype."
        )


def _validate_positive_finite(
    value: float,
    name: str,
) -> None:
    """
    Validate a positive finite numeric value.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be numeric."
        ) from error

    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be finite and greater than zero."
        )
