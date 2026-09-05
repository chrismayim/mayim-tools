"""
Mayim Tools - Stage 5 Selective Flow Enforcement
=================================================

Orchestrates native Stage 5 depression enforcement decisions.

The orchestrator consumes:

    - A DEM array.
    - Depression feature records.
    - Depression classification results.
    - Depression masks.
    - Pit coordinates.
    - Spill elevations.

It applies the following decision sequence:

    ARTIFACT:
        Single-cell pit -> de-pit.
        Otherwise -> constrained breach.
        Breach failure -> confined fill.

    REAL_FEATURE:
        Preserve unchanged.

    REVIEW_REQUIRED:
        Preserve unchanged and record analyst review.

This module produces a depression-preserving surface and a
hydrology-ready surface. The two surfaces are deliberately separate.

IP status
---------
Original Mayim orchestration code.

This module uses native Mayim implementations only:

    - depitting.py
    - breaching.py
    - filling.py

It does not import or call WhiteboxTools, RichDEM, TauDEM or any other
third-party hydrological implementation.

The decision logic is based on the revised Mayim research methodology
and the published literature cited by that methodology.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mayim_tools.hydrology.enforcement.breaching import (
    apply_breach_path,
    least_cost_breach,
)
from mayim_tools.hydrology.enforcement.depitting import depit_single_cell
from mayim_tools.hydrology.enforcement.filling import (
    confined_priority_flood_fill,
)

ARTIFACT = "ARTIFACT"
REAL_FEATURE = "REAL_FEATURE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"

DECISION_UNCHANGED = 0
DECISION_DEPITTED = 1
DECISION_BREACHED = 2
DECISION_FILLED = 3
DECISION_REAL_PRESERVED = 4
DECISION_REVIEW_PRESERVED = 5


def enforce_selectively(
    dem: np.ndarray,
    depression_records: Mapping[int, Mapping[str, Any]],
    depression_masks: Mapping[int, np.ndarray],
    max_breach_length: int,
    max_breach_depth: float,
    nodata: float,
    connectivity: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Apply classification-driven Stage 5 selective enforcement.

    The function returns two surfaces:

        preserved_dem:
            A depression-preserving surface. REAL_FEATURE and
            REVIEW_REQUIRED depressions remain unchanged.

        hydrology_ready_dem:
            A fully enforced derivative. In the current implementation,
            REAL_FEATURE and REVIEW_REQUIRED depressions also remain
            unchanged because they must not be silently modified.
            A future explicit user-approved override may produce a
            separately documented fully drained derivative.

        decision_codes:
            Raster containing the decision applied to each cell.

        audit_records:
            One audit dictionary per depression.

    Parameters
    ----------
    dem:
        Two-dimensional DEM array.
    depression_records:
        Mapping from depression ID to a record containing at least:

            classification
            pit_row
            pit_col
            spill_elevation
            area_cells

        ``classification`` may be a string or a classification-result
        object exposing a ``classification`` attribute.

    depression_masks:
        Mapping from depression ID to Boolean masks with the same shape
        as ``dem``.

    max_breach_length:
        Maximum permitted breach path length in cells.

    max_breach_depth:
        Maximum permitted local excavation depth.

    nodata:
        NoData sentinel.

    connectivity:
        Breach connectivity. Must be 4 or 8.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]
        Preserved DEM, hydrology-ready DEM, decision-code raster and
        per-depression audit records.

    Raises
    ------
    TypeError
        If typed inputs are of the wrong type.
    ValueError
        If inputs are invalid or a depression record is incomplete.
    """
    _validate_inputs(
        dem=dem,
        depression_records=depression_records,
        depression_masks=depression_masks,
        max_breach_length=max_breach_length,
        max_breach_depth=max_breach_depth,
        connectivity=connectivity,
    )

    preserved_dem = dem.astype(np.float64, copy=True)
    hydrology_ready_dem = dem.astype(np.float64, copy=True)

    valid_mask = np.isfinite(dem) & (dem != nodata)

    decision_codes = np.zeros(
        dem.shape,
        dtype=np.uint8,
    )
    decision_codes[~valid_mask] = 255

    audit_records: list[dict[str, Any]] = []

    for depression_id in sorted(depression_records, key=lambda value: int(value)):
        depression_id = int(depression_id)
        record = depression_records[depression_id]

        if depression_id not in depression_masks:
            raise ValueError(f"Missing depression mask for depression {depression_id}.")

        depression_mask = depression_masks[depression_id]

        if not isinstance(depression_mask, np.ndarray):
            raise TypeError(f"Depression mask {depression_id} must be a NumPy array.")

        if depression_mask.shape != dem.shape:
            raise ValueError(f"Depression mask {depression_id} has the wrong shape.")

        if depression_mask.dtype != np.bool_:
            raise ValueError(
                f"Depression mask {depression_id} must have Boolean dtype."
            )

        classification = _classification_value(record)
        pit = _pit_from_record(record)
        spill_elevation = _spill_from_record(record)

        cell_mask = depression_mask & valid_mask

        if not np.any(cell_mask):
            raise ValueError(f"Depression {depression_id} contains no valid cells.")

        if classification == REAL_FEATURE:
            decision_codes[cell_mask] = DECISION_REAL_PRESERVED

            audit_records.append(
                {
                    "depression_id": depression_id,
                    "classification": REAL_FEATURE,
                    "decision": "preserved_real_feature",
                    "decision_code": DECISION_REAL_PRESERVED,
                    "modified": False,
                    "method": "none",
                    "reason": "classified as real feature",
                }
            )
            continue

        if classification == REVIEW_REQUIRED:
            decision_codes[cell_mask] = DECISION_REVIEW_PRESERVED

            audit_records.append(
                {
                    "depression_id": depression_id,
                    "classification": REVIEW_REQUIRED,
                    "decision": "preserved_pending_review",
                    "decision_code": DECISION_REVIEW_PRESERVED,
                    "modified": False,
                    "method": "none",
                    "reason": "classification requires analyst review",
                }
            )
            continue

        if classification != ARTIFACT:
            raise ValueError(
                f"Unknown classification for depression "
                f"{depression_id}: {classification}"
            )

        area_cells = int(
            record.get(
                "area_cells",
                np.sum(cell_mask),
            )
        )

        if area_cells == 1:
            result, depit_audit = depit_single_cell(
                dem=preserved_dem,
                pit=pit,
                nodata=nodata,
            )

            change_mask = result != preserved_dem
            preserved_dem = result
            hydrology_ready_dem = result.copy()

            decision_codes[cell_mask] = DECISION_DEPITTED

            audit_records.append(
                {
                    "depression_id": depression_id,
                    "classification": ARTIFACT,
                    "decision": "depitted",
                    "decision_code": DECISION_DEPITTED,
                    "modified": bool(depit_audit["elevation_change"] > 0.0),
                    "method": "single_cell_depitting",
                    "details": depit_audit,
                    "changed_cell_count": int(np.sum(change_mask)),
                }
            )
            continue

        breach_path, breach_audit = least_cost_breach(
            dem=preserved_dem,
            pit=pit,
            spill_elevation=spill_elevation,
            max_length=max_breach_length,
            max_depth=max_breach_depth,
            nodata=nodata,
            connectivity=connectivity,
        )

        if breach_path is not None:
            result, apply_audit = apply_breach_path(
                dem=preserved_dem,
                path=breach_path,
                spill_elevation=spill_elevation,
                nodata=nodata,
            )

            change_mask = result != preserved_dem
            preserved_dem = result
            hydrology_ready_dem = result.copy()

            decision_codes[cell_mask] = DECISION_BREACHED

            audit_records.append(
                {
                    "depression_id": depression_id,
                    "classification": ARTIFACT,
                    "decision": "breached",
                    "decision_code": DECISION_BREACHED,
                    "modified": bool(np.any(change_mask)),
                    "method": "constrained_least_cost_breach",
                    "search": breach_audit,
                    "application": apply_audit,
                    "changed_cell_count": int(np.sum(change_mask)),
                }
            )
            continue

        filled, fill_audit = confined_priority_flood_fill(
            dem=preserved_dem,
            depression_mask=depression_mask,
            spill_elevation=spill_elevation,
            nodata=nodata,
            connectivity=connectivity,
        )

        change_mask = filled != preserved_dem
        preserved_dem = filled
        hydrology_ready_dem = filled.copy()

        decision_codes[cell_mask] = DECISION_FILLED

        audit_records.append(
            {
                "depression_id": depression_id,
                "classification": ARTIFACT,
                "decision": "filled_fallback",
                "decision_code": DECISION_FILLED,
                "modified": bool(np.any(change_mask)),
                "method": "confined_priority_flood_fill",
                "search": breach_audit,
                "application": fill_audit,
                "changed_cell_count": int(np.sum(change_mask)),
            }
        )

    return (
        preserved_dem,
        hydrology_ready_dem,
        decision_codes,
        audit_records,
    )


def _classification_value(record: Mapping[str, Any]) -> str:
    """
    Extract a classification label from a depression record.
    """
    classification = record.get("classification")

    if hasattr(classification, "classification"):
        classification = classification.classification

    if not isinstance(classification, str):
        raise TypeError("Depression classification must be a string or result object.")

    return classification


def _pit_from_record(record: Mapping[str, Any]) -> tuple[int, int]:
    """
    Extract a pit coordinate from a depression record.
    """
    if "pit" in record:
        pit = record["pit"]

        if isinstance(pit, (tuple, list)) and len(pit) == 2:
            return int(pit[0]), int(pit[1])

    if "pit_row" in record and "pit_col" in record:
        return int(record["pit_row"]), int(record["pit_col"])

    raise ValueError("Depression record must contain pit or pit_row and pit_col.")


def _spill_from_record(record: Mapping[str, Any]) -> float:
    """
    Extract and validate a spill elevation.
    """
    if "spill_elevation" not in record:
        raise ValueError("Depression record must contain spill_elevation.")

    spill_elevation = float(record["spill_elevation"])

    if not np.isfinite(spill_elevation):
        raise ValueError("spill_elevation must be finite.")

    return spill_elevation


def _validate_inputs(
    dem: np.ndarray,
    depression_records: Mapping[int, Mapping[str, Any]],
    depression_masks: Mapping[int, np.ndarray],
    max_breach_length: int,
    max_breach_depth: float,
    connectivity: int,
) -> None:
    """
    Validate orchestration inputs.
    """
    if not isinstance(dem, np.ndarray):
        raise TypeError("dem must be a NumPy array.")

    if dem.ndim != 2:
        raise ValueError("dem must be a two-dimensional array.")

    if not isinstance(depression_records, Mapping):
        raise TypeError("depression_records must be a mapping.")

    if not isinstance(depression_masks, Mapping):
        raise TypeError("depression_masks must be a mapping.")

    if max_breach_length <= 0:
        raise ValueError("max_breach_length must be greater than zero.")

    if not np.isfinite(max_breach_depth) or max_breach_depth <= 0:
        raise ValueError("max_breach_depth must be finite and greater than zero.")

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be either 4 or 8.")
