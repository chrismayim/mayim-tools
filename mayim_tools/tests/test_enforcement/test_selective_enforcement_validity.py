"""
Regression and diagnostic tests for Stage 5 selective flow enforcement.

These tests use small synthetic DEMs to verify the native Mayim
enforcement modules without running the full QGIS workflow.

Purpose
-------
1. Confirm the current behaviour of Stage 5 helper functions.
2. Provide a reproducible baseline before changing production code.
3. Document known hydrological limitations explicitly.

Important
---------
One test is intentionally marked xfail because the current single-cell
de-pitting behaviour does not guarantee a strictly lower downstream
neighbour for D8 routing.
"""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.hydrology.enforcement.depitting import depit_single_cell
from mayim_tools.hydrology.enforcement.enforcement import (
    ARTIFACT,
    DECISION_DEPITTED,
    DECISION_REAL_PRESERVED,
    DECISION_REVIEW_PRESERVED,
    REAL_FEATURE,
    REVIEW_REQUIRED,
    enforce_selectively,
)
from mayim_tools.hydrology.enforcement.filling import (
    confined_priority_flood_fill,
)

NODATA = -9999.0


def analyse_interior_surface(
    dem: np.ndarray,
    nodata: float = NODATA,
) -> dict[str, int]:
    """
    Count interior local minima and interior flat cells.

    Boundary cells are excluded because they may legitimately drain
    outside the raster extent. An interior cell is classified as:

    - local minimum: no valid neighbour is lower and no valid neighbour
      is equal
    - flat cell: no valid neighbour is lower and at least one valid
      neighbour is equal
    - draining cell: at least one valid neighbour is lower

    Parameters
    ----------
    dem : np.ndarray
        Two-dimensional DEM array.
    nodata : float
        NoData sentinel value.

    Returns
    -------
    dict[str, int]
        Counts of interior valid cells, interior local minima,
        interior flat cells, and interior cells with a lower neighbour.
    """
    valid = np.isfinite(dem) & (dem != nodata)
    n_rows, n_cols = dem.shape

    interior_valid_cells = 0
    interior_local_minima = 0
    interior_flat_cells = 0
    interior_cells_with_lower_neighbour = 0

    for row in range(1, n_rows - 1):
        for col in range(1, n_cols - 1):
            if not valid[row, col]:
                continue

            interior_valid_cells += 1
            centre = float(dem[row, col])

            lower_found = False
            equal_found = False
            valid_neighbour_found = False

            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue

                    neighbour_row = row + row_offset
                    neighbour_col = col + col_offset

                    if not valid[neighbour_row, neighbour_col]:
                        continue

                    valid_neighbour_found = True
                    neighbour = float(dem[neighbour_row, neighbour_col])

                    if neighbour < centre:
                        lower_found = True
                        break

                    if neighbour == centre:
                        equal_found = True

                if lower_found:
                    break

            if lower_found:
                interior_cells_with_lower_neighbour += 1
            elif equal_found:
                interior_flat_cells += 1
            elif valid_neighbour_found:
                interior_local_minima += 1

    return {
        "interior_valid_cells": interior_valid_cells,
        "interior_local_minima": interior_local_minima,
        "interior_flat_cells": interior_flat_cells,
        "interior_cells_with_lower_neighbour": (interior_cells_with_lower_neighbour),
    }


def test_depit_single_cell_raises_pit_to_lowest_neighbour() -> None:
    """
    A single-cell pit should be raised to its lowest valid neighbour.

    This test documents the current behaviour of depit_single_cell().
    It does not yet require a strictly lower downstream neighbour.
    """
    dem = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, 5.0, 10.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )

    result, audit = depit_single_cell(
        dem=dem,
        pit=(1, 1),
        nodata=NODATA,
    )

    assert result[1, 1] == 10.0
    assert audit["modified"] is True
    assert audit["original_elevation"] == 5.0
    assert audit["new_elevation"] == 10.0
    assert audit["lowest_neighbour_elevation"] == 10.0


@pytest.mark.xfail(
    reason=(
        "Current depitting behaviour removes the local minimum but may "
        "leave an interior flat cell, so the result is not guaranteed "
        "to be D8-ready."
    ),
    strict=False,
)
def test_depit_single_cell_is_not_yet_strictly_d8_ready() -> None:
    """
    Document the known limitation of single-cell depitting.

    The current algorithm raises the pit to the lowest neighbour, which
    can leave a flat rather than a strictly draining cell.
    """
    dem = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, 5.0, 10.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )

    result, _audit = depit_single_cell(
        dem=dem,
        pit=(1, 1),
        nodata=NODATA,
    )

    stats = analyse_interior_surface(result, nodata=NODATA)

    assert stats["interior_local_minima"] == 0
    assert stats["interior_flat_cells"] == 0


def test_confined_priority_flood_fill_raises_only_masked_cells() -> None:
    """
    Confined fill should raise only cells inside the supplied mask.

    Cells outside the depression mask must remain unchanged.
    """
    dem = np.array(
        [
            [12.0, 12.0, 12.0, 12.0],
            [12.0, 8.0, 8.0, 12.0],
            [12.0, 8.0, 9.0, 12.0],
            [12.0, 12.0, 12.0, 12.0],
        ],
        dtype=np.float64,
    )

    depression_mask = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    result, audit = confined_priority_flood_fill(
        dem=dem,
        depression_mask=depression_mask,
        spill_elevation=11.0,
        nodata=NODATA,
        connectivity=8,
    )

    expected = np.array(
        [
            [12.0, 12.0, 12.0, 12.0],
            [12.0, 11.0, 11.0, 12.0],
            [12.0, 11.0, 11.0, 12.0],
            [12.0, 12.0, 12.0, 12.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected)
    assert audit["modified_cells"] == 4
    assert audit["spill_elevation"] == 11.0


def test_enforce_selectively_preserves_real_feature() -> None:
    """
    REAL_FEATURE depressions must be preserved unchanged.
    """
    dem = np.array(
        [
            [15.0, 15.0, 15.0],
            [15.0, 10.0, 15.0],
            [15.0, 15.0, 15.0],
        ],
        dtype=np.float64,
    )

    depression_mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    records = {
        1: {
            "classification": REAL_FEATURE,
            "pit_row": 1,
            "pit_col": 1,
            "spill_elevation": 15.0,
            "area_cells": 1,
        }
    }

    masks = {1: depression_mask}

    preserved_dem, hydrology_ready_dem, decision_codes, audits = enforce_selectively(
        dem=dem,
        depression_records=records,
        depression_masks=masks,
        max_breach_length=10,
        max_breach_depth=2.0,
        nodata=NODATA,
        connectivity=8,
    )

    np.testing.assert_allclose(preserved_dem, dem)
    np.testing.assert_allclose(hydrology_ready_dem, dem)
    assert decision_codes[1, 1] == DECISION_REAL_PRESERVED
    assert audits[0]["decision"] == "preserved_real_feature"
    assert audits[0]["modified"] is False


def test_enforce_selectively_preserves_review_required() -> None:
    """
    REVIEW_REQUIRED depressions must be preserved unchanged.
    """
    dem = np.array(
        [
            [15.0, 15.0, 15.0],
            [15.0, 10.0, 15.0],
            [15.0, 15.0, 15.0],
        ],
        dtype=np.float64,
    )

    depression_mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    records = {
        1: {
            "classification": REVIEW_REQUIRED,
            "pit_row": 1,
            "pit_col": 1,
            "spill_elevation": 15.0,
            "area_cells": 1,
        }
    }

    masks = {1: depression_mask}

    preserved_dem, hydrology_ready_dem, decision_codes, audits = enforce_selectively(
        dem=dem,
        depression_records=records,
        depression_masks=masks,
        max_breach_length=10,
        max_breach_depth=2.0,
        nodata=NODATA,
        connectivity=8,
    )

    np.testing.assert_allclose(preserved_dem, dem)
    np.testing.assert_allclose(hydrology_ready_dem, dem)
    assert decision_codes[1, 1] == DECISION_REVIEW_PRESERVED
    assert audits[0]["decision"] == "preserved_pending_review"
    assert audits[0]["modified"] is False


def test_enforce_selectively_depits_single_cell_artifact() -> None:
    """
    A single-cell ARTIFACT depression should be de-pitted.
    """
    dem = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, 5.0, 10.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )

    depression_mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    records = {
        1: {
            "classification": ARTIFACT,
            "pit_row": 1,
            "pit_col": 1,
            "spill_elevation": 10.0,
            "area_cells": 1,
        }
    }

    masks = {1: depression_mask}

    preserved_dem, hydrology_ready_dem, decision_codes, audits = enforce_selectively(
        dem=dem,
        depression_records=records,
        depression_masks=masks,
        max_breach_length=10,
        max_breach_depth=2.0,
        nodata=NODATA,
        connectivity=8,
    )

    assert preserved_dem[1, 1] == 10.0
    assert hydrology_ready_dem[1, 1] == 10.0
    assert decision_codes[1, 1] == DECISION_DEPITTED
    assert audits[0]["decision"] == "depitted"


def test_analyse_interior_surface_counts_local_minimum() -> None:
    """
    The diagnostic helper should detect one interior local minimum.
    """
    dem = np.array(
        [
            [12.0, 12.0, 12.0],
            [12.0, 10.0, 12.0],
            [12.0, 12.0, 12.0],
        ],
        dtype=np.float64,
    )

    stats = analyse_interior_surface(dem, nodata=NODATA)

    assert stats["interior_valid_cells"] == 1
    assert stats["interior_local_minima"] == 1
    assert stats["interior_flat_cells"] == 0
    assert stats["interior_cells_with_lower_neighbour"] == 0


def test_analyse_interior_surface_counts_flat_cell() -> None:
    """
    The diagnostic helper should detect an interior flat cell.
    """
    dem = np.array(
        [
            [12.0, 12.0, 12.0, 12.0],
            [12.0, 10.0, 10.0, 12.0],
            [12.0, 12.0, 12.0, 12.0],
        ],
        dtype=np.float64,
    )

    stats = analyse_interior_surface(dem, nodata=NODATA)

    assert stats["interior_valid_cells"] == 2
    assert stats["interior_local_minima"] == 0
    assert stats["interior_flat_cells"] == 2
    assert stats["interior_cells_with_lower_neighbour"] == 0
