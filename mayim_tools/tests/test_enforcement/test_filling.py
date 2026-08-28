"""Tests for native confined Priority-Flood filling.

These tests validate that filling is confined to a supplied depression
footprint. They do not use WhiteboxTools, RichDEM or TauDEM.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.enforcement.filling import (
    confined_priority_flood_fill,
)


class TestConfinedPriorityFloodFill:
    """Tests for confined_priority_flood_fill()."""

    def test_fills_only_cells_inside_the_depression_mask(self):
        """Cells inside the mask are raised to the spill elevation."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 8.0, 2.0, 8.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_mask = np.zeros_like(dem, dtype=bool)
        depression_mask[1:4, 1:4] = True

        result, audit = confined_priority_flood_fill(
            dem=dem,
            depression_mask=depression_mask,
            spill_elevation=8.0,
            nodata=-9999.0,
        )

        assert result[2, 2] == 8.0
        assert np.all(result[depression_mask] >= dem[depression_mask])
        assert audit["method"] == "confined_priority_flood_fill"

    def test_cells_outside_mask_remain_unchanged(self):
        """The function must never modify cells outside the mask."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 8.0, 2.0, 8.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_mask = np.zeros_like(dem, dtype=bool)
        depression_mask[1:4, 1:4] = True

        result, _ = confined_priority_flood_fill(
            dem=dem,
            depression_mask=depression_mask,
            spill_elevation=8.0,
            nodata=-9999.0,
        )

        outside = ~depression_mask
        assert np.array_equal(result[outside], dem[outside])

    def test_input_array_is_not_modified(self):
        """The input DEM must not be modified in place."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        original = dem.copy()
        depression_mask = np.zeros_like(dem, dtype=bool)
        depression_mask[1, 1] = True

        result, _ = confined_priority_flood_fill(
            dem=dem,
            depression_mask=depression_mask,
            spill_elevation=10.0,
            nodata=-9999.0,
        )

        assert np.array_equal(dem, original)
        assert result is not dem

    def test_nodata_cells_are_preserved(self):
        """NoData cells must remain unchanged."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 2.0, -9999.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_mask = np.zeros_like(dem, dtype=bool)
        depression_mask[1, 1:3] = True

        result, audit = confined_priority_flood_fill(
            dem=dem,
            depression_mask=depression_mask,
            spill_elevation=10.0,
            nodata=-9999.0,
        )

        assert result[1, 2] == -9999.0
        assert audit["nodata_cells"] == 1

    def test_filling_never_lowers_elevation(self):
        """The operation must never lower a valid elevation."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_mask = np.zeros_like(dem, dtype=bool)
        depression_mask[1, 1] = True

        result, _ = confined_priority_flood_fill(
            dem=dem,
            depression_mask=depression_mask,
            spill_elevation=10.0,
            nodata=-9999.0,
        )

        valid = dem != -9999.0
        assert np.all(result[valid] >= dem[valid])

    def test_no_op_when_cells_are_already_at_spill_level(self):
        """No changes occur when the depression is already filled."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 8.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_mask = np.zeros_like(dem, dtype=bool)
        depression_mask[1, 1] = True

        result, audit = confined_priority_flood_fill(
            dem=dem,
            depression_mask=depression_mask,
            spill_elevation=8.0,
            nodata=-9999.0,
        )

        assert np.array_equal(result, dem)
        assert audit["modified_cells"] == 0

    def test_empty_mask_is_rejected(self):
        """An empty depression mask must be rejected."""
        dem = np.ones((3, 3), dtype=np.float64)
        depression_mask = np.zeros_like(dem, dtype=bool)

        with pytest.raises(ValueError, match="empty"):
            confined_priority_flood_fill(
                dem=dem,
                depression_mask=depression_mask,
                spill_elevation=10.0,
                nodata=-9999.0,
            )

    def test_mask_shape_must_match_dem(self):
        """The mask must have the same dimensions as the DEM."""
        dem = np.ones((3, 3), dtype=np.float64)
        depression_mask = np.ones((2, 2), dtype=bool)

        with pytest.raises(ValueError, match="shape"):
            confined_priority_flood_fill(
                dem=dem,
                depression_mask=depression_mask,
                spill_elevation=10.0,
                nodata=-9999.0,
            )

    def test_invalid_connectivity_is_rejected(self):
        """Only four- or eight-connected processing is valid."""
        dem = np.ones((3, 3), dtype=np.float64)
        depression_mask = np.ones_like(dem, dtype=bool)

        with pytest.raises(ValueError, match="connectivity"):
            confined_priority_flood_fill(
                dem=dem,
                depression_mask=depression_mask,
                spill_elevation=10.0,
                nodata=-9999.0,
                connectivity=6,
            )

    def test_audit_records_change_statistics(self):
        """The audit record contains useful modification statistics."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_mask = np.zeros_like(dem, dtype=bool)
        depression_mask[1, 1] = True

        _, audit = confined_priority_flood_fill(
            dem=dem,
            depression_mask=depression_mask,
            spill_elevation=10.0,
            nodata=-9999.0,
        )

        assert audit["modified_cells"] == 1
        assert audit["total_elevation_change"] == 8.0
        assert audit["maximum_change"] == 8.0
        assert audit["spill_elevation"] == 10.0
