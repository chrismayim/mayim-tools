"""
Tests for native Stage 7 adaptive hydrography enforcement.

These tests use synthetic DEM and Boolean-mask arrays. They do not use
WhiteboxTools, RichDEM, TauDEM or another hydrological implementation.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.hydrography.enforcement import (
    enforce_hydrography,
)


class TestEnforceHydrography:
    """Tests for enforce_hydrography()."""

    def test_eligible_hydrography_cells_are_lowered(self):
        """Eligible mapped hydrography cells are enforced."""
        dem = np.full(
            (5, 5),
            100.0,
            dtype=np.float64,
        )

        hydrography_mask = np.zeros_like(dem, dtype=bool)
        hydrography_mask[2, 1:4] = True

        eligible_mask = hydrography_mask.copy()

        result, difference, mask, audit = enforce_hydrography(
            dem=dem,
            hydrography_mask=hydrography_mask,
            eligible_mask=eligible_mask,
            cell_size=1.0,
            vertical_accuracy=2.0,
            maximum_burn_depth=5.0,
            nodata=-9999.0,
        )

        assert np.all(result[hydrography_mask] < dem[hydrography_mask])
        assert np.all(difference[hydrography_mask] < 0.0)
        assert np.all(mask[hydrography_mask] == 1)
        assert audit["modified_cells"] == 3
        assert audit["maximum_burn_depth"] == 2.0

    def test_non_eligible_hydrography_is_not_modified(self):
        """Mapped hydrography outside the eligible mask is preserved."""
        dem = np.full(
            (5, 5),
            100.0,
            dtype=np.float64,
        )

        hydrography_mask = np.zeros_like(dem, dtype=bool)
        hydrography_mask[2, 1:4] = True

        eligible_mask = np.zeros_like(dem, dtype=bool)
        eligible_mask[2, 2] = True

        result, difference, mask, audit = enforce_hydrography(
            dem=dem,
            hydrography_mask=hydrography_mask,
            eligible_mask=eligible_mask,
            cell_size=1.0,
            vertical_accuracy=2.0,
            maximum_burn_depth=5.0,
            nodata=-9999.0,
        )

        assert result[2, 1] == dem[2, 1]
        assert result[2, 3] == dem[2, 3]
        assert result[2, 2] < dem[2, 2]
        assert difference[2, 1] == 0.0
        assert difference[2, 3] == 0.0
        assert mask[2, 1] == 2
        assert mask[2, 3] == 2
        assert audit["modified_cells"] == 1

    def test_input_dem_is_not_modified(self):
        """The original DEM remains unchanged."""
        dem = np.full(
            (5, 5),
            100.0,
            dtype=np.float64,
        )

        hydrography_mask = np.zeros_like(dem, dtype=bool)
        hydrography_mask[2, 2] = True

        original = dem.copy()

        enforce_hydrography(
            dem=dem,
            hydrography_mask=hydrography_mask,
            eligible_mask=hydrography_mask,
            cell_size=1.0,
            vertical_accuracy=2.0,
            maximum_burn_depth=5.0,
            nodata=-9999.0,
        )

        assert np.array_equal(dem, original)

    def test_burn_depth_is_bounded(self):
        """Burn depth cannot exceed configured bounds."""
        dem = np.full(
            (5, 5),
            100.0,
            dtype=np.float64,
        )

        hydrography_mask = np.zeros_like(dem, dtype=bool)
        hydrography_mask[2, 2] = True

        result, difference, _, audit = enforce_hydrography(
            dem=dem,
            hydrography_mask=hydrography_mask,
            eligible_mask=hydrography_mask,
            cell_size=10.0,
            vertical_accuracy=1.5,
            maximum_burn_depth=10.0,
            nodata=-9999.0,
        )

        assert result[2, 2] >= 98.5
        assert difference[2, 2] >= -1.5
        assert audit["maximum_burn_depth"] == 1.5

    def test_nodata_cells_are_preserved(self):
        """NoData cells are never modified."""
        dem = np.full(
            (5, 5),
            100.0,
            dtype=np.float64,
        )
        dem[2, 2] = -9999.0

        hydrography_mask = np.zeros_like(dem, dtype=bool)
        hydrography_mask[2, 2] = True

        result, difference, mask, _ = enforce_hydrography(
            dem=dem,
            hydrography_mask=hydrography_mask,
            eligible_mask=hydrography_mask,
            cell_size=1.0,
            vertical_accuracy=2.0,
            maximum_burn_depth=5.0,
            nodata=-9999.0,
        )

        assert result[2, 2] == -9999.0
        assert difference[2, 2] == 0.0
        assert mask[2, 2] == 255

    def test_conflict_cells_are_excluded(self):
        """Cells excluded from eligibility are not enforced."""
        dem = np.full(
            (5, 5),
            100.0,
            dtype=np.float64,
        )

        hydrography_mask = np.zeros_like(dem, dtype=bool)
        hydrography_mask[2, 1:4] = True

        eligible_mask = np.zeros_like(dem, dtype=bool)
        eligible_mask[2, 1] = True
        eligible_mask[2, 2] = True

        conflict_mask = np.zeros_like(dem, dtype=bool)
        conflict_mask[2, 2] = True

        eligible_mask[conflict_mask] = False

        result, _, mask, audit = enforce_hydrography(
            dem=dem,
            hydrography_mask=hydrography_mask,
            eligible_mask=eligible_mask,
            cell_size=1.0,
            vertical_accuracy=2.0,
            maximum_burn_depth=5.0,
            nodata=-9999.0,
        )

        assert result[2, 1] < dem[2, 1]
        assert result[2, 2] == dem[2, 2]
        assert mask[2, 2] != 1
        assert audit["modified_cells"] == 1

    def test_upstream_area_can_scale_burn_depth(self):
        """Larger contributing area receives a larger bounded depth."""
        dem = np.full(
            (5, 5),
            100.0,
            dtype=np.float64,
        )

        hydrography_mask = np.zeros_like(dem, dtype=bool)
        hydrography_mask[2, 1:4] = True

        upstream_area = np.zeros_like(dem, dtype=np.float64)
        upstream_area[2, 1] = 10.0
        upstream_area[2, 2] = 50.0
        upstream_area[2, 3] = 100.0

        result, difference, _, audit = enforce_hydrography(
            dem=dem,
            hydrography_mask=hydrography_mask,
            eligible_mask=hydrography_mask,
            cell_size=1.0,
            vertical_accuracy=2.0,
            maximum_burn_depth=5.0,
            nodata=-9999.0,
            upstream_area=upstream_area,
            reference_upstream_area=100.0,
        )

        depths = -difference[2, 1:4]

        assert depths[0] < depths[1]
        assert depths[1] < depths[2]
        assert np.all(depths <= audit["maximum_burn_depth"])
        assert np.all(result[hydrography_mask] <= dem[hydrography_mask])

    def test_invalid_mask_shapes_are_rejected(self):
        """Input masks must match the DEM shape."""
        dem = np.ones(
            (5, 5),
            dtype=np.float64,
        )
        hydrography_mask = np.ones(
            (4, 4),
            dtype=bool,
        )
        eligible_mask = np.ones(
            (5, 5),
            dtype=bool,
        )

        with pytest.raises(ValueError, match="same shape"):
            enforce_hydrography(
                dem=dem,
                hydrography_mask=hydrography_mask,
                eligible_mask=eligible_mask,
                cell_size=1.0,
                vertical_accuracy=1.0,
                maximum_burn_depth=1.0,
                nodata=-9999.0,
            )

    def test_invalid_parameters_are_rejected(self):
        """Physical and processing bounds must be positive."""
        dem = np.ones(
            (5, 5),
            dtype=np.float64,
        )
        mask = np.ones(
            (5, 5),
            dtype=bool,
        )

        with pytest.raises(ValueError, match="cell_size"):
            enforce_hydrography(
                dem=dem,
                hydrography_mask=mask,
                eligible_mask=mask,
                cell_size=0.0,
                vertical_accuracy=1.0,
                maximum_burn_depth=1.0,
                nodata=-9999.0,
            )

        with pytest.raises(ValueError, match="vertical_accuracy"):
            enforce_hydrography(
                dem=dem,
                hydrography_mask=mask,
                eligible_mask=mask,
                cell_size=1.0,
                vertical_accuracy=0.0,
                maximum_burn_depth=1.0,
                nodata=-9999.0,
            )

        with pytest.raises(ValueError, match="maximum_burn_depth"):
            enforce_hydrography(
                dem=dem,
                hydrography_mask=mask,
                eligible_mask=mask,
                cell_size=1.0,
                vertical_accuracy=1.0,
                maximum_burn_depth=0.0,
                nodata=-9999.0,
            )
