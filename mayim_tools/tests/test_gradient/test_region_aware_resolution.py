"""Tests for region-aware Stage 6 flat resolution.

These tests validate that disconnected flat regions are resolved
independently. They use native Mayim code only and do not use
WhiteboxTools, RichDEM or TauDEM.
"""

import numpy as np

from mayim_tools.hydrology.gradient.gradient_resolution import (
    resolve_flats,
)


class TestRegionAwareResolution:
    """Tests for region-aware flat resolution."""

    def test_disconnected_regions_are_resolved_independently(self):
        """Two disconnected regions receive separate corrections."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 5.0, 5.0, 10.0, 7.0, 7.0, 10.0],
                [10.0, 5.0, 5.0, 10.0, 7.0, 7.0, 3.0],
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        flat_mask = np.zeros_like(dem, dtype=bool)
        flat_mask[1:3, 1:3] = True
        flat_mask[1:3, 4:6] = True

        higher_boundary = np.zeros_like(dem, dtype=bool)
        higher_boundary[1, 1] = True
        higher_boundary[1, 4] = True

        lower_boundary = np.zeros_like(dem, dtype=bool)
        lower_boundary[2, 2] = True
        lower_boundary[2, 5] = True

        region_ids = np.zeros_like(dem, dtype=np.int32)
        region_ids[1:3, 1:3] = 1
        region_ids[1:3, 4:6] = 2

        result, audit = resolve_flats(
            dem=dem,
            flat_mask=flat_mask,
            higher_boundary=higher_boundary,
            lower_boundary=lower_boundary,
            cell_size=1.0,
            vertical_accuracy=0.1,
            nodata=-9999.0,
            region_ids=region_ids,
        )

        assert np.any(result[region_ids == 1] != dem[region_ids == 1])
        assert np.any(result[region_ids == 2] != dem[region_ids == 2])

        assert np.array_equal(
            result[region_ids == 0],
            dem[region_ids == 0],
        )

        assert audit["region_count"] == 2
        assert len(audit["regions"]) == 2

    def test_non_flat_cells_remain_unchanged(self):
        """Cells outside all flat regions remain unchanged."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 5.0, 5.0, 5.0, 10.0],
                [10.0, 5.0, 5.0, 5.0, 10.0],
                [10.0, 5.0, 5.0, 5.0, 3.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        flat_mask = np.zeros_like(dem, dtype=bool)
        flat_mask[1:4, 1:4] = True

        higher_boundary = np.zeros_like(dem, dtype=bool)
        higher_boundary[1, 1] = True

        lower_boundary = np.zeros_like(dem, dtype=bool)
        lower_boundary[3, 3] = True

        region_ids = np.zeros_like(dem, dtype=np.int32)
        region_ids[1:4, 1:4] = 1

        result, _ = resolve_flats(
            dem=dem,
            flat_mask=flat_mask,
            higher_boundary=higher_boundary,
            lower_boundary=lower_boundary,
            cell_size=1.0,
            vertical_accuracy=0.1,
            nodata=-9999.0,
            region_ids=region_ids,
        )

        assert np.array_equal(
            result[~flat_mask],
            dem[~flat_mask],
        )

    def test_region_ids_must_match_dem_shape(self):
        """Region IDs must have the same shape as the DEM."""
        dem = np.ones((5, 5), dtype=np.float64)
        flat_mask = np.ones((5, 5), dtype=bool)
        higher_boundary = np.zeros((5, 5), dtype=bool)
        lower_boundary = np.ones((5, 5), dtype=bool)
        region_ids = np.ones((4, 4), dtype=np.int32)

        try:
            resolve_flats(
                dem=dem,
                flat_mask=flat_mask,
                higher_boundary=higher_boundary,
                lower_boundary=lower_boundary,
                cell_size=1.0,
                vertical_accuracy=0.1,
                nodata=-9999.0,
                region_ids=region_ids,
            )
        except ValueError as error:
            assert "region_ids" in str(error)
        else:
            raise AssertionError(
                "Expected ValueError for mismatched region_ids shape."
            )

    def test_region_ids_are_optional_for_backward_compatibility(self):
        """The original API remains valid without region IDs."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 5.0, 3.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        flat_mask = np.zeros_like(dem, dtype=bool)
        flat_mask[1, 1] = True

        higher_boundary = np.zeros_like(dem, dtype=bool)
        higher_boundary[1, 1] = True

        lower_boundary = np.zeros_like(dem, dtype=bool)
        lower_boundary[1, 1] = True

        result, audit = resolve_flats(
            dem=dem,
            flat_mask=flat_mask,
            higher_boundary=higher_boundary,
            lower_boundary=lower_boundary,
            cell_size=1.0,
            vertical_accuracy=0.1,
            nodata=-9999.0,
        )

        assert result.shape == dem.shape
        assert audit["modified_cells"] >= 0
