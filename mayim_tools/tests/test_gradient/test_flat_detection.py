"""Tests for native Stage 6 flat-surface detection.

These tests do not use WhiteboxTools, RichDEM or TauDEM.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.gradient.flat_detection import (
    detect_flats,
)


class TestDetectFlats:
    """Tests for detect_flats()."""

    def test_monotonic_dem_has_no_flats(self):
        """A strictly descending DEM with no equal neighbours has no flats."""
        dem = np.array(
            [
                [16.0, 15.0, 14.0, 13.0],
                [12.0, 11.0, 10.0, 9.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        assert not np.any(flat_mask)
        assert not np.any(higher_boundary)
        assert not np.any(lower_boundary)


    def test_flat_interior_is_detected(self):
        """A flat interior surrounded by higher and lower terrain
        is correctly identified."""
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

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        assert np.any(flat_mask)
        assert np.any(higher_boundary)
        assert np.any(lower_boundary)

    def test_nodata_cells_are_not_flat(self):
        """NoData cells must not appear in any output mask."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, -9999.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        assert not flat_mask[1, 1]
        assert not higher_boundary[1, 1]
        assert not lower_boundary[1, 1]

    def test_output_shapes_match_input(self):
        """All output arrays have the same shape as the input DEM."""
        dem = np.ones((6, 8), dtype=np.float64)

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        assert flat_mask.shape == dem.shape
        assert higher_boundary.shape == dem.shape
        assert lower_boundary.shape == dem.shape
        assert flat_mask.dtype == np.bool_
        assert higher_boundary.dtype == np.bool_
        assert lower_boundary.dtype == np.bool_

    def test_invalid_dem_is_rejected(self):
        """A three-dimensional array is rejected."""
        dem = np.ones((3, 3, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="two-dimensional"):
            detect_flats(dem=dem, nodata=-9999.0)

    def test_higher_boundary_is_subset_of_flat(self):
        """Higher boundary cells must be a subset of flat cells."""
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

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        assert np.all(flat_mask[higher_boundary])
        assert np.all(flat_mask[lower_boundary])
