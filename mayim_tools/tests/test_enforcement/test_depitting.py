"""Tests for native Stage 5 single-cell de-pitting.

These tests use synthetic DEM arrays with hand-calculated expected
results. They do not use WhiteboxTools, RichDEM or TauDEM.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.enforcement.depitting import (
    depit_single_cell,
)


class TestDepitSingleCell:
    """Tests for depit_single_cell()."""

    def test_pit_is_raised_to_lowest_valid_neighbour(self):
        """The pit is raised to the lowest valid neighbour elevation."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        result, audit = depit_single_cell(
            dem=dem,
            pit=(1, 1),
            nodata=-9999.0,
        )

        assert result[1, 1] == 8.0
        assert result[1, 2] == 8.0
        assert np.array_equal(
            result[np.arange(result.shape[0]) != 1],
            dem[np.arange(dem.shape[0]) != 1],
        )

        assert audit["row"] == 1
        assert audit["column"] == 1
        assert audit["original_elevation"] == 2.0
        assert audit["new_elevation"] == 8.0
        assert audit["elevation_change"] == 6.0
        assert audit["method"] == "single_cell_depitting"

    def test_lowest_neighbour_is_selected(self):
        """The lowest valid neighbouring cell determines the new elevation."""
        dem = np.array(
            [
                [10.0, 7.0, 10.0],
                [6.0, 2.0, 9.0],
                [10.0, 8.0, 10.0],
            ],
            dtype=np.float64,
        )

        result, audit = depit_single_cell(
            dem=dem,
            pit=(1, 1),
            nodata=-9999.0,
        )

        assert result[1, 1] == 6.0
        assert audit["new_elevation"] == 6.0
        assert audit["elevation_change"] == 4.0

    def test_original_array_is_not_modified(self):
        """The input DEM must remain unchanged."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        original = dem.copy()

        result, _ = depit_single_cell(
            dem=dem,
            pit=(1, 1),
            nodata=-9999.0,
        )

        assert np.array_equal(dem, original)
        assert result is not dem

    def test_nodata_neighbours_are_ignored(self):
        """NoData neighbours must not be selected."""
        dem = np.array(
            [
                [-9999.0, -9999.0, -9999.0],
                [-9999.0, 2.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        result, audit = depit_single_cell(
            dem=dem,
            pit=(1, 1),
            nodata=-9999.0,
        )

        assert result[1, 1] == 8.0
        assert audit["new_elevation"] == 8.0

    def test_nodata_values_remain_unchanged(self):
        """NoData values in the input remain unchanged."""
        dem = np.array(
            [
                [-9999.0, 10.0, 10.0],
                [10.0, 2.0, 8.0],
                [10.0, 10.0, -9999.0],
            ],
            dtype=np.float64,
        )

        result, _ = depit_single_cell(
            dem=dem,
            pit=(1, 1),
            nodata=-9999.0,
        )

        assert result[0, 0] == -9999.0
        assert result[2, 2] == -9999.0

    def test_non_pit_cell_is_not_lowered(self):
        """A cell higher than its lowest neighbour is not lowered."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 8.0, 2.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        result, audit = depit_single_cell(
            dem=dem,
            pit=(1, 1),
            nodata=-9999.0,
        )

        assert result[1, 1] == 8.0
        assert audit["elevation_change"] == 0.0
        assert audit["modified"] is False

    def test_invalid_pit_coordinate_is_rejected(self):
        """Pit coordinates outside the DEM are rejected."""
        dem = np.ones((3, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="outside"):
            depit_single_cell(
                dem=dem,
                pit=(5, 5),
                nodata=-9999.0,
            )

    def test_no_valid_neighbours_are_rejected(self):
        """A pit surrounded by NoData is rejected."""
        dem = np.array(
            [
                [-9999.0, -9999.0, -9999.0],
                [-9999.0, 2.0, -9999.0],
                [-9999.0, -9999.0, -9999.0],
            ],
            dtype=np.float64,
        )

        with pytest.raises(ValueError, match="valid neighbours"):
            depit_single_cell(
                dem=dem,
                pit=(1, 1),
                nodata=-9999.0,
            )

    def test_result_shape_matches_input(self):
        """The output array has the same shape as the input."""
        dem = np.ones((8, 11), dtype=np.float64) * 10.0
        dem[4, 5] = 2.0

        result, _ = depit_single_cell(
            dem=dem,
            pit=(4, 5),
            nodata=-9999.0,
        )

        assert result.shape == dem.shape

    def test_elevation_is_never_lowered(self):
        """De-pitting cannot lower any DEM cell."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        result, _ = depit_single_cell(
            dem=dem,
            pit=(1, 1),
            nodata=-9999.0,
        )

        valid = dem != -9999.0
        assert np.all(result[valid] >= dem[valid])
