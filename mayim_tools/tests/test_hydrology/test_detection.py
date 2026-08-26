"""Tests for native Mayim depression detection.

These tests use synthetic DEMs with known expected behaviour.
They do not use WhiteboxTools, RichDEM or another hydrological
implementation as a runtime dependency.
"""

import numpy as np

from mayim_tools.hydrology.depression.detection import (
    detect_depressions,
    identify_spill_points,
)


class TestDetectDepressions:
    """Tests for detect_depressions()."""

    def test_depression_free_dem_has_no_depressions(self):
        """A monotonically descending DEM should have no depressions."""
        dem = np.array(
            [
                [10.0, 9.0, 8.0, 7.0, 6.0],
                [9.0, 8.0, 7.0, 6.0, 5.0],
                [8.0, 7.0, 6.0, 5.0, 4.0],
                [7.0, 6.0, 5.0, 4.0, 3.0],
                [6.0, 5.0, 4.0, 3.0, 2.0],
            ],
            dtype=np.float64,
        )

        depression_ids, pit_cells, count = detect_depressions(
            dem,
            nodata=-9999.0,
        )

        assert count == 0
        assert np.all(depression_ids == 0)
        assert not np.any(pit_cells)

    def test_single_isolated_pit_is_detected(self):
        """A single interior pit should be detected."""
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

        depression_ids, pit_cells, count = detect_depressions(
            dem,
            nodata=-9999.0,
        )

        assert count == 1

        pit_locations = np.argwhere(pit_cells)
        assert pit_locations.tolist() == [[2, 2]]

        assert depression_ids[2, 2] > 0

    def test_nodata_cells_are_not_depressions(self):
        """NoData cells must never receive a depression ID."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 8.0, -9999.0, 8.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_ids, pit_cells, _ = detect_depressions(
            dem,
            nodata=-9999.0,
        )

        assert depression_ids[2, 2] == -1
        assert not pit_cells[2, 2]

    def test_output_shapes_match_input(self):
        """Detection outputs must have the same shape as the input."""
        dem = np.ones((7, 9), dtype=np.float64)

        depression_ids, pit_cells, count = detect_depressions(
            dem,
            nodata=-9999.0,
        )

        assert depression_ids.shape == dem.shape
        assert pit_cells.shape == dem.shape
        assert np.issubdtype(depression_ids.dtype, np.integer)
        assert pit_cells.dtype == np.bool_
        assert count >= 0


class TestIdentifySpillPoints:
    """Tests for identify_spill_points()."""

    def test_single_pit_spill_elevation(self):
        """The lowest escape elevation should be identified."""
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

        depression_ids, _, count = detect_depressions(
            dem,
            nodata=-9999.0,
        )

        assert count == 1

        spill_points = identify_spill_points(
            dem,
            depression_ids,
            nodata=-9999.0,
        )

        depression_id = int(depression_ids[2, 2])
        assert depression_id in spill_points
        assert spill_points[depression_id] == 8.0

    def test_depression_free_dem_has_no_spill_points(self):
        """A depression-free DEM should return an empty dictionary."""
        dem = np.arange(25, dtype=np.float64).reshape((5, 5))

        depression_ids, _, count = detect_depressions(
            dem,
            nodata=-9999.0,
        )

        spill_points = identify_spill_points(
            dem,
            depression_ids,
            nodata=-9999.0,
        )

        assert count == 0
        assert spill_points == {}
