"""Tests for native Mayim depression feature calculations.

These tests use small synthetic DEMs with hand-calculated expected
results. They do not use WhiteboxTools, RichDEM, NetworkX or any
other third-party hydrological implementation.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.depression.features import (
    calculate_depression_features,
)


class TestCalculateDepressionFeatures:
    """Tests for calculate_depression_features()."""

    def test_single_depression_features(self):
        """Calculate the expected features for one 3x3 depression."""
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

        depression_ids = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        result = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={1: 10.0},
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert list(result) == [1]

        features = result[1]

        # The lowest cell is the centre cell.
        assert features["pit_row"] == 2
        assert features["pit_col"] == 2
        assert features["pit_elevation"] == 2.0

        # Spill elevation is supplied as 10.0.
        assert features["spill_elevation"] == 10.0
        assert features["depth"] == 8.0

        # Nine valid cells in the 3x3 depression.
        assert features["area_cells"] == 9
        assert features["area_map_units"] == 9.0

        # Eight outer cells are perimeter cells.
        assert features["perimeter_cells"] == 8

        # Volume:
        #   Eight cells at elevation 8: 8 * (10 - 8) = 16
        #   One cell at elevation 2:   1 * (10 - 2) = 8
        #   Total volume = 24 cubic map units.
        assert features["volume_estimate"] == 24.0

        assert features["touches_boundary"] is False
        assert features["elongation_index"] == pytest.approx(0.5625)

    def test_cell_size_scales_area_and_volume(self):
        """Cell size scales planimetric area and volume."""
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

        depression_ids = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        result = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={1: 10.0},
            cell_size=5.0,
            nodata=-9999.0,
        )

        features = result[1]

        # Cell area = 5 * 5 = 25.
        assert features["area_cells"] == 9
        assert features["area_map_units"] == 225.0

        # Volume = 24 elevation-map-unit cells * 25 cell-area units.
        assert features["volume_estimate"] == 600.0

    def test_multiple_depressions_are_returned_separately(self):
        """Each positive depression ID receives its own feature record."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 2.0, 10.0, 10.0, 4.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_ids = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        result = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={
                1: 10.0,
                2: 10.0,
            },
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert sorted(result) == [1, 2]
        assert result[1]["pit_elevation"] == 2.0
        assert result[2]["pit_elevation"] == 4.0
        assert result[1]["depth"] == 8.0
        assert result[2]["depth"] == 6.0

    def test_boundary_connected_depression_is_flagged(self):
        """A depression touching the raster edge is flagged."""
        dem = np.array(
            [
                [2.0, 8.0, 10.0, 10.0],
                [8.0, 8.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_ids = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        result = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={1: 10.0},
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert result[1]["touches_boundary"] is True

    def test_nodata_cells_are_excluded(self):
        """NoData cells are excluded from feature calculations."""
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

        depression_ids = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        result = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={1: 10.0},
            cell_size=1.0,
            nodata=-9999.0,
        )

        features = result[1]

        assert features["area_cells"] == 8
        assert features["area_map_units"] == 8.0
        assert features["pit_elevation"] == 8.0
        assert features["depth"] == 2.0
        assert features["volume_estimate"] == 16.0

    def test_empty_depression_ids_return_empty_dictionary(self):
        """No positive depression IDs produce no feature records."""
        dem = np.ones((4, 4), dtype=np.float64) * 10.0
        depression_ids = np.zeros((4, 4), dtype=np.int32)

        result = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={},
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert result == {}

    def test_missing_spill_point_defaults_to_pit_elevation(self):
        """A missing spill point produces zero calculated depth."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_ids = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int32,
        )

        result = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={},
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert result[1]["pit_elevation"] == 2.0
        assert result[1]["spill_elevation"] == 2.0
        assert result[1]["depth"] == 0.0
        assert result[1]["volume_estimate"] == 0.0


class TestCalculateDepressionFeaturesValidation:
    """Tests for invalid inputs."""

    def test_dem_must_be_two_dimensional(self):
        """A three-dimensional DEM is rejected."""
        dem = np.ones((2, 2, 2), dtype=np.float64)
        depression_ids = np.ones((2, 2), dtype=np.int32)

        with pytest.raises(ValueError, match="two-dimensional"):
            calculate_depression_features(
                dem=dem,
                depression_ids=depression_ids,
                spill_points={1: 10.0},
                cell_size=1.0,
                nodata=-9999.0,
            )

    def test_depression_ids_must_be_two_dimensional(self):
        """A three-dimensional ID array is rejected."""
        dem = np.ones((2, 2), dtype=np.float64)
        depression_ids = np.ones((2, 2, 2), dtype=np.int32)

        with pytest.raises(ValueError, match="two-dimensional"):
            calculate_depression_features(
                dem=dem,
                depression_ids=depression_ids,
                spill_points={1: 10.0},
                cell_size=1.0,
                nodata=-9999.0,
            )

    def test_array_shapes_must_match(self):
        """DEM and depression IDs must have identical dimensions."""
        dem = np.ones((3, 3), dtype=np.float64)
        depression_ids = np.ones((2, 2), dtype=np.int32)

        with pytest.raises(ValueError, match="identical shapes"):
            calculate_depression_features(
                dem=dem,
                depression_ids=depression_ids,
                spill_points={1: 10.0},
                cell_size=1.0,
                nodata=-9999.0,
            )

    def test_cell_size_must_be_positive(self):
        """Zero or negative cell sizes are rejected."""
        dem = np.ones((3, 3), dtype=np.float64)
        depression_ids = np.ones((3, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="cell_size must be positive"):
            calculate_depression_features(
                dem=dem,
                depression_ids=depression_ids,
                spill_points={1: 10.0},
                cell_size=0.0,
                nodata=-9999.0,
            )

    def test_spill_elevation_below_pit_is_rejected(self):
        """A spill elevation below the pit elevation is invalid."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 5.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_ids = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int32,
        )

        with pytest.raises(
            ValueError,
            match="spill elevation",
        ):
            calculate_depression_features(
                dem=dem,
                depression_ids=depression_ids,
                spill_points={1: 4.0},
                cell_size=1.0,
                nodata=-9999.0,
            )
