"""Tests for native Stage 5 constrained least-cost breaching.

These tests validate path search and constraints only. They do not use
WhiteboxTools, RichDEM, TauDEM or any other hydrological implementation.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.enforcement.breaching import (
    least_cost_breach,
)


class TestLeastCostBreach:
    """Tests for least_cost_breach()."""

    def test_finds_a_path_from_pit_to_spill_target(self):
        """A valid path should be found from the pit to the outlet."""
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

        result = least_cost_breach(
            dem=dem,
            pit=(2, 2),
            spill_elevation=8.0,
            max_length=10,
            max_depth=10.0,
            nodata=-9999.0,
        )

        path, audit = result

        assert path is not None
        assert len(path) >= 2
        assert path[0] == (2, 2)
        assert audit["method"] == "constrained_least_cost_breach"
        assert audit["path_length"] == len(path) - 1

    def test_path_cells_are_contiguous(self):
        """Successive path cells must be neighbours."""
        dem = np.full((7, 7), 10.0, dtype=np.float64)
        dem[3, 3] = 2.0

        path, _ = least_cost_breach(
            dem=dem,
            pit=(3, 3),
            spill_elevation=10.0,
            max_length=20,
            max_depth=20.0,
            nodata=-9999.0,
        )

        assert path is not None

        import itertools

        for current, following in itertools.pairwise(path):
            row_distance = abs(current[0] - following[0])
            col_distance = abs(current[1] - following[1])

            assert row_distance <= 1
            assert col_distance <= 1
            assert row_distance + col_distance > 0

    def test_path_does_not_exceed_maximum_length(self):
        """The search must reject paths beyond max_length."""
        dem = np.full((9, 9), 10.0, dtype=np.float64)
        dem[4, 4] = 2.0

        path, audit = least_cost_breach(
            dem=dem,
            pit=(4, 4),
            spill_elevation=0.0,
            max_length=1,
            max_depth=20.0,
            nodata=-9999.0,
        )

        assert path is None
        assert audit["success"] is False
        assert audit["failure_reason"] == "maximum_constraints_exceeded"

    def test_path_does_not_exceed_maximum_depth(self):
        """The search must reject paths exceeding max_depth."""
        dem = np.array(
            [
                [100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 95.0, 95.0, 95.0, 100.0],
                [100.0, 95.0, 2.0, 95.0, 100.0],
                [100.0, 95.0, 95.0, 95.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        )

        path, audit = least_cost_breach(
            dem=dem,
            pit=(2, 2),
            spill_elevation=90.0,
            max_length=20,
            max_depth=1.0,
            nodata=-9999.0,
        )

        assert path is None
        assert audit["success"] is False
        assert audit["failure_reason"] == "maximum_constraints_exceeded"

    def test_nodata_cells_are_not_traversed(self):
        """NoData cells must be treated as impassable."""
        dem = np.full((5, 5), 10.0, dtype=np.float64)
        dem[2, 2] = 2.0
        dem[2, 3] = -9999.0
        dem[1, 2] = -9999.0
        dem[3, 2] = -9999.0

        path, audit = least_cost_breach(
            dem=dem,
            pit=(2, 2),
            spill_elevation=10.0,
            max_length=20,
            max_depth=20.0,
            nodata=-9999.0,
        )

        assert path is not None
        assert all(dem[row, col] != -9999.0 for row, col in path)
        assert audit["nodata_cells_encountered"] >= 0

    def test_input_dem_is_not_modified(self):
        """The breach search must not modify the input array."""
        dem = np.full((5, 5), 10.0, dtype=np.float64)
        dem[2, 2] = 2.0
        original = dem.copy()

        least_cost_breach(
            dem=dem,
            pit=(2, 2),
            spill_elevation=10.0,
            max_length=20,
            max_depth=20.0,
            nodata=-9999.0,
        )

        assert np.array_equal(dem, original)

    def test_invalid_pit_coordinate_is_rejected(self):
        """Pit coordinates outside the DEM must be rejected."""
        dem = np.ones((5, 5), dtype=np.float64)

        with pytest.raises(ValueError, match="outside"):
            least_cost_breach(
                dem=dem,
                pit=(10, 10),
                spill_elevation=1.0,
                max_length=10,
                max_depth=10.0,
                nodata=-9999.0,
            )

    def test_invalid_constraints_are_rejected(self):
        """Length and depth constraints must be positive."""
        dem = np.ones((5, 5), dtype=np.float64)

        with pytest.raises(ValueError, match="max_length"):
            least_cost_breach(
                dem=dem,
                pit=(2, 2),
                spill_elevation=1.0,
                max_length=0,
                max_depth=10.0,
                nodata=-9999.0,
            )

        with pytest.raises(ValueError, match="max_depth"):
            least_cost_breach(
                dem=dem,
                pit=(2, 2),
                spill_elevation=1.0,
                max_length=10,
                max_depth=0.0,
                nodata=-9999.0,
            )

    def test_invalid_connectivity_is_rejected(self):
        """Only four- and eight-connected searches are supported."""
        dem = np.ones((5, 5), dtype=np.float64)

        with pytest.raises(ValueError, match="connectivity"):
            least_cost_breach(
                dem=dem,
                pit=(2, 2),
                spill_elevation=1.0,
                max_length=10,
                max_depth=10.0,
                nodata=-9999.0,
                connectivity=6,
            )

    def test_audit_contains_cost_and_depth_information(self):
        """A successful result must expose auditable path statistics."""
        dem = np.full((5, 5), 10.0, dtype=np.float64)
        dem[2, 2] = 2.0

        path, audit = least_cost_breach(
            dem=dem,
            pit=(2, 2),
            spill_elevation=10.0,
            max_length=20,
            max_depth=20.0,
            nodata=-9999.0,
        )

        assert path is not None
        assert audit["success"] is True
        assert audit["path_length"] > 0
        assert audit["total_cost"] >= 0.0
        assert audit["maximum_excavation_depth"] >= 0.0
        assert "path" in audit
