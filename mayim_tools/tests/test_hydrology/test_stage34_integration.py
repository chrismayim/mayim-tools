"""Integration tests for native Stage 3 and Stage 4 processing.

These tests validate the native Mayim pipeline components together:

    detect_depressions
    identify_spill_points
    build_hierarchy
    calculate_depression_features
    classify_depressions

No QGIS runtime is required.
No WhiteboxTools, RichDEM or third-party hydrological implementation
is used.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.depression.classification import (
    ARTIFACT,
    REAL_FEATURE,
    REVIEW_REQUIRED,
    classify_depressions,
)
from mayim_tools.hydrology.depression.detection import (
    detect_depressions,
    identify_spill_points,
)
from mayim_tools.hydrology.depression.features import (
    calculate_depression_features,
)
from mayim_tools.hydrology.depression.hierarchy import (
    build_hierarchy,
)


class TestStage34Integration:
    """End-to-end native Stage 3/4 tests."""

    def test_single_depression_pipeline(self):
        """A simple isolated depression should pass through all stages."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0,  8.0,  8.0,  8.0, 10.0],
                [10.0,  8.0,  2.0,  8.0, 10.0],
                [10.0,  8.0,  8.0,  8.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_ids, pit_cells, count = detect_depressions(
            dem=dem,
            nodata=-9999.0,
        )

        assert count == 1

        spill_points = identify_spill_points(
            dem=dem,
            depression_ids=depression_ids,
            nodata=-9999.0,
        )

        hierarchy = build_hierarchy(
            dem=dem,
            depression_ids=depression_ids,
            pit_cells=pit_cells,
            spill_points=spill_points,
            nodata=-9999.0,
            cell_size=1.0,
        )

        assert hierarchy.total_depressions == 1
        assert hierarchy.root_count == 1
        assert hierarchy.max_depth == 0

        features = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points=spill_points,
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert list(features) == [1]
        assert features[1]["pit_elevation"] == 2.0
        assert features[1]["spill_elevation"] == 8.0
        assert features[1]["depth"] == 6.0

        results = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
        )

        assert list(results) == [1]
        assert results[1].classification in {
            ARTIFACT,
            REAL_FEATURE,
            REVIEW_REQUIRED,
        }

    def test_two_independent_depressions_pipeline(self):
        """Two separate depressions should remain separate."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0,  2.0, 10.0, 10.0,  4.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        depression_ids, pit_cells, count = detect_depressions(
            dem=dem,
            nodata=-9999.0,
        )

        assert count == 2

        spill_points = identify_spill_points(
            dem=dem,
            depression_ids=depression_ids,
            nodata=-9999.0,
        )

        hierarchy = build_hierarchy(
            dem=dem,
            depression_ids=depression_ids,
            pit_cells=pit_cells,
            spill_points=spill_points,
            nodata=-9999.0,
            cell_size=1.0,
        )

        assert hierarchy.total_depressions == 2

        features = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points=spill_points,
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert sorted(features) == [1, 2]

        results = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
        )

        assert sorted(results) == [1, 2]

    def test_boundary_connected_depression_pipeline(self):
        """A boundary-connected depression must be flagged in features."""
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

        pit_cells = np.zeros_like(depression_ids, dtype=bool)
        pit_cells[0, 0] = True

        spill_points = {1: 10.0}

        hierarchy = build_hierarchy(
            dem=dem,
            depression_ids=depression_ids,
            pit_cells=pit_cells,
            spill_points=spill_points,
            nodata=-9999.0,
            cell_size=1.0,
        )

        assert hierarchy.total_depressions == 1

        features = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points=spill_points,
            cell_size=1.0,
            nodata=-9999.0,
        )

        assert features[1]["touches_boundary"] is True

    def test_known_basin_evidence_can_preserve_depression(self):
        """Known-basin evidence should reduce artifact likelihood."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0,  2.0, 10.0],
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

        features = calculate_depression_features(
            dem=dem,
            depression_ids=depression_ids,
            spill_points={1: 10.0},
            cell_size=1.0,
            nodata=-9999.0,
        )

        result_without = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
        )[1]

        result_with = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
            evidence={1: {"known_basin": True}},
        )[1]

        assert result_with.artifact_score < result_without.artifact_score

    @pytest.mark.xfail(
        reason=(
            "Nested depression and merge-event behaviour is still a "
            "prototype. This test records the expected future behaviour "
            "required by the revised methodology."
        ),
        strict=False,
    )
    def test_nested_depression_pipeline_future_expected_behaviour(self):
        """Future target: nested depressions should create hierarchy depth."""
        dem = np.array(
            [
                [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
                [12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0],
                [12.0, 10.0,  6.0,  6.0,  6.0, 10.0, 12.0],
                [12.0, 10.0,  6.0,  2.0,  6.0, 10.0, 12.0],
                [12.0, 10.0,  6.0,  6.0,  6.0, 10.0, 12.0],
                [12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0],
                [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            ],
            dtype=np.float64,
        )

        depression_ids, pit_cells, count = detect_depressions(
            dem=dem,
            nodata=-9999.0,
        )

        spill_points = identify_spill_points(
            dem=dem,
            depression_ids=depression_ids,
            nodata=-9999.0,
        )

        hierarchy = build_hierarchy(
            dem=dem,
            depression_ids=depression_ids,
            pit_cells=pit_cells,
            spill_points=spill_points,
            nodata=-9999.0,
            cell_size=1.0,
        )

        # Future expectation: an inner depression nested inside an
        # outer depression should create hierarchy depth >= 1.
        assert hierarchy.max_depth >= 1
