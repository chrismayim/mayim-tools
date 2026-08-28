"""Tests for native Mayim depression classification.

These tests validate the inspectable Stage 4 weighted artifact-likelihood
classifier. They do not use WhiteboxTools, RichDEM, TauDEM or any other
third-party hydrological implementation.
"""

import pytest

from mayim_tools.hydrology.depression.classification import (
    ARTIFACT,
    REAL_FEATURE,
    REVIEW_REQUIRED,
    ClassificationResult,
    classify_depression,
    classify_depressions,
)


def make_features(
    depression_id: int = 1,
    depth: float = 2.0,
    area_cells: float = 9.0,
    elongation_index: float = 0.5,
) -> dict:
    """Create a minimal valid depression feature dictionary."""
    return {
        "depression_id": depression_id,
        "depth": depth,
        "area_cells": area_cells,
        "elongation_index": elongation_index,
    }


class TestClassifyDepression:
    """Tests for classify_depression()."""

    def test_score_is_bounded_between_zero_and_one(self):
        """The final artifact score is always clamped to [0, 1]."""
        result = classify_depression(
            features=make_features(
                depth=0.1,
                area_cells=1.0,
                elongation_index=0.0,
            ),
            vertical_accuracy=1.0,
            evidence={
                "infrastructure": True,
                "dsm_bias": True,
            },
        )

        assert 0.0 <= result.artifact_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_shallow_depression_is_more_artifact_like_than_deep_one(self):
        """A shallow depression relative to RMSE has higher artifact score."""
        shallow = classify_depression(
            features=make_features(depth=0.5),
            vertical_accuracy=1.0,
        )

        deep = classify_depression(
            features=make_features(depth=10.0),
            vertical_accuracy=1.0,
        )

        assert shallow.artifact_score > deep.artifact_score
        assert shallow.depth_ratio < deep.depth_ratio
        assert shallow.depth_score > deep.depth_score

    def test_small_depression_is_more_artifact_like_than_large_one(self):
        """Small depressions should receive higher area-based artifact evidence."""
        small = classify_depression(
            features=make_features(area_cells=1.0),
            vertical_accuracy=1.0,
        )

        large = classify_depression(
            features=make_features(area_cells=100.0),
            vertical_accuracy=1.0,
        )

        assert small.artifact_score > large.artifact_score
        assert small.area_score > large.area_score

    def test_irregular_shape_is_more_artifact_like_than_compact_shape(self):
        """Lower compactness should increase the shape-artifact score."""
        irregular = classify_depression(
            features=make_features(elongation_index=0.1),
            vertical_accuracy=1.0,
        )

        compact = classify_depression(
            features=make_features(elongation_index=0.9),
            vertical_accuracy=1.0,
        )

        assert irregular.shape_score > compact.shape_score
        assert irregular.artifact_score > compact.artifact_score

    def test_infrastructure_evidence_increases_score(self):
        """Infrastructure evidence should raise the artifact likelihood."""
        without_evidence = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence={},
        )

        with_infrastructure = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence={"infrastructure": True},
        )

        assert with_infrastructure.infrastructure_evidence is True
        assert with_infrastructure.artifact_score > without_evidence.artifact_score

    def test_known_basin_evidence_decreases_score(self):
        """Known-basin evidence should reduce the artifact likelihood."""
        without_evidence = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence={},
        )

        with_known_basin = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence={"known_basin": True},
        )

        assert with_known_basin.known_basin_evidence is True
        assert with_known_basin.artifact_score < without_evidence.artifact_score

    def test_dsm_bias_evidence_increases_score(self):
        """DSM-bias evidence should raise the artifact likelihood."""
        without_evidence = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence={},
        )

        with_dsm_bias = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence={"dsm_bias": True},
        )

        assert with_dsm_bias.dsm_bias_evidence is True
        assert with_dsm_bias.artifact_score > without_evidence.artifact_score

    def test_high_score_returns_artifact(self):
        """Strong artifact evidence should return ARTIFACT."""
        result = classify_depression(
            features=make_features(
                depth=0.1,
                area_cells=1.0,
                elongation_index=0.0,
            ),
            vertical_accuracy=1.0,
            evidence={
                "infrastructure": True,
                "dsm_bias": True,
            },
        )

        assert result.classification == ARTIFACT
        assert result.review_required is False
        assert result.artifact_score >= 0.75

    def test_low_score_returns_real_feature(self):
        """Low artifact evidence should return REAL_FEATURE."""
        result = classify_depression(
            features=make_features(
                depth=10.0,
                area_cells=100.0,
                elongation_index=1.0,
            ),
            vertical_accuracy=0.5,
            evidence={
                "known_basin": True,
            },
        )

        assert result.classification == REAL_FEATURE
        assert result.review_required is False
        assert result.artifact_score <= 0.45

    def test_intermediate_score_returns_review_required(self):
        """Borderline scores should be exported as REVIEW_REQUIRED."""
        result = classify_depression(
            features=make_features(
                depth=2.0,
                area_cells=9.0,
                elongation_index=0.5,
            ),
            vertical_accuracy=1.0,
            evidence={},
        )

        assert result.classification == REVIEW_REQUIRED
        assert result.review_required is True
        assert 0.45 < result.artifact_score < 0.75

    def test_missing_evidence_does_not_crash(self):
        """Classification must still work when no evidence is supplied."""
        result = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence=None,
        )

        assert isinstance(result, ClassificationResult)
        assert result.infrastructure_evidence is False
        assert result.known_basin_evidence is False
        assert result.dsm_bias_evidence is False

    def test_result_is_deterministic(self):
        """The same inputs must always produce the same result."""
        features = make_features(
            depth=1.0,
            area_cells=5.0,
            elongation_index=0.25,
        )
        evidence = {
            "infrastructure": True,
            "dsm_bias": False,
            "known_basin": False,
        }

        result_a = classify_depression(
            features=features,
            vertical_accuracy=1.0,
            evidence=evidence,
        )
        result_b = classify_depression(
            features=features,
            vertical_accuracy=1.0,
            evidence=evidence,
        )

        assert result_a == result_b
        assert result_a.to_dict() == result_b.to_dict()

    def test_result_serialisation_contains_score_contributions(self):
        """The result dictionary must contain inspectable contributions."""
        result = classify_depression(
            features=make_features(),
            vertical_accuracy=1.0,
            evidence={"infrastructure": True},
        )

        data = result.to_dict()

        assert data["depression_id"] == 1
        assert "artifact_score" in data
        assert "classification" in data
        assert "confidence" in data
        assert "depth_ratio" in data
        assert "depth_score" in data
        assert "area_score" in data
        assert "shape_score" in data
        assert "score_contributions" in data
        assert "explanation" in data

        contributions = data["score_contributions"]
        assert "depth" in contributions
        assert "area" in contributions
        assert "shape" in contributions

    def test_invalid_vertical_accuracy_is_rejected(self):
        """Vertical accuracy must be positive and finite."""
        with pytest.raises(
            ValueError,
            match="vertical_accuracy",
        ):
            classify_depression(
                features=make_features(),
                vertical_accuracy=0.0,
            )

    def test_missing_required_feature_field_is_rejected(self):
        """Required feature fields must be present."""
        incomplete = {
            "depression_id": 1,
            "depth": 2.0,
            "area_cells": 9.0,
            # elongation_index intentionally missing
        }

        with pytest.raises(
            ValueError,
            match="Missing required depression feature",
        ):
            classify_depression(
                features=incomplete,
                vertical_accuracy=1.0,
            )


class TestClassifyDepressions:
    """Tests for classify_depressions()."""

    def test_multiple_depressions_are_returned_in_sorted_order(self):
        """Collection classification must be deterministic by ID."""
        features = {
            5: make_features(depression_id=5, depth=0.1, area_cells=1.0),
            2: make_features(depression_id=2, depth=10.0, area_cells=100.0),
            9: make_features(depression_id=9, depth=2.0, area_cells=9.0),
        }

        results = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
        )

        assert list(results.keys()) == [2, 5, 9]

    def test_per_depression_evidence_is_used(self):
        """Per-depression evidence dictionaries are applied correctly."""
        features = {
            1: make_features(depression_id=1),
            2: make_features(depression_id=2),
        }

        results = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
            evidence={
                1: {"infrastructure": True},
                2: {"known_basin": True},
            },
        )

        assert results[1].infrastructure_evidence is True
        assert results[1].known_basin_evidence is False

        assert results[2].infrastructure_evidence is False
        assert results[2].known_basin_evidence is True

        assert results[1].artifact_score > results[2].artifact_score

    def test_string_keys_in_evidence_dictionary_are_supported(self):
        """String-form depression IDs are accepted in the evidence map."""
        features = {
            1: make_features(depression_id=1),
        }

        results = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
            evidence={
                "1": {"infrastructure": True},
            },
        )

        assert results[1].infrastructure_evidence is True

    def test_common_evidence_dictionary_is_used_for_all_depressions(self):
        """A single evidence dictionary can be shared across all IDs."""
        features = {
            1: make_features(depression_id=1),
            2: make_features(depression_id=2),
        }

        results = classify_depressions(
            depression_features=features,
            vertical_accuracy=1.0,
            evidence={"dsm_bias": True},
        )

        assert results[1].dsm_bias_evidence is True
        assert results[2].dsm_bias_evidence is True

    def test_invalid_feature_collection_is_rejected(self):
        """The feature collection must be a dictionary."""
        with pytest.raises(
            ValueError,
            match="dictionary keyed by ID",
        ):
            classify_depressions(
                depression_features=[],
                vertical_accuracy=1.0,
            )

