"""
Mayim Tools - Depression Classification
========================================

Implements Stage 4 of the Mayim Tools DEM hydrological-conditioning
methodology.

This module assigns each depression an inspectable artifact-likelihood
score using multiple geomorphometric and contextual criteria. It does
not modify DEM elevations.

Possible outcomes
-----------------
ARTIFACT
    The depression has a high artifact likelihood and may be passed
    to Stage 5 selective flow enforcement.

REAL_FEATURE
    The depression has a low artifact likelihood and should normally
    be preserved.

REVIEW_REQUIRED
    The available evidence is insufficient for automatic treatment.
    The depression should be preserved until an analyst reviews it.

Methodology basis
-----------------
This implementation follows the revised Mayim research methodology:

- Stage 4 uses a multi-criteria, resolution-adaptive artifact-likelihood
  score rather than a single depth or area threshold.
- The score combines:
    * depth relative to vertical accuracy,
    * area relative to raster-cell scale,
    * shape regularity,
    * infrastructure evidence,
    * known-basin evidence,
    * DSM-bias evidence.
- Thresholds and weights are configuration, not hidden code.
- Borderline cases are exported as REVIEW_REQUIRED.

IP status
---------
Original Mayim implementation.

This module uses only Python standard-library components and NumPy.
It does not import or call:

- WhiteboxTools
- RichDEM
- TauDEM
- NetworkX
- any third-party hydrological implementation

The implementation must remain based on the published methodology and
the cited academic literature rather than third-party source code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite

import numpy as np

ARTIFACT = "ARTIFACT"
REAL_FEATURE = "REAL_FEATURE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"

VALID_CLASSIFICATIONS = {
    ARTIFACT,
    REAL_FEATURE,
    REVIEW_REQUIRED,
}


@dataclass(frozen=True)
class ClassificationConfig:
    """
    Configuration for the Stage 4 artifact-likelihood classifier.

    The default threshold and margin follow the illustrative structure
    in the revised Mayim research paper:

        threshold = 0.60
        margin = 0.15

    This creates:

        score >= 0.75 -> ARTIFACT
        score <= 0.45 -> REAL_FEATURE
        otherwise     -> REVIEW_REQUIRED

    Attributes
    ----------
    threshold:
        Central classification threshold in the range 0.0 to 1.0.
    margin:
        Width of the review margin around the threshold.
    area_reference_cells:
        Reference depression size used by the area score.
    depth_transition_ratio:
        Depth-to-accuracy ratio at which the depth score is
        approximately 0.5.
    weight_depth:
        Weight for depth relative to vertical accuracy.
    weight_area:
        Weight for area relative to cell scale.
    weight_shape:
        Weight for shape-related artifact evidence.
    weight_infrastructure:
        Weight for infrastructure evidence.
    weight_known_basin:
        Magnitude of the known-basin evidence penalty.
    weight_dsm_bias:
        Weight for DSM-bias evidence.
    """

    threshold: float = 0.60
    margin: float = 0.15
    area_reference_cells: float = 9.0
    depth_transition_ratio: float = 2.0
    weight_depth: float = 0.30
    weight_area: float = 0.15
    weight_shape: float = 0.20
    weight_infrastructure: float = 0.25
    weight_known_basin: float = 0.25
    weight_dsm_bias: float = 0.10

    def __post_init__(self) -> None:
        """Validate classifier configuration."""
        numeric_values = {
            "threshold": self.threshold,
            "margin": self.margin,
            "area_reference_cells": self.area_reference_cells,
            "depth_transition_ratio": self.depth_transition_ratio,
            "weight_depth": self.weight_depth,
            "weight_area": self.weight_area,
            "weight_shape": self.weight_shape,
            "weight_infrastructure": self.weight_infrastructure,
            "weight_known_basin": self.weight_known_basin,
            "weight_dsm_bias": self.weight_dsm_bias,
        }

        for name, value in numeric_values.items():
            if not isfinite(float(value)):
                raise ValueError(
                    f"Classification configuration '{name}' must be finite."
                )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")

        if not 0.0 <= self.margin <= 1.0:
            raise ValueError("margin must be between 0.0 and 1.0.")

        if self.threshold - self.margin < 0.0:
            raise ValueError("threshold minus margin cannot be less than 0.0.")

        if self.threshold + self.margin > 1.0:
            raise ValueError("threshold plus margin cannot exceed 1.0.")

        if self.area_reference_cells <= 0:
            raise ValueError("area_reference_cells must be greater than zero.")

        if self.depth_transition_ratio <= 0:
            raise ValueError("depth_transition_ratio must be greater than zero.")

        for name, value in numeric_values.items():
            if name.startswith("weight_") and value < 0.0:
                raise ValueError(f"Classification weight '{name}' cannot be negative.")

        if self.total_weight <= 0.0:
            raise ValueError("At least one classification weight must be positive.")

    @property
    def total_weight(self) -> float:
        """Return the sum of all configured criterion weights."""
        return float(
            self.weight_depth
            + self.weight_area
            + self.weight_shape
            + self.weight_infrastructure
            + self.weight_known_basin
            + self.weight_dsm_bias
        )

    def to_dict(self) -> dict[str, float]:
        """Return the configuration as a serialisable dictionary."""
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ClassificationResult:
    """
    Inspectable classification result for one depression.

    Attributes
    ----------
    depression_id:
        Unique depression identifier.
    artifact_score:
        Final score in the range 0.0 to 1.0.
    classification:
        One of ARTIFACT, REAL_FEATURE or REVIEW_REQUIRED.
    confidence:
        Distance from the central threshold, expressed as a score
        between 0.0 and 1.0.
    review_required:
        True when analyst review is required.
    depth_ratio:
        Depression depth divided by vertical accuracy.
    depth_score:
        Artifact evidence from the depth-to-accuracy relationship.
    area_score:
        Artifact evidence from depression area.
    shape_score:
        Artifact evidence from shape. Higher means more elongated
        or less compact.
    infrastructure_evidence:
        Whether infrastructure evidence was supplied and positive.
    known_basin_evidence:
        Whether known-basin evidence was supplied and positive.
    dsm_bias_evidence:
        Whether DSM-bias evidence was supplied and positive.
    score_contributions:
        Weighted contribution of each criterion.
    explanation:
        Human-readable explanation of the result.
    """

    depression_id: int
    artifact_score: float
    classification: str
    confidence: float
    review_required: bool
    depth_ratio: float
    depth_score: float
    area_score: float
    shape_score: float
    infrastructure_evidence: bool
    known_basin_evidence: bool
    dsm_bias_evidence: bool
    score_contributions: dict[str, float]
    explanation: str

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dictionary.

        :returns: Dictionary representation of the result.
        """
        return asdict(self)


def classify_depression(
    features: dict,
    vertical_accuracy: float,
    evidence: dict | None = None,
    thresholds: dict | None = None,
) -> ClassificationResult:
    """
    Classify one depression using an inspectable weighted score.

    Parameters
    ----------
    features:
        Feature dictionary produced by
        ``calculate_depression_features()``.

        Required keys:

            depression_id
            depth
            area_cells
            elongation_index

    vertical_accuracy:
        DEM vertical accuracy in metres, expressed as RMSE or a
        conservative source-based estimate.

    evidence:
        Optional dictionary containing positive evidence flags:

            infrastructure
            known_basin
            dsm_bias

        The following aliases are also accepted:

            infrastructure_evidence
            known_basin_evidence
            dsm_bias_evidence

    thresholds:
        Optional classifier configuration overrides. Accepted keys
        correspond to ``ClassificationConfig`` fields.

    Returns
    -------
    ClassificationResult
        Inspectable classification result.

    Raises
    ------
    ValueError
        If required features or parameters are invalid.

    Notes
    -----
    The score is an artifact-likelihood score:

        0.0 = low artifact likelihood
        1.0 = high artifact likelihood

    The depth criterion gives higher artifact evidence to depressions
    whose depth is small relative to DEM vertical accuracy.

    The area criterion gives higher artifact evidence to small
    depressions.

    The shape criterion uses:

        shape_score = 1 - elongation_index

    because the feature module's elongation_index is actually a
    compactness proxy. A narrow or irregular depression therefore
    receives a higher shape-artifact score.
    """
    config = _build_config(thresholds)
    _validate_features(features)
    _validate_vertical_accuracy(vertical_accuracy)

    evidence_values = evidence or {}

    depression_id = int(features["depression_id"])
    depth = float(features["depth"])
    area_cells = float(features["area_cells"])
    elongation_index = float(features["elongation_index"])

    if depth < 0.0:
        raise ValueError(f"Depression {depression_id} depth cannot be negative.")

    if area_cells <= 0.0:
        raise ValueError(f"Depression {depression_id} area_cells must be positive.")

    if not 0.0 <= elongation_index <= 1.0:
        raise ValueError(
            f"Depression {depression_id} elongation_index must be "
            "between 0.0 and 1.0."
        )

    depth_ratio = depth / float(vertical_accuracy)

    depth_score = _depth_artifact_score(
        depth_ratio=depth_ratio,
        transition_ratio=config.depth_transition_ratio,
    )

    area_score = _area_artifact_score(
        area_cells=area_cells,
        reference_cells=config.area_reference_cells,
    )

    shape_score = float(np.clip(1.0 - elongation_index, 0.0, 1.0))

    infrastructure = _evidence_flag(
        evidence_values,
        "infrastructure",
        "infrastructure_evidence",
    )

    known_basin = _evidence_flag(
        evidence_values,
        "known_basin",
        "known_basin_evidence",
    )

    dsm_bias = _evidence_flag(
        evidence_values,
        "dsm_bias",
        "dsm_bias_evidence",
    )

    contributions = {
        "depth": config.weight_depth * depth_score,
        "area": config.weight_area * area_score,
        "shape": config.weight_shape * shape_score,
    }

    supplied_evidence_weight = 0.0

    if infrastructure:
        contributions["infrastructure"] = config.weight_infrastructure
        supplied_evidence_weight += config.weight_infrastructure

    if known_basin:
        contributions["known_basin"] = -config.weight_known_basin
        supplied_evidence_weight += config.weight_known_basin

    if dsm_bias:
        contributions["dsm_bias"] = config.weight_dsm_bias
        supplied_evidence_weight += config.weight_dsm_bias

    base_weight = config.weight_depth + config.weight_area + config.weight_shape

    denominator = base_weight + supplied_evidence_weight
    raw_score = sum(contributions.values()) / denominator
    artifact_score = float(np.clip(raw_score, 0.0, 1.0))

    classification = _classification_from_score(
        score=artifact_score,
        threshold=config.threshold,
        margin=config.margin,
    )

    review_required = classification == REVIEW_REQUIRED
    confidence = _classification_confidence(
        score=artifact_score,
        threshold=config.threshold,
        margin=config.margin,
    )

    explanation = _build_explanation(
        classification=classification,
        artifact_score=artifact_score,
        depth_ratio=depth_ratio,
        area_score=area_score,
        shape_score=shape_score,
        infrastructure=infrastructure,
        known_basin=known_basin,
        dsm_bias=dsm_bias,
    )

    return ClassificationResult(
        depression_id=depression_id,
        artifact_score=artifact_score,
        classification=classification,
        confidence=confidence,
        review_required=review_required,
        depth_ratio=float(depth_ratio),
        depth_score=float(depth_score),
        area_score=float(area_score),
        shape_score=float(shape_score),
        infrastructure_evidence=infrastructure,
        known_basin_evidence=known_basin,
        dsm_bias_evidence=dsm_bias,
        score_contributions={key: float(value) for key, value in contributions.items()},
        explanation=explanation,
    )


def classify_depressions(
    depression_features: dict[int, dict],
    vertical_accuracy: float,
    evidence: dict[int, dict] | dict | None = None,
    thresholds: dict | None = None,
) -> dict[int, ClassificationResult]:
    """
    Classify multiple depressions in deterministic ID order.

    Parameters
    ----------
    depression_features:
        Mapping from depression ID to feature dictionaries.

    vertical_accuracy:
        DEM vertical accuracy in metres.

    evidence:
        Optional evidence dictionary.

        Two forms are supported:

            Common evidence for all depressions:
                {
                    "infrastructure": True,
                    "known_basin": False,
                }

            Per-depression evidence:
                {
                    1: {"infrastructure": True},
                    2: {"known_basin": True},
                }

    thresholds:
        Optional classifier configuration overrides.

    Returns
    -------
    dict[int, ClassificationResult]
        Results sorted by integer depression ID.
    """
    if not isinstance(depression_features, dict):
        raise ValueError("depression_features must be a dictionary keyed by ID.")

    evidence_values = evidence if isinstance(evidence, dict) else {}

    results: dict[int, ClassificationResult] = {}

    for depression_id in sorted(
        depression_features,
        key=lambda value: int(value),
    ):
        features = dict(depression_features[depression_id])
        features.setdefault("depression_id", int(depression_id))

        if isinstance(evidence, dict) and (
            depression_id in evidence or str(depression_id) in evidence
        ):
            per_depression_evidence = evidence.get(
                depression_id,
                evidence.get(str(depression_id), {}),
            )
        else:
            per_depression_evidence = evidence_values

        results[int(depression_id)] = classify_depression(
            features=features,
            vertical_accuracy=vertical_accuracy,
            evidence=per_depression_evidence,
            thresholds=thresholds,
        )

    return results


def _build_config(
    thresholds: dict | None,
) -> ClassificationConfig:
    """
    Build a ClassificationConfig from optional overrides.

    Unknown configuration keys are rejected so that spelling errors do
    not silently change classifier behaviour.

    :param thresholds: Optional dictionary of configuration overrides.
    :returns: Validated ClassificationConfig.
    """
    if thresholds is None:
        return ClassificationConfig()

    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dictionary or None.")

    valid_keys = set(ClassificationConfig.__dataclass_fields__)
    unknown_keys = set(thresholds) - valid_keys

    if unknown_keys:
        raise ValueError(
            "Unknown classification configuration key(s): " f"{sorted(unknown_keys)}"
        )

    return ClassificationConfig(**thresholds)


def _validate_features(features: dict) -> None:
    """
    Validate the minimum feature fields required by the classifier.

    :param features: Depression feature dictionary.
    :raises ValueError: If required values are absent or invalid.
    """
    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary.")

    required_keys = {
        "depression_id",
        "depth",
        "area_cells",
        "elongation_index",
    }

    missing_keys = required_keys - set(features)

    if missing_keys:
        raise ValueError(
            "Missing required depression feature(s): " f"{sorted(missing_keys)}"
        )

    for key in (
        "depth",
        "area_cells",
        "elongation_index",
    ):
        try:
            value = float(features[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Depression feature '{key}' must be numeric.") from error

        if not isfinite(value):
            raise ValueError(f"Depression feature '{key}' must be finite.")


def _validate_vertical_accuracy(
    vertical_accuracy: float,
) -> None:
    """
    Validate vertical accuracy.

    :param vertical_accuracy: DEM vertical accuracy in metres.
    :raises ValueError: If the value is not positive and finite.
    """
    try:
        value = float(vertical_accuracy)
    except (TypeError, ValueError) as error:
        raise ValueError("vertical_accuracy must be numeric.") from error

    if not isfinite(value) or value <= 0.0:
        raise ValueError("vertical_accuracy must be finite and greater than zero.")


def _depth_artifact_score(
    depth_ratio: float,
    transition_ratio: float,
) -> float:
    """
    Convert depth-to-accuracy ratio into artifact evidence.

    A shallow depression relative to the vertical accuracy receives a
    higher score. A deep depression receives a lower score.

    The logistic transition is centred on ``transition_ratio``:

        score = 1 / (1 + exp(depth_ratio - transition_ratio))

    :param depth_ratio: Depression depth divided by vertical accuracy.
    :param transition_ratio: Logistic transition point.
    :returns: Score from 0.0 to 1.0.
    """
    exponent = float(
        np.clip(
            depth_ratio - transition_ratio,
            -60.0,
            60.0,
        )
    )

    return float(1.0 / (1.0 + exp(exponent)))


def _area_artifact_score(
    area_cells: float,
    reference_cells: float,
) -> float:
    """
    Convert depression area into artifact evidence.

    Smaller depressions receive higher artifact evidence. The score is
    0.5 when area_cells equals reference_cells:

        score = reference_cells / (reference_cells + area_cells)

    :param area_cells: Depression area in cells.
    :param reference_cells: Reference area in cells.
    :returns: Score from 0.0 to 1.0.
    """
    score = reference_cells / (reference_cells + max(area_cells, 0.0))

    return float(np.clip(score, 0.0, 1.0))


def _evidence_flag(
    evidence: dict,
    primary_key: str,
    alias_key: str,
) -> bool:
    """
    Read a Boolean evidence flag using a primary key or alias.

    Missing evidence is treated as False. This is intentional: absent
    evidence must not be treated as positive evidence.

    :param evidence: Evidence dictionary.
    :param primary_key: Preferred evidence key.
    :param alias_key: Accepted alternate key.
    :returns: Boolean evidence value.
    """
    if not isinstance(evidence, dict):
        return False

    if primary_key in evidence:
        return bool(evidence[primary_key])

    if alias_key in evidence:
        return bool(evidence[alias_key])

    return False


def _classification_from_score(
    score: float,
    threshold: float,
    margin: float,
) -> str:
    """
    Convert an artifact score into a classification.

    Scores above the upper review boundary are classified as artifacts.
    Scores below the lower review boundary are classified as real
    features. Scores within the review band require analyst review.

    :param score: Artifact-likelihood score.
    :param threshold: Central classification threshold.
    :param margin: Review margin around the threshold.
    :returns: Classification label.
    """
    upper_boundary = threshold + margin
    lower_boundary = threshold - margin

    if score >= upper_boundary:
        return ARTIFACT

    if score <= lower_boundary:
        return REAL_FEATURE

    return REVIEW_REQUIRED


def _classification_confidence(
    score: float,
    threshold: float,
    margin: float,
) -> float:
    """
    Calculate a simple confidence measure.

    Confidence is zero at the central threshold and reaches one at or
    beyond either edge of the review band.

    This is a decision-separation indicator, not a statistically
    calibrated probability.

    :param score: Artifact-likelihood score.
    :param threshold: Central classification threshold.
    :param margin: Review margin.
    :returns: Confidence from 0.0 to 1.0.
    """
    if margin <= 0.0:
        return 1.0 if score != threshold else 0.0

    confidence = abs(score - threshold) / margin
    return float(np.clip(confidence, 0.0, 1.0))


def _build_explanation(
    classification: str,
    artifact_score: float,
    depth_ratio: float,
    area_score: float,
    shape_score: float,
    infrastructure: bool,
    known_basin: bool,
    dsm_bias: bool,
) -> str:
    """
    Build a concise human-readable classification explanation.

    :param classification: Final classification label.
    :param artifact_score: Final artifact-likelihood score.
    :param depth_ratio: Depth-to-accuracy ratio.
    :param area_score: Area-derived artifact score.
    :param shape_score: Shape-derived artifact score.
    :param infrastructure: Infrastructure evidence flag.
    :param known_basin: Known-basin evidence flag.
    :param dsm_bias: DSM-bias evidence flag.
    :returns: Explanation string.
    """
    evidence_terms = []

    if infrastructure:
        evidence_terms.append("infrastructure evidence")

    if known_basin:
        evidence_terms.append("known-basin evidence")

    if dsm_bias:
        evidence_terms.append("DSM-bias evidence")

    if evidence_terms:
        evidence_text = ", ".join(evidence_terms)
    else:
        evidence_text = "no positive contextual evidence"

    if classification == ARTIFACT:
        decision_text = "High artifact likelihood"
    elif classification == REAL_FEATURE:
        decision_text = "Low artifact likelihood"
    else:
        decision_text = "Evidence is borderline and requires analyst review"

    return (
        f"{decision_text}; score={artifact_score:.3f}, "
        f"depth_ratio={depth_ratio:.3f}, "
        f"area_score={area_score:.3f}, "
        f"shape_score={shape_score:.3f}; "
        f"{evidence_text}."
    )
