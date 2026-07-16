"""Fixture-backed VOI for explainability and model transparency."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class ExplainabilityTransparencyResult:
    """Result envelope for adoption and governance value of explanations."""

    value: float
    adopted_model_indices: np.ndarray
    adoption_probability: float
    transparency_value: float
    governance_value: float
    predictive_value: float
    trust_value: float
    robustness: float
    audit_cost: float
    adoption_threshold: float
    transparency_weight: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_explainability_transparency(
    predictive_utilities: np.ndarray | list[float],
    explanation_quality: np.ndarray | list[float],
    transparency_evidence: np.ndarray | list[float],
    trust_scores: np.ndarray | list[float],
    governance_impacts: np.ndarray | list[float],
    audit_costs: np.ndarray | list[float],
    *,
    adoption_threshold: float = 0.5,
    transparency_weight: float = 0.5,
) -> ExplainabilityTransparencyResult:
    """Estimate decision value from interpretable, transparent model evidence.

    Each row is a candidate model or governance scenario. Adoption is based on
    the mean of explanation quality, transparency evidence, and trust. The
    value separates predictive utility from transparency-driven adoption and
    governance value. The method remains fixture-backed pending external
    model validation, open-data provenance, and cross-language parity.
    """
    arrays = [
        np.asarray(values, dtype=DEFAULT_DTYPE)
        for values in (
            predictive_utilities,
            explanation_quality,
            transparency_evidence,
            trust_scores,
            governance_impacts,
            audit_costs,
        )
    ]
    predictive, quality, evidence, trust, governance, audits = arrays
    if predictive.ndim != 1 or len(predictive) < 2:
        raise_input_error(
            "predictive_utilities must be a vector with at least two rows."
        )
    if any(array.ndim != 1 or len(array) != len(predictive) for array in arrays[1:]):
        raise_input_error(
            "explainability inputs must have the same one-dimensional length."
        )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise_input_error("explainability inputs must be finite.")
    if any(np.any((array < 0) | (array > 1)) for array in (quality, evidence, trust)):
        raise_input_error(
            "explanation quality, transparency evidence, and trust must be between zero and one."
        )
    if np.any(governance < 0) or np.any(audits < 0):
        raise_input_error("governance impacts and audit costs must be non-negative.")
    if not np.isfinite(adoption_threshold) or not 0 <= adoption_threshold <= 1:
        raise_input_error("adoption_threshold must be finite and between zero and one.")
    if not np.isfinite(transparency_weight) or not 0 <= transparency_weight <= 1:
        raise_input_error(
            "transparency_weight must be finite and between zero and one."
        )

    adoption_score = (quality + evidence + trust) / 3.0
    adopted = adoption_score >= adoption_threshold
    adoption_probability = float(np.mean(adopted))
    predictive_value = float(np.sum(predictive[adopted]))
    trust_value = float(np.sum(trust[adopted] * governance[adopted]))
    governance_value = float(np.sum(governance[adopted] * evidence[adopted]))
    transparency_value = (
        transparency_weight * governance_value
        + (1.0 - transparency_weight) * trust_value
    )
    audit_cost = float(np.sum(audits[adopted]))
    expected_value = predictive_value + transparency_value - audit_cost
    baseline_value = float(np.max(predictive))
    value = max(0.0, expected_value - baseline_value)
    robustness = float(1.0 - np.std(adoption_score))
    diagnostics: dict[str, object] = {
        "model_count": len(predictive),
        "adopted_model_count": int(np.sum(adopted)),
        "transparency_evidence": True,
        "adoption_score_definition": "mean of explanation quality, transparency evidence, and trust",
        "predictive_improvement_separated": True,
        "model_governance_uncertainty": float(np.mean(1.0 - evidence)),
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed open-model evidence snapshot committed",
        "stable_promotion": "blocked pending external validation, provenance, parity, and governance review",
    }
    return ExplainabilityTransparencyResult(
        value=value,
        adopted_model_indices=np.flatnonzero(adopted),
        adoption_probability=adoption_probability,
        transparency_value=transparency_value,
        governance_value=governance_value,
        predictive_value=predictive_value,
        trust_value=trust_value,
        robustness=robustness,
        audit_cost=audit_cost,
        adoption_threshold=adoption_threshold,
        transparency_weight=transparency_weight,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_explainability_transparency",
            method_family="explainability_transparency",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
