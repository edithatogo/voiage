"""Fixture-backed VOI for interoperability and data standardization."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class InteroperabilityStandardizationResult:
    """Result envelope for harmonized evidence reuse and standardization."""

    value: float
    reusable_evidence_indices: np.ndarray
    reuse_probability: float
    harmonization_score: float
    standardization_value: float
    data_usability_value: float
    reduced_transformation_error: float
    evidence_reuse_value: float
    standardization_cost: float
    baseline_value: float
    expected_value_harmonized: float
    harmonization_threshold: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_interoperability_standardization(
    evidence_utilities: np.ndarray | list[float],
    schema_compatibility: np.ndarray | list[float],
    semantic_alignment: np.ndarray | list[float],
    data_usability: np.ndarray | list[float],
    transformation_error_rates: np.ndarray | list[float],
    standardization_costs: np.ndarray | list[float],
    reuse_probabilities: np.ndarray | list[float],
    *,
    harmonization_threshold: float = 0.5,
) -> InteroperabilityStandardizationResult:
    """Estimate the value of harmonizing evidence for cross-site reuse.

    Each row represents an evidence source or schema mapping. Sources become
    reusable when the mean of schema compatibility, semantic alignment, and
    data usability meets ``harmonization_threshold``. The output separates
    evidence reuse, reduced transformation error, and standardization cost.
    This remains fixture-backed pending licensed cross-site data, binding
    parity, and mature/stable governance review.
    """
    arrays = [
        np.asarray(values, dtype=DEFAULT_DTYPE)
        for values in (
            evidence_utilities,
            schema_compatibility,
            semantic_alignment,
            data_usability,
            transformation_error_rates,
            standardization_costs,
            reuse_probabilities,
        )
    ]
    utilities, compatibility, semantics, usability, errors, costs, reuse = arrays
    if utilities.ndim != 1 or len(utilities) < 2:
        raise_input_error("evidence_utilities must be a vector with at least two rows.")
    if any(array.ndim != 1 or len(array) != len(utilities) for array in arrays[1:]):
        raise_input_error(
            "interoperability inputs must have the same one-dimensional length."
        )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise_input_error("interoperability inputs must be finite.")
    if np.any(utilities < 0) or np.any(costs < 0):
        raise_input_error(
            "evidence utilities and standardization costs must be non-negative."
        )
    if any(
        np.any((array < 0) | (array > 1))
        for array in (compatibility, semantics, usability, errors, reuse)
    ):
        raise_input_error(
            "interoperability rates and probabilities must be between zero and one."
        )
    if (
        not np.isfinite(harmonization_threshold)
        or not 0 <= harmonization_threshold <= 1
    ):
        raise_input_error(
            "harmonization_threshold must be finite and between zero and one."
        )

    harmonization_scores = (compatibility + semantics + usability) / 3.0
    reusable = harmonization_scores >= harmonization_threshold
    reuse_probability = float(np.mean(reusable))
    harmonization_score = float(np.mean(harmonization_scores))
    standardization_cost = float(np.sum(costs[reusable]))
    evidence_reuse_value = float(np.sum(utilities[reusable] * reuse[reusable]))
    data_usability_value = float(np.sum(utilities[reusable] * usability[reusable]))
    reduced_transformation_error = float(
        np.sum(utilities[reusable] * errors[reusable] * semantics[reusable])
    )
    standardization_value = (
        evidence_reuse_value + data_usability_value - standardization_cost
    )
    expected_value = standardization_value + reduced_transformation_error
    baseline_value = float(np.max(utilities))
    value = max(0.0, expected_value - baseline_value)
    diagnostics: dict[str, object] = {
        "source_count": len(utilities),
        "reusable_source_count": int(np.sum(reusable)),
        "evidence_reuse": True,
        "harmonization_score_definition": "mean of schema compatibility, semantic alignment, and data usability",
        "data_usability_definition": "utility-weighted usability for harmonized reusable evidence",
        "transformation_error_definition": "utility-weighted source error reduced by semantic alignment",
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed cross-site common-data-model snapshot committed",
        "stable_promotion": "blocked pending open-data provenance, cross-language parity, and governance review",
    }
    return InteroperabilityStandardizationResult(
        value=value,
        reusable_evidence_indices=np.flatnonzero(reusable),
        reuse_probability=reuse_probability,
        harmonization_score=harmonization_score,
        standardization_value=standardization_value,
        data_usability_value=data_usability_value,
        reduced_transformation_error=reduced_transformation_error,
        evidence_reuse_value=evidence_reuse_value,
        standardization_cost=standardization_cost,
        baseline_value=baseline_value,
        expected_value_harmonized=expected_value,
        harmonization_threshold=harmonization_threshold,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_interoperability_standardization",
            method_family="interoperability_standardization",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
