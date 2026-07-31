"""Reporting helpers for governed VOI payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voiage.contracts.estimation import EstimationVarianceResult


def build_cheers_reporting(
    *,
    analysis_type: str,
    method_family: str,
    method_maturity: str,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
    perspective_ids: list[str] | None = None,
    perspective_labels: list[str] | None = None,
    population: float | None = None,
    estimator: str | None = None,
    seed: int | None = None,
    provenance: dict[str, Any] | None = None,
    reproducibility: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build a CHEERS-VOI aligned reporting payload.

    The payload is intentionally conservative: it captures the fields that are
    broadly useful for reporting and reproducibility without forcing every
    frontier method to invent an artificial decision-problem identity.
    """
    payload: dict[str, object] = {
        "reporting_standard": "CHEERS-VOI",
        "analysis_type": analysis_type,
        "method_family": method_family,
        "method_maturity": method_maturity,
        "analysis_id": analysis_id,
        "decision_problem_id": decision_problem_id,
        "decision_context": decision_context,
        "population": population,
        "estimator": estimator,
        "seed": seed,
        "provenance": dict(provenance or {}),
        "reproducibility": dict(reproducibility or {}),
        "diagnostics": dict(diagnostics or {}),
    }
    if perspective_ids is not None:
        payload["perspective_ids"] = [str(item) for item in perspective_ids]
    if perspective_labels is not None:
        payload["perspective_labels"] = [str(item) for item in perspective_labels]
    return payload


def build_estimation_variance_reporting(
    result: EstimationVarianceResult,
) -> dict[str, object]:
    """Build a portable report without changing the numerical result."""
    return {
        "reporting_standard": "VOIAGE estimation-variance v1",
        "method_family": "estimation-focused-variance-voi",
        "method_id": result.method_id,
        "target": result.target.model_dump(mode="json"),
        "functional_units": result.functional_units,
        "comparison": [
            {
                "information_state": "current",
                "functional_value": result.prior_functional,
            },
            {
                "information_state": "after_information",
                "functional_value": result.expected_posterior_functional,
            },
        ],
        "raw_reduction": result.raw_reduction,
        "absolute_reduction": result.absolute_reduction,
        "relative_reduction": result.relative_reduction,
        "negative_estimate_policy": result.negative_estimate_policy,
        "zero_variance_policy": result.zero_variance_policy,
        "assurance": result.diagnostics.model_dump(mode="json"),
        "provenance": result.provenance.model_dump(mode="json"),
        "maturity": "experimental",
        "limitations": [
            "The executable runtime supports scalar variance targets only.",
            "Vector covariance scalarization remains pending scientific review.",
        ],
    }
