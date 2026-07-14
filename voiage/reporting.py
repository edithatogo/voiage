"""Reporting helpers for CHEERS-VOI-aligned payloads."""

from typing import Any


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
