"""Fixture-backed VOI for evidence obsolescence and refresh decisions."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class EvidenceObsolescenceRefreshResult:
    """Result envelope for living evidence and refresh value."""

    value: float
    selected_refresh_indices: np.ndarray
    evidence_age_months: float
    obsolescence_risk: float
    refresh_burden: float
    update_cadence_months: float
    living_review_value: float
    model_refresh_value: float
    obsolete_decision_risk: float
    refresh_value: float
    update_cost: float
    baseline_value: float
    expected_value_refresh: float
    refresh_threshold: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_evidence_obsolescence_refresh(
    evidence_values: np.ndarray | list[float],
    evidence_age_months: np.ndarray | list[float],
    half_lives_months: np.ndarray | list[float],
    obsolescence_risks: np.ndarray | list[float],
    refresh_costs: np.ndarray | list[float],
    living_review_values: np.ndarray | list[float],
    model_refresh_values: np.ndarray | list[float],
    drift_rates: np.ndarray | list[float],
    *,
    refresh_threshold: float = 0.5,
    refresh_cadence_months: float = 12.0,
) -> EvidenceObsolescenceRefreshResult:
    """Estimate value from refreshing aging or drifting evidence.

    A source is selected when its obsolescence risk multiplied by its drift
    rate reaches ``refresh_threshold``. The method remains fixture-backed
    pending longitudinal evidence data, binding parity, and promotion review.
    """
    arrays = [
        np.asarray(values, dtype=DEFAULT_DTYPE)
        for values in (
            evidence_values,
            evidence_age_months,
            half_lives_months,
            obsolescence_risks,
            refresh_costs,
            living_review_values,
            model_refresh_values,
            drift_rates,
        )
    ]
    values, ages, half_lives, risks, costs, living, model, drift = arrays
    if values.ndim != 1 or len(values) < 2:
        raise_input_error("evidence_values must be a vector with at least two rows.")
    if any(array.ndim != 1 or len(array) != len(values) for array in arrays[1:]):
        raise_input_error("refresh inputs must have the same one-dimensional length.")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise_input_error("refresh inputs must be finite.")
    if np.any(values < 0) or np.any(ages < 0) or np.any(half_lives <= 0):
        raise_input_error(
            "values and ages must be non-negative; half-lives must be positive."
        )
    if np.any(costs < 0) or np.any(living < 0) or np.any(model < 0):
        raise_input_error("refresh costs and refresh values must be non-negative.")
    if any(np.any((array < 0) | (array > 1)) for array in (risks, drift)):
        raise_input_error(
            "obsolescence risks and drift rates must be between zero and one."
        )
    if not np.isfinite(refresh_threshold) or not 0 <= refresh_threshold <= 1:
        raise_input_error("refresh_threshold must be finite and between zero and one.")
    if not np.isfinite(refresh_cadence_months) or refresh_cadence_months <= 0:
        raise_input_error("refresh_cadence_months must be finite and positive.")

    priority = risks * drift
    selected = priority >= refresh_threshold
    age = float(np.mean(ages))
    obsolescence = float(np.mean(risks))
    refresh_burden = float(np.sum(costs[selected]))
    update_cost = refresh_burden
    living_value = float(np.sum(living[selected]))
    model_value = float(np.sum(model[selected]))
    obsolete_decision_risk = float(np.sum(values[selected] * priority[selected]))
    refresh_value = float(obsolete_decision_risk + living_value + model_value)
    expected_value = refresh_value - update_cost
    baseline = float(np.max(values))
    value = max(0.0, expected_value - baseline)
    diagnostics: dict[str, object] = {
        "source_count": len(values),
        "selected_source_count": int(np.sum(selected)),
        "evidence_age_definition": "mean age in months across evidence sources",
        "obsolescence_definition": "mean source-level obsolescence risk",
        "update_cadence_definition": "configured recurring refresh interval in months",
        "priority_definition": "obsolescence risk multiplied by drift rate",
        "parity_status": "deferred",
        "open_data_status": "blocked: no longitudinal evidence-refresh snapshot with provenance committed",
        "stable_promotion": "blocked pending longitudinal data, cross-language parity, and governance review",
    }
    return EvidenceObsolescenceRefreshResult(
        value=value,
        selected_refresh_indices=np.flatnonzero(selected),
        evidence_age_months=age,
        obsolescence_risk=obsolescence,
        refresh_burden=refresh_burden,
        update_cadence_months=float(refresh_cadence_months),
        living_review_value=living_value,
        model_refresh_value=model_value,
        obsolete_decision_risk=obsolete_decision_risk,
        refresh_value=refresh_value,
        update_cost=update_cost,
        baseline_value=baseline,
        expected_value_refresh=expected_value,
        refresh_threshold=refresh_threshold,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_evidence_obsolescence_refresh",
            method_family="evidence_obsolescence_refresh",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
