"""Fixture-backed VOI for replication and reproducibility decisions."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class ReplicationReproducibilityResult:
    """Result envelope for replication, audit, and reanalysis value."""

    value: float
    selected_replication_indices: np.ndarray
    replication_probability: float
    reproducibility_risk: float
    audit_burden: float
    credibility_impact: float
    evidence_downgrade: float
    replication_value: float
    reanalysis_value: float
    audit_cost: float
    baseline_value: float
    expected_value_replication: float
    replication_threshold: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_replication_reproducibility(
    evidence_values: np.ndarray | list[float],
    replication_probabilities: np.ndarray | list[float],
    reproducibility_failure_risks: np.ndarray | list[float],
    audit_costs: np.ndarray | list[float],
    reanalysis_values: np.ndarray | list[float],
    credibility_adjustments: np.ndarray | list[float],
    evidence_downgrades: np.ndarray | list[float],
    *,
    replication_threshold: float = 0.5,
) -> ReplicationReproducibilityResult:
    """Estimate value from independent replication and reproducibility work.

    Rows are selected when successful replication probability, adjusted for
    reproducibility failure risk, meets ``replication_threshold``. This is
    fixture-backed pending real evidence-production data, binding parity, and
    mature/stable governance review.
    """
    arrays = [
        np.asarray(values, dtype=DEFAULT_DTYPE)
        for values in (
            evidence_values,
            replication_probabilities,
            reproducibility_failure_risks,
            audit_costs,
            reanalysis_values,
            credibility_adjustments,
            evidence_downgrades,
        )
    ]
    values, probabilities, risks, costs, reanalysis, credibility, downgrades = arrays
    if values.ndim != 1 or len(values) < 2:
        raise_input_error("evidence_values must be a vector with at least two rows.")
    if any(array.ndim != 1 or len(array) != len(values) for array in arrays[1:]):
        raise_input_error(
            "replication inputs must have the same one-dimensional length."
        )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise_input_error("replication inputs must be finite.")
    if np.any(values < 0) or np.any(costs < 0) or np.any(reanalysis < 0):
        raise_input_error(
            "evidence values, reanalysis values, and audit costs must be non-negative."
        )
    if any(
        np.any((array < 0) | (array > 1))
        for array in (probabilities, risks, downgrades)
    ):
        raise_input_error(
            "replication probabilities, risks, and downgrades must be between zero and one."
        )
    if np.any(credibility < 0):
        raise_input_error("credibility_adjustments must be non-negative.")
    if not np.isfinite(replication_threshold) or not 0 <= replication_threshold <= 1:
        raise_input_error(
            "replication_threshold must be finite and between zero and one."
        )

    successful_probability = probabilities * (1.0 - risks)
    selected = successful_probability >= replication_threshold
    replication_probability = float(np.mean(selected))
    reproducibility_risk = float(np.mean(risks))
    audit_burden = float(np.sum(costs[selected]))
    credibility_impact = float(np.sum(values[selected] * (credibility[selected] - 1.0)))
    evidence_downgrade = float(np.sum(values[selected] * downgrades[selected]))
    replication_value = float(
        np.sum(
            values[selected] * successful_probability[selected] * credibility[selected]
        )
    )
    reanalysis_value = float(np.sum(reanalysis[selected]))
    audit_cost = audit_burden
    expected_value = (
        replication_value + reanalysis_value - audit_cost - evidence_downgrade
    )
    baseline_value = float(np.max(values))
    value = max(0.0, expected_value - baseline_value)
    diagnostics: dict[str, object] = {
        "source_count": len(values),
        "selected_source_count": int(np.sum(selected)),
        "replication_value_definition": "utility-weighted successful replication probability and credibility adjustment",
        "reproducibility_risk_definition": "mean probability of reproducibility failure",
        "evidence_downgrade_definition": "utility-weighted downgrade applied to selected evidence",
        "parity_status": "deferred",
        "open_data_status": "blocked: no evidence-production snapshot with provenance committed",
        "stable_promotion": "blocked pending open-data provenance, cross-language parity, and governance review",
    }
    return ReplicationReproducibilityResult(
        value=value,
        selected_replication_indices=np.flatnonzero(selected),
        replication_probability=replication_probability,
        reproducibility_risk=reproducibility_risk,
        audit_burden=audit_burden,
        credibility_impact=credibility_impact,
        evidence_downgrade=evidence_downgrade,
        replication_value=replication_value,
        reanalysis_value=reanalysis_value,
        audit_cost=audit_cost,
        baseline_value=baseline_value,
        expected_value_replication=expected_value,
        replication_threshold=replication_threshold,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_replication_reproducibility",
            method_family="replication_reproducibility",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
