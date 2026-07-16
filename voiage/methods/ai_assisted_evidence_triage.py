"""Fixture-backed VOI for AI-assisted evidence triage."""

from dataclasses import dataclass
import math

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class AIAssistedEvidenceTriageResult:
    """Result envelope for human-in-the-loop evidence triage economics."""

    value: float
    selected_item_indices: np.ndarray
    triage_threshold: float
    sensitivity: float
    specificity: float
    effective_sensitivity: float
    effective_specificity: float
    false_exclusion_risk: float
    false_inclusion_burden: float
    effective_false_exclusion_impact: float
    effective_false_inclusion_impact: float
    reviewer_time_saved: float
    audit_value: float
    audit_items: int
    extraction_error_burden: float
    model_drift: float
    expected_value_ai_triage: float
    baseline_value: float
    automation_cost: float
    audit_cost: float
    human_override_rate: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_ai_assisted_evidence_triage(
    relevance_labels: np.ndarray | list[int] | list[bool],
    triage_scores: np.ndarray | list[float],
    decision_impacts: np.ndarray | list[float],
    reviewer_time_minutes: np.ndarray | list[float],
    extraction_error_rates: np.ndarray | list[float],
    *,
    triage_threshold: float = 0.5,
    audit_sample_rate: float = 0.0,
    human_override_rate: float = 0.0,
    model_drift: float = 0.0,
    automation_cost: float = 0.0,
    audit_cost_per_item: float = 0.0,
    reviewer_cost_per_minute: float = 1.0,
) -> AIAssistedEvidenceTriageResult:
    """Estimate the value of automated evidence triage with human oversight.

    ``relevance_labels`` and ``triage_scores`` form a deterministic reference
    evaluation set. Scores at or above ``triage_threshold`` are selected for
    downstream review. Audit sampling and human overrides recover a bounded
    fraction of false exclusions and false inclusions, while extraction error,
    model drift, automation cost, and reviewer time are included in the net
    decision value. The result remains fixture-backed pending external model
    validation, licensed evidence corpora, and cross-language parity.
    """
    labels = np.asarray(relevance_labels, dtype=int)
    scores = np.asarray(triage_scores, dtype=DEFAULT_DTYPE)
    impacts = np.asarray(decision_impacts, dtype=DEFAULT_DTYPE)
    reviewer_time = np.asarray(reviewer_time_minutes, dtype=DEFAULT_DTYPE)
    extraction_errors = np.asarray(extraction_error_rates, dtype=DEFAULT_DTYPE)
    arrays = (scores, impacts, reviewer_time, extraction_errors)
    if labels.ndim != 1 or len(labels) < 2 or not np.all(np.isin(labels, [0, 1])):
        raise_input_error("relevance_labels must be a binary one-dimensional vector.")
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise_input_error(
            "relevance_labels must contain both relevant and irrelevant items."
        )
    if any(array.ndim != 1 or len(array) != len(labels) for array in arrays):
        raise_input_error("triage arrays must have the same one-dimensional length.")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise_input_error("triage_scores must be finite values between zero and one.")
    if not np.all(np.isfinite(impacts)) or np.any(impacts < 0):
        raise_input_error("decision_impacts must be finite and non-negative.")
    if not np.all(np.isfinite(reviewer_time)) or np.any(reviewer_time < 0):
        raise_input_error("reviewer_time_minutes must be finite and non-negative.")
    if not np.all(np.isfinite(extraction_errors)) or np.any(
        (extraction_errors < 0) | (extraction_errors > 1)
    ):
        raise_input_error(
            "extraction_error_rates must be finite values between zero and one."
        )
    bounded = {
        "triage_threshold": triage_threshold,
        "audit_sample_rate": audit_sample_rate,
        "human_override_rate": human_override_rate,
        "model_drift": model_drift,
    }
    for name, value in bounded.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise_input_error(f"{name} must be finite and between zero and one.")
    costs = {
        "automation_cost": automation_cost,
        "audit_cost_per_item": audit_cost_per_item,
        "reviewer_cost_per_minute": reviewer_cost_per_minute,
    }
    for name, value in costs.items():
        if not np.isfinite(value) or value < 0:
            raise_input_error(f"{name} must be finite and non-negative.")

    predicted = scores >= triage_threshold
    relevant = labels == 1
    irrelevant = ~relevant
    true_positive = relevant & predicted
    false_negative = relevant & ~predicted
    false_positive = irrelevant & predicted
    true_negative = irrelevant & ~predicted
    relevant_count = int(np.sum(relevant))
    irrelevant_count = int(np.sum(irrelevant))
    sensitivity = float(np.sum(true_positive) / relevant_count)
    specificity = float(np.sum(true_negative) / irrelevant_count)

    raw_false_exclusion = float(np.sum(impacts[false_negative]))
    raw_false_inclusion = float(np.sum(impacts[false_positive]))
    audit_items = math.ceil(np.sum(~predicted) * audit_sample_rate)
    audit_value = raw_false_exclusion * audit_sample_rate
    override_recovery = human_override_rate * raw_false_exclusion
    effective_false_exclusion = max(
        0.0, raw_false_exclusion - audit_value - override_recovery
    )
    effective_false_inclusion = raw_false_inclusion * (1.0 - human_override_rate)
    relevant_impact = float(np.sum(impacts[relevant]))
    irrelevant_impact = float(np.sum(impacts[irrelevant]))
    effective_sensitivity = max(
        0.0, 1.0 - effective_false_exclusion / max(relevant_impact, 1.0)
    )
    effective_specificity = max(
        0.0, 1.0 - effective_false_inclusion / max(irrelevant_impact, 1.0)
    )
    false_exclusion_risk = effective_false_exclusion / max(relevant_impact, 1.0)
    extraction_error_burden = float(
        np.sum(impacts[predicted] * extraction_errors[predicted])
    )
    false_inclusion_burden = effective_false_inclusion + extraction_error_burden
    baseline_time = float(np.sum(reviewer_time))
    selected_time = float(np.sum(reviewer_time[predicted]))
    excluded_time = reviewer_time[~predicted]
    audit_time = audit_items * float(np.mean(excluded_time)) if audit_items else 0.0
    automated_time = selected_time + audit_time
    reviewer_time_saved = baseline_time - automated_time
    audit_cost = audit_items * audit_cost_per_item
    drift_penalty = model_drift * relevant_impact
    baseline_value = relevant_impact - baseline_time * reviewer_cost_per_minute
    expected_value = (
        relevant_impact
        - effective_false_exclusion
        - false_inclusion_burden
        - drift_penalty
        - automation_cost
        - audit_cost
        - automated_time * reviewer_cost_per_minute
    )
    value = max(0.0, expected_value - baseline_value)
    diagnostics: dict[str, object] = {
        "item_count": len(labels),
        "selected_item_count": int(np.sum(predicted)),
        "audit_items": audit_items,
        "human_in_the_loop": True,
        "human_override_rate": human_override_rate,
        "triage_threshold": triage_threshold,
        "false_exclusion_definition": "decision-impact-weighted relevant items not selected after audit and override recovery",
        "false_inclusion_definition": "decision-impact-weighted irrelevant items selected after override recovery",
        "audit_policy": "deterministic rate over non-selected items",
        "model_drift_penalty": drift_penalty,
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed evidence-corpus snapshot committed",
        "stable_promotion": "blocked pending external model validation, corpus governance, and parity review",
    }
    return AIAssistedEvidenceTriageResult(
        value=value,
        selected_item_indices=np.flatnonzero(predicted),
        triage_threshold=triage_threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        effective_sensitivity=effective_sensitivity,
        effective_specificity=effective_specificity,
        false_exclusion_risk=false_exclusion_risk,
        false_inclusion_burden=false_inclusion_burden,
        effective_false_exclusion_impact=effective_false_exclusion,
        effective_false_inclusion_impact=effective_false_inclusion,
        reviewer_time_saved=reviewer_time_saved,
        audit_value=audit_value,
        audit_items=audit_items,
        extraction_error_burden=extraction_error_burden,
        model_drift=model_drift,
        expected_value_ai_triage=expected_value,
        baseline_value=baseline_value,
        automation_cost=automation_cost,
        audit_cost=audit_cost,
        human_override_rate=human_override_rate,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_ai_assisted_evidence_triage",
            method_family="ai_assisted_evidence_triage",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
