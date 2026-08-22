"""ML, LLM, and Agent Value of Information: Decision-Focused Model Value & Policy Uplift VOI (#576, #578).

This module implements:
1. Decision-focused model evaluation, validation, and refresh triggers (#576):
   comparing models by downstream decision loss and policy regret rather than
   pure accuracy or AUC.
2. Policy and uplift value of information (#578): valuing heterogeneous treatment
   effects (CATE), potential outcomes, intervention costs, and constrained targeting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from voiage.exceptions import raise_input_error

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_VALUE_SCHEMA_PATH = (
    _ROOT
    / "specs"
    / "ml-voi"
    / "schemas"
    / "v1"
    / "decision-focused-model-value.schema.json"
)
_POLICY_UPLIFT_SCHEMA_PATH = (
    _ROOT / "specs" / "ml-voi" / "schemas" / "v1" / "policy-uplift-voi.schema.json"
)


def _current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CandidateModelEvaluation:
    """Evaluation summary for a single candidate ML model."""

    model_id: str
    predictive_metric_score: float
    expected_decision_value: float
    policy_regret: float


@dataclass(frozen=True)
class DecisionFocusedModelValueResult:
    """Result of decision-focused model value evaluation and refresh analysis."""

    evaluation_id: str
    evaluated_at: str
    candidate_models: list[CandidateModelEvaluation]
    selected_model: str
    downstream_metrics: dict[str, float]
    refresh_recommendation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluated_at": self.evaluated_at,
            "candidate_models": [asdict(m) for m in self.candidate_models],
            "selected_model": self.selected_model,
            "downstream_metrics": dict(self.downstream_metrics),
            "refresh_recommendation": dict(self.refresh_recommendation),
        }


@dataclass(frozen=True)
class UpliftVOIResult:
    """Result of heterogeneous policy and uplift VOI calculation."""

    evaluation_id: str
    evaluated_at: str
    status_quo_value: float
    optimal_policy_value: float
    uplift_evpi: float
    budget_utilized: float
    units_targeted: int
    subgroup_evppi: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluated_at": self.evaluated_at,
            "status_quo_value": self.status_quo_value,
            "optimal_policy_value": self.optimal_policy_value,
            "uplift_evpi": self.uplift_evpi,
            "budget_utilized": self.budget_utilized,
            "units_targeted": self.units_targeted,
            "subgroup_evppi": dict(self.subgroup_evppi),
        }


def evaluate_decision_focused_model_value(
    candidate_predictions: dict[str, np.ndarray],
    actual_outcomes: np.ndarray,
    intervention_cost: float,
    intervention_payoff: float,
    predictive_scores: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
    evaluation_id: str = "eval_ml_model_01",
    current_production_model_id: str | None = None,
    regret_refresh_threshold: float = 1000.0,
) -> DecisionFocusedModelValueResult:
    """Evaluate candidate models by downstream economic decision value.

    Parameters
    ----------
    candidate_predictions : dict[str, np.ndarray]
        Mapping from model_id to 1D array of predicted probabilities.
    actual_outcomes : np.ndarray
        1D array of true binary outcomes (0 or 1).
    intervention_cost : float
        Cost of taking the proactive intervention per unit.
    intervention_payoff : float
        Payoff/value saved when intervening on a positive outcome unit.
    predictive_scores : dict[str, float], optional
        Standard predictive metric (e.g. AUC-ROC or Brier score) for reference.
    decision_threshold : float
        Probability threshold for triggering intervention. Default 0.5.
    evaluation_id : str
        Unique evaluation ID.
    current_production_model_id : str, optional
        ID of currently deployed baseline model.
    regret_refresh_threshold : float
        Dollar threshold of policy regret above which refresh is recommended.

    Returns
    -------
    DecisionFocusedModelValueResult
        Decision value, regret, chosen model, and refresh recommendation.
    """
    if not candidate_predictions:
        raise_input_error("candidate_predictions mapping cannot be empty.")
    if len(actual_outcomes) == 0:
        raise_input_error("actual_outcomes array cannot be empty.")

    scores = predictive_scores or {}
    n_units = len(actual_outcomes)

    # Calculate perfect clairvoyance oracle value (EVPI ceiling)
    # Intervene whenever actual_outcome == 1 and payoff > cost
    perfect_interventions = (actual_outcomes == 1) & (
        intervention_payoff > intervention_cost
    )
    oracle_value = float(
        np.sum(perfect_interventions * (intervention_payoff - intervention_cost))
    )

    evaluations: list[CandidateModelEvaluation] = []
    values_by_model: dict[str, float] = {}

    for model_id, preds in candidate_predictions.items():
        if len(preds) != n_units:
            raise_input_error(
                f"Model '{model_id}' predictions length ({len(preds)}) "
                f"does not match actual_outcomes length ({n_units})."
            )
        intervene_flags = preds >= decision_threshold
        # Net value = (intervened & true_positive) * payoff - (intervened) * cost
        net_value = float(
            np.sum(
                intervene_flags
                * (actual_outcomes * intervention_payoff - intervention_cost)
            )
        )
        values_by_model[model_id] = net_value
        regret = max(0.0, oracle_value - net_value)

        evaluations.append(
            CandidateModelEvaluation(
                model_id=model_id,
                predictive_metric_score=float(scores.get(model_id, 0.0)),
                expected_decision_value=net_value,
                policy_regret=regret,
            )
        )

    # Sort candidate models by expected decision value descending
    evaluations.sort(key=lambda m: m.expected_decision_value, reverse=True)
    best_model = evaluations[0]

    baseline_val = (
        values_by_model[current_production_model_id]
        if current_production_model_id
        and current_production_model_id in values_by_model
        else evaluations[-1].expected_decision_value
    )

    upgrade_value = max(0.0, best_model.expected_decision_value - baseline_val)
    should_refresh = (
        best_model.model_id != current_production_model_id
        and upgrade_value >= regret_refresh_threshold
    )

    refresh_rec = {
        "should_refresh": bool(should_refresh),
        "rationale": (
            f"Upgrading from '{current_production_model_id}' to '{best_model.model_id}' "
            f"yields ${upgrade_value:,.2f} incremental net decision value."
            if should_refresh
            else "Current model is optimal or upgrade gain is below refresh threshold."
        ),
        "drift_threshold_exceeded": bool(should_refresh),
    }

    return DecisionFocusedModelValueResult(
        evaluation_id=evaluation_id,
        evaluated_at=_current_iso_timestamp(),
        candidate_models=evaluations,
        selected_model=best_model.model_id,
        downstream_metrics={
            "max_decision_value": best_model.expected_decision_value,
            "min_policy_regret": best_model.policy_regret,
            "value_of_model_upgrade": upgrade_value,
        },
        refresh_recommendation=refresh_rec,
    )


def compute_policy_uplift_voi(
    cate_samples: np.ndarray,
    intervention_cost: float,
    payoff_multiplier: float,
    budget_constraint: float | None = None,
    subgroups: dict[str, np.ndarray] | None = None,
    evaluation_id: str = "eval_uplift_01",
) -> UpliftVOIResult:
    """Compute heterogeneous policy and uplift Value of Information.

    Parameters
    ----------
    cate_samples : np.ndarray
        2D array of shape (n_simulations, n_units) containing posterior CATE draws
        (treatment effect on event probability or outcome).
    intervention_cost : float
        Unit cost to apply intervention.
    payoff_multiplier : float
        Value generated per unit of treatment effect (e.g. CLV or saved cost).
    budget_constraint : float, optional
        Maximum total spend allowed across units.
    subgroups : dict[str, np.ndarray], optional
        Boolean masks of shape (n_units,) defining named customer segments.
    evaluation_id : str
        Unique evaluation identifier.

    Returns
    -------
    UpliftVOIResult
        Expected net benefit under current policy, optimal policy, and uplift EVPI.
    """
    if cate_samples.ndim != 2:
        raise_input_error("cate_samples must be a 2D array of shape (n_sims, n_units).")
    n_sims, n_units = cate_samples.shape
    if n_sims == 0 or n_units == 0:
        raise_input_error("cate_samples cannot have zero dimensions.")

    # Net benefit per unit under mean expected CATE
    mean_cate = np.mean(cate_samples, axis=0)
    expected_unit_net_benefit = mean_cate * payoff_multiplier - intervention_cost

    # Current optimal policy without sample-level clairvoyance
    # Intervene if expected net benefit > 0
    if budget_constraint is not None:
        max_units = int(budget_constraint // intervention_cost)
        sorted_indices = np.argsort(expected_unit_net_benefit)[::-1]
        target_indices = sorted_indices[: max(0, max_units)]
        # Only take positive net benefit units
        target_indices = [
            idx for idx in target_indices if expected_unit_net_benefit[idx] > 0
        ]
        chosen_units_mask = np.zeros(n_units, dtype=bool)
        chosen_units_mask[target_indices] = True
    else:
        chosen_units_mask = expected_unit_net_benefit > 0

    units_targeted = int(np.sum(chosen_units_mask))
    budget_utilized = float(units_targeted * intervention_cost)

    # Current expected policy value
    status_quo_value = 0.0  # Baseline doing nothing
    current_policy_value = float(np.sum(expected_unit_net_benefit[chosen_units_mask]))

    # Perfect information clairvoyance value: for each simulation s, choose optimal units
    sample_unit_net_benefits = cate_samples * payoff_multiplier - intervention_cost

    perfect_values: list[float] = []
    for s in range(n_sims):
        s_benefits = sample_unit_net_benefits[s]
        if budget_constraint is not None:
            max_units = int(budget_constraint // intervention_cost)
            s_sorted = np.argsort(s_benefits)[::-1]
            s_targets = s_sorted[: max(0, max_units)]
            s_targets = [idx for idx in s_targets if s_benefits[idx] > 0]
            val = float(np.sum(s_benefits[s_targets])) if s_targets else 0.0
        else:
            val = float(np.sum(s_benefits[s_benefits > 0]))
        perfect_values.append(val)

    expected_perfect_value = float(np.mean(perfect_values))
    uplift_evpi = max(0.0, expected_perfect_value - current_policy_value)

    # Subgroup EVPPI calculations if provided
    subgroup_evppi: dict[str, float] = {}
    if subgroups:
        for sg_name, sg_mask in subgroups.items():
            if len(sg_mask) != n_units:
                continue
            # EVPPI on subgroup: resolve uncertainty for subgroup units only
            sg_evppi_val = float(
                np.mean(
                    np.maximum(0.0, sample_unit_net_benefits[:, sg_mask]).sum(axis=1)
                )
                - np.sum(np.maximum(0.0, expected_unit_net_benefit[sg_mask]))
            )
            subgroup_evppi[sg_name] = max(0.0, sg_evppi_val)

    return UpliftVOIResult(
        evaluation_id=evaluation_id,
        evaluated_at=_current_iso_timestamp(),
        status_quo_value=status_quo_value,
        optimal_policy_value=current_policy_value,
        uplift_evpi=uplift_evpi,
        budget_utilized=budget_utilized,
        units_targeted=units_targeted,
        subgroup_evppi=subgroup_evppi,
    )


def validate_decision_focused_model_value(
    record_dict: dict[str, Any], schema_path: Path | None = None
) -> bool:
    """Validate a dictionary against the decision-focused model value schema."""
    s_path = schema_path or _MODEL_VALUE_SCHEMA_PATH
    if not s_path.is_file():
        raise_input_error(f"Schema not found at {s_path}")
    schema = json.loads(s_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=record_dict, schema=schema)
    return True


def validate_policy_uplift_voi(
    record_dict: dict[str, Any], schema_path: Path | None = None
) -> bool:
    """Validate a dictionary against the policy uplift VOI schema."""
    s_path = schema_path or _POLICY_UPLIFT_SCHEMA_PATH
    if not s_path.is_file():
        raise_input_error(f"Schema not found at {s_path}")
    schema = json.loads(s_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=record_dict, schema=schema)
    return True
