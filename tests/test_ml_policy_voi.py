"""Tests for ML, LLM, and Agent VOI: Decision-Focused Model Value and Policy Uplift (#576, #578)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.ml_policy_voi import (
    compute_policy_uplift_voi,
    evaluate_decision_focused_model_value,
    validate_decision_focused_model_value,
    validate_policy_uplift_voi,
)


def test_decision_focused_model_evaluation_divergence_from_pure_metrics() -> None:
    # 10 test units with true binary outcomes (e.g. churn = 1)
    actual_outcomes = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    intervention_cost = 100.0
    intervention_payoff = (
        500.0  # Net gain per true positive = 400, net loss per false positive = -100
    )

    # Model A: High predictive rank score, but poor calibration around 0.5 threshold
    model_a_preds = np.array(
        [0.48, 0.49, 0.47, 0.51, 0.52, 0.53, 0.49, 0.51, 0.48, 0.52]
    )

    # Model B: Well-calibrated around decision threshold
    model_b_preds = np.array(
        [0.85, 0.90, 0.75, 0.10, 0.15, 0.20, 0.80, 0.05, 0.95, 0.12]
    )

    scores = {"model_a_high_auc": 0.92, "model_b_calibrated": 0.85}

    result = evaluate_decision_focused_model_value(
        candidate_predictions={
            "model_a_high_auc": model_a_preds,
            "model_b_calibrated": model_b_preds,
        },
        actual_outcomes=actual_outcomes,
        intervention_cost=intervention_cost,
        intervention_payoff=intervention_payoff,
        predictive_scores=scores,
        current_production_model_id="model_a_high_auc",
        regret_refresh_threshold=500.0,
    )

    # Model B should be selected despite lower predictive score
    assert result.selected_model == "model_b_calibrated"
    assert (
        result.downstream_metrics["max_decision_value"]
        > result.candidate_models[1].expected_decision_value
    )
    assert result.refresh_recommendation["should_refresh"] is True
    assert validate_decision_focused_model_value(result.to_dict()) is True


def test_decision_focused_model_evaluation_no_production_or_already_optimal() -> None:
    actual_outcomes = np.array([1, 0, 1])
    preds = {"m1": np.array([0.9, 0.1, 0.8])}

    # Case 1: No current production model
    res1 = evaluate_decision_focused_model_value(
        candidate_predictions=preds,
        actual_outcomes=actual_outcomes,
        intervention_cost=10.0,
        intervention_payoff=100.0,
        current_production_model_id=None,
    )
    assert res1.selected_model == "m1"
    assert res1.refresh_recommendation["should_refresh"] is False

    # Case 2: Current production model is already the best model
    res2 = evaluate_decision_focused_model_value(
        candidate_predictions=preds,
        actual_outcomes=actual_outcomes,
        intervention_cost=10.0,
        intervention_payoff=100.0,
        current_production_model_id="m1",
    )
    assert res2.refresh_recommendation["should_refresh"] is False


def test_compute_policy_uplift_voi() -> None:
    # 50 simulations, 10 units
    np.random.seed(42)
    # Mean treatment effect is positive for units 0..4, negative for 5..9
    cate_samples = np.random.normal(
        loc=[0.2, 0.15, 0.25, 0.1, 0.08, -0.05, -0.1, -0.02, 0.01, -0.15],
        scale=0.05,
        size=(100, 10),
    )

    intervention_cost = 20.0
    payoff_multiplier = 300.0  # Expected unit benefit = CATE * 300 - 20

    result = compute_policy_uplift_voi(
        cate_samples=cate_samples,
        intervention_cost=intervention_cost,
        payoff_multiplier=payoff_multiplier,
    )

    assert result.status_quo_value == 0.0
    assert result.optimal_policy_value > 0.0
    assert result.uplift_evpi >= 0.0
    assert result.units_targeted >= 3
    assert validate_policy_uplift_voi(result.to_dict()) is True


def test_policy_uplift_with_budget_and_subgroup_evppi() -> None:
    np.random.seed(42)
    cate_samples = np.random.normal(loc=0.15, scale=0.08, size=(100, 8))
    intervention_cost = 50.0
    payoff_multiplier = 500.0
    budget_constraint = 150.0  # Max 3 units targeted

    subgroups = {
        "enterprise_tier": np.array(
            [True, True, True, True, False, False, False, False]
        ),
        "smb_tier": np.array([False, False, False, False, True, True, True, True]),
        "invalid_size_subgroup": np.array([True, False]),  # Should be skipped safely
    }

    result = compute_policy_uplift_voi(
        cate_samples=cate_samples,
        intervention_cost=intervention_cost,
        payoff_multiplier=payoff_multiplier,
        budget_constraint=budget_constraint,
        subgroups=subgroups,
    )

    assert result.units_targeted <= 3
    assert result.budget_utilized <= budget_constraint
    assert "enterprise_tier" in result.subgroup_evppi
    assert "smb_tier" in result.subgroup_evppi
    assert "invalid_size_subgroup" not in result.subgroup_evppi
    assert validate_policy_uplift_voi(result.to_dict()) is True


def test_policy_uplift_with_tight_budget_and_negative_effects() -> None:
    # Test case where budget is small and effects are all negative
    cate_samples = np.full((10, 5), -0.1)
    result = compute_policy_uplift_voi(
        cate_samples=cate_samples,
        intervention_cost=50.0,
        payoff_multiplier=100.0,
        budget_constraint=50.0,
    )
    assert result.units_targeted == 0
    assert result.optimal_policy_value == 0.0


def test_error_handling() -> None:
    with pytest.raises(InputError, match="cannot be empty"):
        evaluate_decision_focused_model_value(
            candidate_predictions={},
            actual_outcomes=np.array([1, 0]),
            intervention_cost=10.0,
            intervention_payoff=50.0,
        )

    with pytest.raises(InputError, match="cannot be empty"):
        evaluate_decision_focused_model_value(
            candidate_predictions={"m1": np.array([0.5])},
            actual_outcomes=np.array([]),
            intervention_cost=10.0,
            intervention_payoff=50.0,
        )

    with pytest.raises(InputError, match="does not match actual_outcomes length"):
        evaluate_decision_focused_model_value(
            candidate_predictions={"m1": np.array([0.5, 0.8])},
            actual_outcomes=np.array([1]),
            intervention_cost=10.0,
            intervention_payoff=50.0,
        )

    with pytest.raises(InputError, match="2D array"):
        compute_policy_uplift_voi(
            cate_samples=np.array([0.1, 0.2]),
            intervention_cost=10.0,
            payoff_multiplier=100.0,
        )

    with pytest.raises(InputError, match="zero dimensions"):
        compute_policy_uplift_voi(
            cate_samples=np.empty((0, 0)),
            intervention_cost=10.0,
            payoff_multiplier=100.0,
        )

    non_existent = Path("specs/ml-voi/schemas/v1/missing.schema.json")
    with pytest.raises(InputError, match="not found"):
        validate_decision_focused_model_value({}, schema_path=non_existent)
    with pytest.raises(InputError, match="not found"):
        validate_policy_uplift_voi({}, schema_path=non_existent)
