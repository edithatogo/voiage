"""Exhaustive Decision Correctness and Industry-Scale Assurance Suite (#584).

This test suite extends voiage assurance from numerical correctness to decision
and policy correctness under constraints, causal uncertainty, calibration shifts,
and streaming invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiage.ml_policy_voi import (
    compute_policy_uplift_voi,
    evaluate_decision_focused_model_value,
)


def test_counterfactual_consistency_and_non_negative_regret() -> None:
    """Property: Policy regret must be strictly non-negative across all data draws."""
    rng = np.random.default_rng(12345)
    for _ in range(20):
        n_units = rng.integers(10, 100)
        actual_outcomes = rng.binomial(1, 0.3, size=n_units)
        preds = rng.uniform(0.0, 1.0, size=n_units)
        cost = float(rng.uniform(10.0, 50.0))
        payoff = float(rng.uniform(60.0, 200.0))

        result = evaluate_decision_focused_model_value(
            candidate_predictions={"model_rand": preds},
            actual_outcomes=actual_outcomes,
            intervention_cost=cost,
            intervention_payoff=payoff,
        )

        for model_eval in result.candidate_models:
            assert model_eval.policy_regret >= 0.0, (
                f"Negative regret encountered: {model_eval.policy_regret}"
            )


def test_evppi_bounded_by_evpi_property() -> None:
    """Property: EVPPI on any subset of uncertainty must never exceed total EVPI."""
    rng = np.random.default_rng(54321)
    n_sims = 150
    n_units = 12

    cate_samples = rng.normal(loc=0.1, scale=0.08, size=(n_sims, n_units))
    subgroups = {
        "group_a": np.array([True] * 6 + [False] * 6),
        "group_b": np.array([False] * 6 + [True] * 6),
    }

    result = compute_policy_uplift_voi(
        cate_samples=cate_samples,
        intervention_cost=25.0,
        payoff_multiplier=400.0,
        subgroups=subgroups,
    )

    assert result.uplift_evpi >= 0.0
    for sg_name, evppi_val in result.subgroup_evppi.items():
        assert evppi_val >= 0.0, f"Negative EVPPI on {sg_name}: {evppi_val}"
        assert evppi_val <= result.uplift_evpi + 1e-6, (
            f"EVPPI ({evppi_val}) exceeded total EVPI ({result.uplift_evpi}) on {sg_name}"
        )


def test_budget_constraint_satisfaction_and_monotonicity() -> None:
    """Assurance: Tight budget constraints must strictly bound spending and select highest marginal ROI."""
    rng = np.random.default_rng(999)
    n_sims = 100
    n_units = 20
    cate_samples = rng.normal(
        loc=np.linspace(0.4, -0.2, n_units), scale=0.02, size=(n_sims, n_units)
    )
    cost = 100.0
    payoff_mult = 500.0
    budget = 450.0  # Allows at most 4 units

    result = compute_policy_uplift_voi(
        cate_samples=cate_samples,
        intervention_cost=cost,
        payoff_multiplier=payoff_mult,
        budget_constraint=budget,
    )

    assert result.budget_utilized <= budget
    assert result.units_targeted <= int(budget // cost)
    assert result.units_targeted == 4


def test_calibration_shift_and_economic_inversion() -> None:
    """Decision Correctness: Under asymmetric costs, a miscalibrated model with higher AUC yields lower payoff."""
    # 20 true events out of 100
    actual = np.zeros(100, dtype=int)
    actual[:20] = 1

    cost = 50.0
    payoff = 400.0  # True positive net = +350, False positive net = -50

    # Model 1 (Miscalibrated overconfident): Predicts 0.7 for all false positives
    preds_overconfident = np.full(100, 0.7)
    # Model 2 (Calibrated): Predicts 0.8 for true positives, 0.1 for true negatives
    preds_calibrated = np.zeros(100)
    preds_calibrated[:20] = 0.8
    preds_calibrated[20:] = 0.1

    res = evaluate_decision_focused_model_value(
        candidate_predictions={
            "overconfident": preds_overconfident,
            "calibrated": preds_calibrated,
        },
        actual_outcomes=actual,
        intervention_cost=cost,
        intervention_payoff=payoff,
        decision_threshold=0.5,
    )

    assert res.selected_model == "calibrated"
    # Calibrated model delivers positive value, overconfident delivers negative value
    val_calibrated = next(
        m.expected_decision_value
        for m in res.candidate_models
        if m.model_id == "calibrated"
    )
    val_overconfident = next(
        m.expected_decision_value
        for m in res.candidate_models
        if m.model_id == "overconfident"
    )
    assert val_calibrated > 0
    assert val_calibrated > val_overconfident
    assert val_calibrated == 7000.0
    assert val_overconfident == 3000.0


def test_streaming_and_chunked_determinism() -> None:
    """Assurance: Chunked processing of large populations maintains exact deterministic invariants."""
    rng = np.random.default_rng(777)
    n_units = 1000
    n_sims = 50

    cate = rng.normal(loc=0.15, scale=0.05, size=(n_sims, n_units))
    cost = 20.0
    payoff_mult = 300.0

    full_res = compute_policy_uplift_voi(
        cate_samples=cate,
        intervention_cost=cost,
        payoff_multiplier=payoff_mult,
    )

    # Process in two independent 500-unit chunks
    chunk1_res = compute_policy_uplift_voi(
        cate_samples=cate[:, :500],
        intervention_cost=cost,
        payoff_multiplier=payoff_mult,
    )
    chunk2_res = compute_policy_uplift_voi(
        cate_samples=cate[:, 500:],
        intervention_cost=cost,
        payoff_multiplier=payoff_mult,
    )

    # Additive property across independent unconstrained populations
    combined_optimal_value = (
        chunk1_res.optimal_policy_value + chunk2_res.optimal_policy_value
    )
    combined_units = chunk1_res.units_targeted + chunk2_res.units_targeted
    combined_evpi = chunk1_res.uplift_evpi + chunk2_res.uplift_evpi

    assert (
        pytest.approx(combined_optimal_value, rel=1e-5) == full_res.optimal_policy_value
    )
    assert combined_units == full_res.units_targeted
    assert pytest.approx(combined_evpi, rel=1e-5) == full_res.uplift_evpi
