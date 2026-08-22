"""Tests for customer churn retention worked example (#574)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples.customer_churn_retention_example import (
    build_decision_problem,
    generate_churn_dataset,
    main,
    run_churn_retention_analysis,
)
from voiage.schema import DecisionProblem

if TYPE_CHECKING:
    import pytest


def test_generate_churn_dataset_deterministic_dimensions() -> None:
    net_benefits, params = generate_churn_dataset(n_samples=500, seed=123)
    assert net_benefits.shape == (500, 3)
    assert len(params) == 6
    for arr in params.values():
        assert len(arr) == 500
        assert not any(float(x) != float(x) for x in arr)  # no NaNs


def test_build_decision_problem_conforms_to_schema() -> None:
    problem = build_decision_problem()
    assert isinstance(problem, DecisionProblem)
    assert problem.decision_problem_id == "churn_retention_campaign_2026"
    assert len(problem.interventions) == 3
    assert problem.reference_intervention is not None
    assert problem.reference_intervention.intervention_id == "status_quo"


def test_run_churn_retention_analysis_metrics() -> None:
    results = run_churn_retention_analysis(n_samples=1000, seed=42)

    assert "decision_problem" in results
    assert "expected_net_benefits" in results
    assert len(results["expected_net_benefits"]) == 3
    assert results["evpi_per_account"] > 0.0
    assert results["evppi_effectiveness_per_account"] >= 0.0
    assert results["evppi_clv_per_account"] >= 0.0
    assert results["population_evsi"] > 0.0
    assert results["enbs_pilot_trial"] > 0.0


def test_customer_churn_main_executes(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "Customer Churn Retention Decision" in captured.out
    assert "EVPI per Account" in captured.out
    assert "ENBS of Pilot Trial" in captured.out
