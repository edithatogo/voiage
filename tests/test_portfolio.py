# tests/test_portfolio.py

"""Unit tests for the portfolio VOI methods in voiage.methods.portfolio."""

import pytest
import numpy as np

from voiage.methods.portfolio import portfolio_voi
from voiage.exceptions import VoiageNotImplementedError
from voiage.schema import PortfolioSpec, PortfolioStudy, DecisionOption, TrialDesign


def dummy_calculator(study: PortfolioStudy) -> float:
    if "Alpha" in study.name:
        return 100.0
    if "Beta" in study.name:
        return 150.0
    if "Gamma" in study.name:
        return 80.0
    if "Delta" in study.name:
        return 120.0
    return 0.0


@pytest.fixture
def studies():
    study_a_design = TrialDesign([DecisionOption("T1", 10)])
    study_b_design = TrialDesign([DecisionOption("T2", 20)])
    study_c_design = TrialDesign([DecisionOption("T3", 15)])
    study_d_design = TrialDesign([DecisionOption("T4", 30)])

    return [
        PortfolioStudy(name="Study Alpha", design=study_a_design, cost=50),
        PortfolioStudy(name="Study Beta", design=study_b_design, cost=60),
        PortfolioStudy(name="Study Gamma", design=study_c_design, cost=40),
        PortfolioStudy(name="Study Delta", design=study_d_design, cost=100),
    ]


def test_portfolio_voi_greedy_no_budget(studies):
    portfolio_spec_no_budget = PortfolioSpec(studies=studies, budget_constraint=None)
    result_no_budget = portfolio_voi(
        portfolio_spec_no_budget, dummy_calculator, "greedy"
    )
    assert len(result_no_budget["selected_studies"]) == 4
    assert np.isclose(result_no_budget["total_value"], 450.0)
    assert np.isclose(result_no_budget["total_cost"], 250.0)


def test_portfolio_voi_greedy_with_budget(studies):
    portfolio_spec_budget_100 = PortfolioSpec(studies=studies, budget_constraint=100)
    result_budget_100 = portfolio_voi(
        portfolio_spec_budget_100, dummy_calculator, "greedy"
    )
    selected_names_b100 = sorted(
        [s.name for s in result_budget_100["selected_studies"]]
    )
    assert selected_names_b100 == sorted(["Study Beta", "Study Gamma"])
    assert np.isclose(result_budget_100["total_value"], 230.0)
    assert np.isclose(result_budget_100["total_cost"], 100.0)


def test_portfolio_voi_greedy_budget_allows_only_highest_ratio(studies):
    portfolio_spec_budget_70 = PortfolioSpec(studies=studies, budget_constraint=70)
    result_budget_70 = portfolio_voi(
        portfolio_spec_budget_70, dummy_calculator, "greedy"
    )
    selected_names_b70 = sorted([s.name for s in result_budget_70["selected_studies"]])
    assert selected_names_b70 == ["Study Beta"]
    assert np.isclose(result_budget_70["total_value"], 150.0)
    assert np.isclose(result_budget_70["total_cost"], 60.0)


def test_portfolio_voi_not_implemented_methods(studies):
    portfolio_spec_no_budget = PortfolioSpec(studies=studies, budget_constraint=None)
    with pytest.raises(VoiageNotImplementedError):
        portfolio_voi(portfolio_spec_no_budget, dummy_calculator, "dynamic_programming")
