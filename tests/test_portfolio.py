# tests/test_portfolio.py

"""Tests for portfolio optimization methods."""

import pytest

from voiage.exceptions import InputError, VoiageNotImplementedError
from voiage.methods.portfolio import portfolio_voi
from voiage.schema import DecisionOption, PortfolioSpec, PortfolioStudy, TrialDesign


# Mock study value calculator for testing
def mock_study_value_calculator(study: PortfolioStudy) -> float:
    """Mock calculator that returns a fixed value based on study name."""
    if "High" in study.name:
        return 100.0
    elif "Medium" in study.name:
        return 50.0
    elif "Low" in study.name:
        return 25.0
    return 0.0


@pytest.fixture()
def sample_studies():
    """Create sample PortfolioStudy objects for testing."""
    # Create trial designs
    design_high = TrialDesign([DecisionOption("Treatment A", 100)])
    design_medium = TrialDesign([DecisionOption("Treatment B", 50)])
    design_low = TrialDesign([DecisionOption("Treatment C", 25)])

    # Create studies
    study_high = PortfolioStudy(name="High Value Study", design=design_high, cost=50.0)
    study_medium = PortfolioStudy(name="Medium Value Study", design=design_medium, cost=30.0)
    study_low = PortfolioStudy(name="Low Value Study", design=design_low, cost=10.0)

    return [study_high, study_medium, study_low]


@pytest.fixture()
def sample_portfolio_spec(sample_studies):
    """Create a sample PortfolioSpec for testing."""
    return PortfolioSpec(studies=sample_studies, budget_constraint=80.0)


def test_portfolio_voi_basic(sample_portfolio_spec):
    """Test basic functionality of portfolio_voi."""
    result = portfolio_voi(
        portfolio_specification=sample_portfolio_spec,
        study_value_calculator=mock_study_value_calculator,
        optimization_method="greedy"
    )

    assert isinstance(result, dict)
    assert "selected_studies" in result
    assert "total_value" in result
    assert "total_cost" in result
    assert "method_details" in result

    # Check types
    assert isinstance(result["selected_studies"], list)
    assert isinstance(result["total_value"], float)
    assert isinstance(result["total_cost"], float)
    assert isinstance(result["method_details"], str)


def test_portfolio_voi_greedy_no_budget(sample_studies):
    """Test greedy optimization with no budget constraint."""
    portfolio_spec = PortfolioSpec(studies=sample_studies, budget_constraint=None)

    result = portfolio_voi(
        portfolio_specification=portfolio_spec,
        study_value_calculator=mock_study_value_calculator,
        optimization_method="greedy"
    )

    # With no budget, all studies should be selected
    assert len(result["selected_studies"]) == 3
    assert result["total_cost"] == 90.0  # 50 + 30 + 10
    assert result["total_value"] == 175.0  # 100 + 50 + 25


def test_portfolio_voi_greedy_with_budget(sample_studies):
    """Test greedy optimization with budget constraint."""
    # Budget of 60 should select High Value (50 cost) and Low Value (10 cost) = 60 total
    # But not Medium Value (30 cost) as High+Medium = 80 > 60
    # However, greedy by value/cost ratio:
    # High: 100/50 = 2.0
    # Medium: 50/30 = 1.67
    # Low: 25/10 = 2.5
    # So order is Low, High, Medium
    # Select Low (10 cost, 25 value) -> remaining 50
    # Select High (50 cost, 100 value) -> remaining 0
    # Total: 60 cost, 125 value

    portfolio_spec = PortfolioSpec(studies=sample_studies, budget_constraint=60.0)

    result = portfolio_voi(
        portfolio_specification=portfolio_spec,
        study_value_calculator=mock_study_value_calculator,
        optimization_method="greedy"
    )

    selected_names = [study.name for study in result["selected_studies"]]
    assert "Low Value Study" in selected_names
    assert "High Value Study" in selected_names
    assert "Medium Value Study" not in selected_names
    assert result["total_cost"] == 60.0
    assert result["total_value"] == 125.0


def test_portfolio_voi_integer_programming(sample_studies):
    """Test integer programming optimization."""
    portfolio_spec = PortfolioSpec(studies=sample_studies, budget_constraint=60.0)

    result = portfolio_voi(
        portfolio_specification=portfolio_spec,
        study_value_calculator=mock_study_value_calculator,
        optimization_method="integer_programming"
    )

    # Check that we get a result with the expected structure
    assert isinstance(result["selected_studies"], list)
    assert isinstance(result["total_value"], (int, float))
    assert isinstance(result["total_cost"], (int, float))


def test_portfolio_voi_single_study():
    """Test portfolio_voi with a single study."""
    design = TrialDesign([DecisionOption("Treatment", 50)])
    study = PortfolioStudy(name="Single Study", design=design, cost=25.0)
    portfolio_spec = PortfolioSpec(studies=[study], budget_constraint=None)

    result = portfolio_voi(
        portfolio_specification=portfolio_spec,
        study_value_calculator=mock_study_value_calculator,
        optimization_method="greedy"
    )

    assert len(result["selected_studies"]) == 1
    assert result["selected_studies"][0].name == "Single Study"
    assert result["total_value"] == 0.0  # Default value from mock calculator
    assert result["total_cost"] == 25.0


def test_portfolio_voi_input_validation():
    """Test input validation for portfolio_voi."""
    # Test invalid portfolio_specification
    with pytest.raises(InputError, match="`portfolio_specification` must be a PortfolioSpec object"):
        portfolio_voi(
            portfolio_specification="not a portfolio spec",
            study_value_calculator=mock_study_value_calculator
        )

    # Test invalid study_value_calculator
    design = TrialDesign([DecisionOption("Treatment", 50)])
    study = PortfolioStudy(name="Test Study", design=design, cost=25.0)
    portfolio_spec = PortfolioSpec(studies=[study], budget_constraint=None)

    with pytest.raises(InputError, match="`study_value_calculator` must be a callable function"):
        portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator="not a function"
        )


def test_portfolio_voi_unimplemented_methods():
    """Test that unimplemented methods raise appropriate errors."""
    design = TrialDesign([DecisionOption("Treatment", 50)])
    study = PortfolioStudy(name="Test Study", design=design, cost=25.0)
    portfolio_spec = PortfolioSpec(studies=[study], budget_constraint=None)

    # Test dynamic programming (not implemented)
    with pytest.raises(VoiageNotImplementedError):
        portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=mock_study_value_calculator,
            optimization_method="dynamic_programming"
        )

    # Test unknown method
    with pytest.raises(VoiageNotImplementedError, match="not recognized or implemented"):
        portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=mock_study_value_calculator,
            optimization_method="unknown_method"
        )


def test_portfolio_voi_zero_cost_studies(sample_studies):
    """Test portfolio optimization with zero cost studies."""
    # Add a zero cost study
    design_zero = TrialDesign([DecisionOption("Treatment D", 10)])
    study_zero = PortfolioStudy(name="Zero Cost Study", design=design_zero, cost=0.0)

    studies_with_zero = sample_studies + [study_zero]
    portfolio_spec = PortfolioSpec(studies=studies_with_zero, budget_constraint=50.0)

    result = portfolio_voi(
        portfolio_specification=portfolio_spec,
        study_value_calculator=mock_study_value_calculator,
        optimization_method="greedy"
    )

    # Zero cost study should be selected if it has positive value
    selected_names = [study.name for study in result["selected_studies"]]
    assert "Zero Cost Study" in selected_names


if __name__ == "__main__":
    pytest.main([__file__])
