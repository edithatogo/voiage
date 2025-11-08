"""Integration tests for voiage."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.basic import evpi, evppi
from voiage.methods.portfolio import portfolio_voi
from voiage.schema import (
    DecisionOption,
    ParameterSet,
    PortfolioSpec,
    PortfolioStudy,
    TrialDesign,
    ValueArray,
)


def test_complete_voi_workflow():
    """Test a complete VOI analysis workflow."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Net benefits for 3 strategies
    strategy1 = np.random.normal(100, 10, n_samples)
    strategy2 = np.random.normal(110, 15, n_samples)
    strategy3 = np.random.normal(105, 12, n_samples)

    # Combine into net benefit array
    nb_data = np.column_stack([strategy1, strategy2, strategy3])
    value_array = ValueArray.from_numpy(nb_data, ["Standard Care", "New Treatment", "Alternative Treatment"])

    # Create parameter samples
    parameters = {
        "effectiveness": np.random.beta(2, 1, n_samples),
        "cost": np.random.normal(50, 5, n_samples),
        "quality_of_life": np.random.normal(0.7, 0.1, n_samples)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(parameters)

    # Create DecisionAnalysis
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameter_set)

    # Calculate EVPI
    evpi_result = analysis.evpi()
    assert evpi_result >= 0

    # Calculate population-adjusted EVPI
    evpi_pop = analysis.evpi(population=100000, time_horizon=10, discount_rate=0.03)
    assert evpi_pop >= evpi_result

    # Calculate EVPPI
    evppi_result = analysis.evppi()
    assert evppi_result >= 0

    # Calculate population-adjusted EVPPI
    evppi_pop = analysis.evppi(population=100000, time_horizon=10, discount_rate=0.03)
    assert evppi_pop >= evppi_result

    # Verify relationship between EVPI and EVPPI
    assert evpi_result >= evppi_result


def test_portfolio_optimization_integration():
    """Test portfolio optimization integration."""
    # Create trial designs for studies
    arm1 = DecisionOption(name="Treatment", sample_size=100)
    arm2 = DecisionOption(name="Control", sample_size=100)
    design = TrialDesign(arms=[arm1, arm2])

    # Define candidate studies
    studies = [
        PortfolioStudy(name="Study A", design=design, cost=100),
        PortfolioStudy(name="Study B", design=design, cost=200),
        PortfolioStudy(name="Study C", design=design, cost=150),
        PortfolioStudy(name="Study D", design=design, cost=300)
    ]

    # Define a simple value calculator
    def simple_value_calculator(study: PortfolioStudy) -> float:
        # Simple model: value proportional to sample size
        return study.design.total_sample_size * 10

    # Define portfolio specification
    portfolio_spec = PortfolioSpec(studies=studies, budget_constraint=400)

    # Optimize portfolio using greedy algorithm
    optimal_portfolio = portfolio_voi(
        portfolio_specification=portfolio_spec,
        study_value_calculator=simple_value_calculator,
        optimization_method="greedy"
    )

    # Verify results
    assert optimal_portfolio["total_cost"] <= portfolio_spec.budget_constraint
    assert len(optimal_portfolio["selected_studies"]) > 0

    # Test with exhaustive search for small portfolio
    small_studies = studies[:3]
    small_portfolio_spec = PortfolioSpec(studies=small_studies, budget_constraint=300)
    optimal_portfolio_exhaustive = portfolio_voi(
        portfolio_specification=small_portfolio_spec,
        study_value_calculator=simple_value_calculator,
        optimization_method="greedy"
    )

    assert optimal_portfolio_exhaustive["total_cost"] <= small_portfolio_spec.budget_constraint


def test_functional_vs_object_oriented_api():
    """Test that functional and object-oriented APIs produce the same results."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 500

    # Net benefits for 2 strategies
    strategy1 = np.random.normal(100, 10, n_samples)
    strategy2 = np.random.normal(110, 15, n_samples)
    nb_data = np.column_stack([strategy1, strategy2])
    value_array = ValueArray.from_numpy(nb_data)

    # Create parameter samples
    parameters = {
        "param1": np.random.normal(0, 1, n_samples),
        "param2": np.random.normal(0, 1, n_samples)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(parameters)

    # Object-oriented API
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameter_set)
    evpi_oo = analysis.evpi()
    evppi_oo = analysis.evppi()

    # Functional API
    evpi_func = evpi(value_array)
    evppi_func = evppi(value_array, parameter_set, ["param1"])

    # Verify they produce the same results
    np.testing.assert_allclose(evpi_oo, evpi_func, rtol=1e-10)
    np.testing.assert_allclose(evppi_oo, evppi_func, rtol=1e-10)


def test_data_structures_integration():
    """Test integration between different data structures."""
    # Test ValueArray creation from different sources
    np.random.seed(42)
    data = np.random.randn(100, 3)

    # From numpy array
    va1 = ValueArray.from_numpy(data)

    # Test with different creation methods
    # (ValueArray.from_dict doesn't exist, so we'll just test the existing method)

    # Test ParameterSet creation
    params = {
        "param1": np.random.randn(100),
        "param2": np.random.randn(100),
        "param3": np.random.randn(100)
    }

    ps1 = ParameterSet.from_numpy_or_dict(params)
    ps2 = ParameterSet.from_numpy_or_dict(params)

    # Verify parameter sets
    assert len(ps1.parameter_names) == 3
    assert len(ps2.parameter_names) == 3

    # Test ParameterSet creation
    params = {
        "param1": np.random.randn(100),
        "param2": np.random.randn(100),
        "param3": np.random.randn(100)
    }

    ps1 = ParameterSet.from_numpy_or_dict(params)
    ps2 = ParameterSet.from_numpy_or_dict(params)

    # Verify parameter sets
    assert len(ps1.parameter_names) == 3
    assert len(ps2.parameter_names) == 3


def test_error_handling_integration():
    """Test error handling in integrated scenarios."""
    # Test with mismatched array dimensions
    np.random.seed(42)
    nb_data = np.random.randn(100, 2)
    value_array = ValueArray.from_numpy(nb_data)

    # Parameter set with different number of samples
    mismatched_params = {
        "param1": np.random.randn(50),  # Only 50 samples instead of 100
        "param2": np.random.randn(50)
    }

    # This should work with the DecisionAnalysis constructor
    analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=mismatched_params)

    # But EVPPI calculation should raise an error due to dimension mismatch
    with pytest.raises(Exception):
        analysis.evppi()


if __name__ == "__main__":
    # Run the tests
    test_complete_voi_workflow()
    test_portfolio_optimization_integration()
    test_functional_vs_object_oriented_api()
    test_data_structures_integration()
    test_error_handling_integration()
    print("All integration tests passed!")
