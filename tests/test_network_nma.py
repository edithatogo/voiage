# tests/test_network_nma.py

"""Tests for Network Meta-Analysis VOI methods."""

import numpy as np
import pytest
import xarray as xr

from voiage.methods.network_nma import (
    evsi_nma,
    _simulate_trial_data_nma,
    _update_nma_posterior,
    _perform_network_meta_analysis,
    calculate_nma_consistency,
    simulate_nma_network_data
)
from voiage.schema import ValueArray, ParameterSet, TrialDesign, DecisionOption
from voiage.exceptions import InputError


# Mock NMA model evaluator for testing
def mock_nma_evaluator(psa_samples, trial_design=None, trial_data=None):
    """Mock NMA evaluator that returns simple net benefits."""
    # Create simple net benefits based on PSA samples
    n_samples = psa_samples.n_samples
    # Create net benefits for 2 strategies
    nb_values = np.random.rand(n_samples, 2) * 100
    # Make strategy 1 slightly better on average
    nb_values[:, 1] += 10
    
    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
        coords={
            "n_samples": np.arange(n_samples),
            "n_strategies": np.arange(2),
            "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
        }
    )
    return ValueArray(dataset=dataset)


@pytest.fixture
def sample_psa():
    """Create a sample ParameterSet for NMA testing."""
    # Create parameters relevant to NMA
    params = {
        "te_treatment_a": np.random.normal(0.1, 0.05, 100),
        "te_treatment_b": np.random.normal(0.3, 0.05, 100),
        "te_treatment_c": np.random.normal(0.2, 0.05, 100),
        "baseline_outcome": np.random.normal(0.5, 0.1, 100),
        "outcome_sd": np.random.uniform(0.1, 0.3, 100)
    }
    
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params.items()},
        coords={"n_samples": np.arange(100)}
    )
    return ParameterSet(dataset=dataset)


@pytest.fixture
def sample_trial_design():
    """Create a sample TrialDesign for NMA testing."""
    arms = [
        DecisionOption(name="Treatment A", sample_size=50),
        DecisionOption(name="Treatment B", sample_size=50),
        DecisionOption(name="Treatment C", sample_size=50)
    ]
    return TrialDesign(arms=arms)


def test_evsi_nma_basic(sample_psa, sample_trial_design):
    """Test basic functionality of evsi_nma."""
    result = evsi_nma(
        nma_model_evaluator=mock_nma_evaluator,
        psa_prior_nma=sample_psa,
        trial_design_new_study=sample_trial_design,
        n_outer_loops=5,  # Use smaller numbers for faster tests
        n_inner_loops=10
    )
    
    assert isinstance(result, float)
    assert result >= 0  # EVSI should be non-negative


def test_evsi_nma_with_population_scaling(sample_psa, sample_trial_design):
    """Test evsi_nma with population scaling."""
    result = evsi_nma(
        nma_model_evaluator=mock_nma_evaluator,
        psa_prior_nma=sample_psa,
        trial_design_new_study=sample_trial_design,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=5,
        n_inner_loops=10
    )
    
    assert isinstance(result, float)
    assert result >= 0


def test_evsi_nma_input_validation(sample_psa, sample_trial_design):
    """Test input validation for evsi_nma."""
    # Test invalid nma_model_evaluator
    with pytest.raises(InputError, match="`nma_model_evaluator` must be a callable function"):
        evsi_nma(
            nma_model_evaluator="not_a_function",
            psa_prior_nma=sample_psa,
            trial_design_new_study=sample_trial_design
        )
    
    # Test invalid psa_prior_nma
    with pytest.raises(InputError, match="`psa_prior_nma` must be a PSASample object"):
        evsi_nma(
            nma_model_evaluator=mock_nma_evaluator,
            psa_prior_nma="not_a_psa",
            trial_design_new_study=sample_trial_design
        )
    
    # Test invalid trial_design_new_study
    with pytest.raises(InputError, match="`trial_design_new_study` must be a TrialDesign object"):
        evsi_nma(
            nma_model_evaluator=mock_nma_evaluator,
            psa_prior_nma=sample_psa,
            trial_design_new_study="not_a_trial_design"
        )
    
    # Test invalid loop parameters
    with pytest.raises(InputError, match="n_outer_loops and n_inner_loops must be positive"):
        evsi_nma(
            nma_model_evaluator=mock_nma_evaluator,
            psa_prior_nma=sample_psa,
            trial_design_new_study=sample_trial_design,
            n_outer_loops=0
        )
    
    with pytest.raises(InputError, match="n_outer_loops and n_inner_loops must be positive"):
        evsi_nma(
            nma_model_evaluator=mock_nma_evaluator,
            psa_prior_nma=sample_psa,
            trial_design_new_study=sample_trial_design,
            n_inner_loops=0
        )


def test_evsi_nma_population_scaling_validation(sample_psa, sample_trial_design):
    """Test population scaling validation in evsi_nma."""
    # Test invalid population
    with pytest.raises(InputError, match="Population must be positive"):
        evsi_nma(
            nma_model_evaluator=mock_nma_evaluator,
            psa_prior_nma=sample_psa,
            trial_design_new_study=sample_trial_design,
            population=0,
            time_horizon=10
        )
    
    # Test invalid time_horizon
    with pytest.raises(InputError, match="Time horizon must be positive"):
        evsi_nma(
            nma_model_evaluator=mock_nma_evaluator,
            psa_prior_nma=sample_psa,
            trial_design_new_study=sample_trial_design,
            population=1000,
            time_horizon=0
        )
    
    # Test invalid discount_rate
    with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
        evsi_nma(
            nma_model_evaluator=mock_nma_evaluator,
            psa_prior_nma=sample_psa,
            trial_design_new_study=sample_trial_design,
            population=1000,
            time_horizon=10,
            discount_rate=1.5
        )


def test_simulate_trial_data_nma():
    """Test _simulate_trial_data_nma function."""
    true_params = {
        "te_treatment_a": 0.2,
        "te_treatment_b": 0.5,
        "baseline_outcome": 0.3,
        "outcome_sd": 0.1
    }
    
    trial_design = TrialDesign([
        DecisionOption(name="Treatment A", sample_size=50),
        DecisionOption(name="Treatment B", sample_size=50)
    ])
    
    simulated_data = _simulate_trial_data_nma(true_params, trial_design)
    
    # Check that we have data for each arm
    assert "Treatment A" in simulated_data
    assert "Treatment B" in simulated_data
    
    # Check data shapes
    assert simulated_data["Treatment A"].shape == (50,)
    assert simulated_data["Treatment B"].shape == (50,)
    
    # Check that data is numpy arrays
    assert isinstance(simulated_data["Treatment A"], np.ndarray)
    assert isinstance(simulated_data["Treatment B"], np.ndarray)


def test_update_nma_posterior(sample_psa, sample_trial_design):
    """Test _update_nma_posterior function."""
    # Create mock trial data
    trial_data = {
        "Treatment A": np.random.normal(0.5, 0.1, 50),
        "Treatment B": np.random.normal(0.7, 0.1, 50)
    }
    
    # Test that the function runs without error
    updated_psa = _update_nma_posterior(sample_psa, trial_data, sample_trial_design)
    
    # Check that we get back a PSASample
    assert isinstance(updated_psa, ParameterSet)
    
    # Check that the number of samples is preserved
    assert updated_psa.n_samples == sample_psa.n_samples


def test_perform_network_meta_analysis():
    """Test _perform_network_meta_analysis function."""
    # Create mock data
    treatment_effects = np.array([0.1, 0.2, 0.15, 0.25])
    se_effects = np.array([0.05, 0.06, 0.04, 0.07])
    study_designs = [[0, 1], [1, 2], [0, 2], [1, 3]]
    
    # Test that the function runs without error
    te, var = _perform_network_meta_analysis(treatment_effects, se_effects, study_designs)
    
    # Check return types
    assert isinstance(te, np.ndarray)
    assert isinstance(var, np.ndarray)
    
    # Check shapes
    assert te.shape == treatment_effects.shape
    assert var.shape == se_effects.shape


def test_calculate_nma_consistency():
    """Test calculate_nma_consistency function."""
    # Create mock data
    treatment_effects = np.array([0.1, 0.2, 0.15, 0.25])
    study_designs = [[0, 1], [1, 2], [0, 2], [1, 3]]
    
    # Test that the function runs without error
    consistency = calculate_nma_consistency(treatment_effects, study_designs)
    
    # Check return type
    assert isinstance(consistency, float)
    
    # Consistency should be non-negative
    assert consistency >= 0


def test_simulate_nma_network_data():
    """Test simulate_nma_network_data function."""
    # Test with small network
    te, se, designs = simulate_nma_network_data(3, 4)
    
    # Check return types
    assert isinstance(te, np.ndarray)
    assert isinstance(se, np.ndarray)
    assert isinstance(designs, list)
    
    # Check shapes
    assert len(te) == 4
    assert len(se) == 4
    assert len(designs) == 4
    
    # Check that each design compares exactly 2 treatments
    for design in designs:
        assert len(design) == 2
        assert all(isinstance(t, int) for t in design)
        assert all(0 <= t < 3 for t in design)  # 3 treatments (0, 1, 2)


if __name__ == "__main__":
    pytest.main([__file__])