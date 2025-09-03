# tests/test_adaptive.py

"""Tests for adaptive trial VOI methods."""

import numpy as np
import pytest
import xarray as xr

from voiage.methods.adaptive import adaptive_evsi
from voiage.schema import ValueArray, ParameterSet, TrialDesign, DecisionOption
from voiage.exceptions import InputError


# Mock adaptive trial simulator for testing
def mock_adaptive_simulator(psa_samples, base_design, adaptive_rules):
    """Mock adaptive trial simulator that returns simple net benefits."""
    # Create simple net benefits based on PSA samples
    n_samples = psa_samples.n_samples
    # Create net benefits for 2 strategies
    nb_values = np.random.rand(n_samples, 2) * 1000
    # Make strategy 1 slightly better on average
    nb_values[:, 1] += 100
    
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
    """Create a sample ParameterSet for testing."""
    params = {
        "effectiveness": np.random.normal(0.7, 0.1, 50),
        "cost": np.random.normal(5000, 500, 50)
    }
    return ParameterSet.from_numpy_or_dict(params)


@pytest.fixture
def sample_trial_design():
    """Create a sample TrialDesign for testing."""
    arms = [
        DecisionOption(name="Treatment A", sample_size=100),
        DecisionOption(name="Treatment B", sample_size=100)
    ]
    return TrialDesign(arms=arms)


@pytest.fixture
def sample_adaptive_rules():
    """Create sample adaptive rules for testing."""
    return {
        "interim_analysis_points": [0.5],
        "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1}
    }


def test_adaptive_evsi_basic(sample_psa, sample_trial_design, sample_adaptive_rules):
    """Test basic functionality of adaptive_evsi."""
    result = adaptive_evsi(
        adaptive_trial_simulator=mock_adaptive_simulator,
        psa_prior=sample_psa,
        base_trial_design=sample_trial_design,
        adaptive_rules=sample_adaptive_rules,
        n_outer_loops=3,
        n_inner_loops=5
    )
    
    assert isinstance(result, float)
    assert result >= 0  # EVSI should be non-negative


def test_adaptive_evsi_with_population_scaling(sample_psa, sample_trial_design, sample_adaptive_rules):
    """Test adaptive_evsi with population scaling."""
    result = adaptive_evsi(
        adaptive_trial_simulator=mock_adaptive_simulator,
        psa_prior=sample_psa,
        base_trial_design=sample_trial_design,
        adaptive_rules=sample_adaptive_rules,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=3,
        n_inner_loops=5
    )
    
    assert isinstance(result, float)
    assert result >= 0


def test_adaptive_evsi_input_validation(sample_psa, sample_trial_design, sample_adaptive_rules):
    """Test input validation for adaptive_evsi."""
    # Test invalid adaptive_trial_simulator
    with pytest.raises(InputError, match="`adaptive_trial_simulator` must be a callable function"):
        adaptive_evsi(
            adaptive_trial_simulator="not_a_function",
            psa_prior=sample_psa,
            base_trial_design=sample_trial_design,
            adaptive_rules=sample_adaptive_rules
        )
    
    # Test invalid psa_prior
    with pytest.raises(InputError, match="`psa_prior` must be a PSASample object"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior="not_a_psa",
            base_trial_design=sample_trial_design,
            adaptive_rules=sample_adaptive_rules
        )
    
    # Test invalid base_trial_design
    with pytest.raises(InputError, match="`base_trial_design` must be a TrialDesign object"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior=sample_psa,
            base_trial_design="not_a_trial_design",
            adaptive_rules=sample_adaptive_rules
        )
    
    # Test invalid adaptive_rules
    with pytest.raises(InputError, match="`adaptive_rules` must be a dictionary"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior=sample_psa,
            base_trial_design=sample_trial_design,
            adaptive_rules="not_a_dict"
        )
    
    # Test invalid loop parameters
    with pytest.raises(InputError, match="n_outer_loops and n_inner_loops must be positive"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior=sample_psa,
            base_trial_design=sample_trial_design,
            adaptive_rules=sample_adaptive_rules,
            n_outer_loops=0
        )
    
    with pytest.raises(InputError, match="n_outer_loops and n_inner_loops must be positive"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior=sample_psa,
            base_trial_design=sample_trial_design,
            adaptive_rules=sample_adaptive_rules,
            n_inner_loops=0
        )


def test_adaptive_evsi_population_scaling_validation(sample_psa, sample_trial_design, sample_adaptive_rules):
    """Test population scaling validation in adaptive_evsi."""
    # Test invalid population
    with pytest.raises(InputError, match="Population must be positive"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior=sample_psa,
            base_trial_design=sample_trial_design,
            adaptive_rules=sample_adaptive_rules,
            population=0,
            time_horizon=10
        )
    
    # Test invalid time_horizon
    with pytest.raises(InputError, match="Time horizon must be positive"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior=sample_psa,
            base_trial_design=sample_trial_design,
            adaptive_rules=sample_adaptive_rules,
            population=1000,
            time_horizon=0
        )
    
    # Test invalid discount_rate
    with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
        adaptive_evsi(
            adaptive_trial_simulator=mock_adaptive_simulator,
            psa_prior=sample_psa,
            base_trial_design=sample_trial_design,
            adaptive_rules=sample_adaptive_rules,
            population=1000,
            time_horizon=10,
            discount_rate=1.5
        )


def test_adaptive_evsi_edge_cases(sample_psa, sample_trial_design, sample_adaptive_rules):
    """Test edge cases for adaptive_evsi."""
    # Test with very small number of loops
    result = adaptive_evsi(
        adaptive_trial_simulator=mock_adaptive_simulator,
        psa_prior=sample_psa,
        base_trial_design=sample_trial_design,
        adaptive_rules=sample_adaptive_rules,
        n_outer_loops=1,
        n_inner_loops=1
    )
    
    assert isinstance(result, float)
    assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__])