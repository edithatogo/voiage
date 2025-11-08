# tests/test_observational.py

"""Tests for observational study VOI methods."""

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import InputError
from voiage.methods.observational import voi_observational
from voiage.schema import ParameterSet, ValueArray


# Mock observational study modeler for testing
def mock_obs_modeler(psa_samples, study_design, bias_models):
    """Mock observational study modeler that returns simple net benefits."""
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
def sample_study_design():
    """Create a sample observational study design for testing."""
    return {
        "study_type": "cohort",
        "sample_size": 1000,
        "variables_collected": ["treatment", "outcome"]
    }


@pytest.fixture
def sample_bias_models():
    """Create sample bias models for testing."""
    return {
        "confounding": {"strength": 0.2},
        "selection_bias": {"probability": 0.1}
    }


def test_voi_observational_basic(sample_psa, sample_study_design, sample_bias_models):
    """Test basic functionality of voi_observational."""
    result = voi_observational(
        obs_study_modeler=mock_obs_modeler,
        psa_prior=sample_psa,
        observational_study_design=sample_study_design,
        bias_models=sample_bias_models,
        n_outer_loops=5
    )

    assert isinstance(result, float)
    assert result >= 0  # VOI should be non-negative


def test_voi_observational_with_population_scaling(sample_psa, sample_study_design, sample_bias_models):
    """Test voi_observational with population scaling."""
    result = voi_observational(
        obs_study_modeler=mock_obs_modeler,
        psa_prior=sample_psa,
        observational_study_design=sample_study_design,
        bias_models=sample_bias_models,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=5
    )

    assert isinstance(result, float)
    assert result >= 0


def test_voi_observational_input_validation(sample_psa, sample_study_design, sample_bias_models):
    """Test input validation for voi_observational."""
    # Test invalid obs_study_modeler
    with pytest.raises(InputError, match="`obs_study_modeler` must be a callable function"):
        voi_observational(
            obs_study_modeler="not_a_function",
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models
        )

    # Test invalid psa_prior
    with pytest.raises(InputError, match="`psa_prior` must be a PSASample object"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior="not_a_psa",
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models
        )

    # Test invalid observational_study_design
    with pytest.raises(InputError, match="`observational_study_design` must be a dictionary"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design="not_a_dict",
            bias_models=sample_bias_models
        )

    # Test invalid bias_models
    with pytest.raises(InputError, match="`bias_models` must be a dictionary"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models="not_a_dict"
        )

    # Test invalid loop parameters
    with pytest.raises(InputError, match="n_outer_loops must be positive"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            n_outer_loops=0
        )


def test_voi_observational_population_scaling_validation(sample_psa, sample_study_design, sample_bias_models):
    """Test population scaling validation in voi_observational."""
    # Test invalid population
    with pytest.raises(InputError, match="Population must be positive"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            population=0,
            time_horizon=10
        )

    # Test invalid time_horizon
    with pytest.raises(InputError, match="Time horizon must be positive"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            population=1000,
            time_horizon=0
        )

    # Test invalid discount_rate
    with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            population=1000,
            time_horizon=10,
            discount_rate=1.5
        )


def test_voi_observational_edge_cases(sample_psa, sample_study_design, sample_bias_models):
    """Test edge cases for voi_observational."""
    # Test with very small number of loops
    result = voi_observational(
        obs_study_modeler=mock_obs_modeler,
        psa_prior=sample_psa,
        observational_study_design=sample_study_design,
        bias_models=sample_bias_models,
        n_outer_loops=1
    )

    assert isinstance(result, float)
    assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__])
