# tests/test_calibration.py

"""Tests for calibration VOI methods."""

import numpy as np
import pytest
import xarray as xr

from voiage.methods.calibration import voi_calibration
from voiage.schema import ValueArray, ParameterSet
from voiage.exceptions import InputError


# Mock calibration study modeler for testing
def mock_cal_modeler(psa_samples, study_design, process_spec):
    """Mock calibration study modeler that returns simple net benefits."""
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
    """Create a sample calibration study design for testing."""
    return {
        "experiment_type": "lab",
        "sample_size": 100,
        "variables_measured": ["parameter_a", "parameter_b"]
    }


@pytest.fixture
def sample_process_spec():
    """Create a sample calibration process specification for testing."""
    return {
        "method": "bayesian",
        "likelihood_function": "normal"
    }


def test_voi_calibration_basic(sample_psa, sample_study_design, sample_process_spec):
    """Test basic functionality of voi_calibration."""
    result = voi_calibration(
        cal_study_modeler=mock_cal_modeler,
        psa_prior=sample_psa,
        calibration_study_design=sample_study_design,
        calibration_process_spec=sample_process_spec,
        n_outer_loops=5
    )
    
    assert isinstance(result, float)
    assert result >= 0  # VOI should be non-negative


def test_voi_calibration_with_population_scaling(sample_psa, sample_study_design, sample_process_spec):
    """Test voi_calibration with population scaling."""
    result = voi_calibration(
        cal_study_modeler=mock_cal_modeler,
        psa_prior=sample_psa,
        calibration_study_design=sample_study_design,
        calibration_process_spec=sample_process_spec,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=5
    )
    
    assert isinstance(result, float)
    assert result >= 0


def test_voi_calibration_input_validation(sample_psa, sample_study_design, sample_process_spec):
    """Test input validation for voi_calibration."""
    # Test invalid cal_study_modeler
    with pytest.raises(InputError, match="`cal_study_modeler` must be a callable function"):
        voi_calibration(
            cal_study_modeler="not_a_function",
            psa_prior=sample_psa,
            calibration_study_design=sample_study_design,
            calibration_process_spec=sample_process_spec
        )
    
    # Test invalid psa_prior
    with pytest.raises(InputError, match="`psa_prior` must be a PSASample object"):
        voi_calibration(
            cal_study_modeler=mock_cal_modeler,
            psa_prior="not_a_psa",
            calibration_study_design=sample_study_design,
            calibration_process_spec=sample_process_spec
        )
    
    # Test invalid calibration_study_design
    with pytest.raises(InputError, match="`calibration_study_design` must be a dictionary"):
        voi_calibration(
            cal_study_modeler=mock_cal_modeler,
            psa_prior=sample_psa,
            calibration_study_design="not_a_dict",
            calibration_process_spec=sample_process_spec
        )
    
    # Test invalid calibration_process_spec
    with pytest.raises(InputError, match="`calibration_process_spec` must be a dictionary"):
        voi_calibration(
            cal_study_modeler=mock_cal_modeler,
            psa_prior=sample_psa,
            calibration_study_design=sample_study_design,
            calibration_process_spec="not_a_dict"
        )
    
    # Test invalid loop parameters
    with pytest.raises(InputError, match="n_outer_loops must be positive"):
        voi_calibration(
            cal_study_modeler=mock_cal_modeler,
            psa_prior=sample_psa,
            calibration_study_design=sample_study_design,
            calibration_process_spec=sample_process_spec,
            n_outer_loops=0
        )


def test_voi_calibration_population_scaling_validation(sample_psa, sample_study_design, sample_process_spec):
    """Test population scaling validation in voi_calibration."""
    # Test invalid population
    with pytest.raises(InputError, match="Population must be positive"):
        voi_calibration(
            cal_study_modeler=mock_cal_modeler,
            psa_prior=sample_psa,
            calibration_study_design=sample_study_design,
            calibration_process_spec=sample_process_spec,
            population=0,
            time_horizon=10
        )
    
    # Test invalid time_horizon
    with pytest.raises(InputError, match="Time horizon must be positive"):
        voi_calibration(
            cal_study_modeler=mock_cal_modeler,
            psa_prior=sample_psa,
            calibration_study_design=sample_study_design,
            calibration_process_spec=sample_process_spec,
            population=1000,
            time_horizon=0
        )
    
    # Test invalid discount_rate
    with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
        voi_calibration(
            cal_study_modeler=mock_cal_modeler,
            psa_prior=sample_psa,
            calibration_study_design=sample_study_design,
            calibration_process_spec=sample_process_spec,
            population=1000,
            time_horizon=10,
            discount_rate=1.5
        )


def test_voi_calibration_edge_cases(sample_psa, sample_study_design, sample_process_spec):
    """Test edge cases for voi_calibration."""
    # Test with very small number of loops
    result = voi_calibration(
        cal_study_modeler=mock_cal_modeler,
        psa_prior=sample_psa,
        calibration_study_design=sample_study_design,
        calibration_process_spec=sample_process_spec,
        n_outer_loops=1
    )
    
    assert isinstance(result, float)
    assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__])