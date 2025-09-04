"""Tests for healthcare utilities."""

import numpy as np
import pytest

from voiage.healthcare.utilities import (
    calculate_qaly,
    discount_qaly,
    aggregate_qaly_over_time,
    markov_cohort_model,
    disease_progression_model
)


def test_calculate_qaly():
    """Test QALY calculation."""
    # Simple case: 3 years with constant utility
    utilities = [0.8, 0.8, 0.8]
    time_periods = [1, 1, 1]
    qaly = calculate_qaly(utilities, time_periods, 0.03)
    
    # Expected: 0.8 * 1 / (1.03^0.5) + 0.8 * 1 / (1.03^1.5) + 0.8 * 1 / (1.03^2.5)
    expected_qaly = (
        0.8 / (1.03 ** 0.5) +
        0.8 / (1.03 ** 1.5) +
        0.8 / (1.03 ** 2.5)
    )
    assert abs(qaly - expected_qaly) < 1e-10


def test_calculate_qaly_variable_utilities():
    """Test QALY calculation with variable utilities."""
    # Declining utility over time (common in progressive diseases)
    utilities = [0.9, 0.7, 0.5, 0.3]
    time_periods = [1, 1, 1, 1]
    qaly = calculate_qaly(utilities, time_periods, 0.03)
    
    expected_qaly = (
        0.9 / (1.03 ** 0.5) +
        0.7 / (1.03 ** 1.5) +
        0.5 / (1.03 ** 2.5) +
        0.3 / (1.03 ** 3.5)
    )
    assert abs(qaly - expected_qaly) < 1e-10


def test_calculate_qaly_invalid_inputs():
    """Test QALY calculation with invalid inputs."""
    # Mismatched array lengths
    with pytest.raises(ValueError):
        calculate_qaly([0.8, 0.7], [1, 1, 1], 0.03)
    
    # Invalid utility values
    with pytest.raises(ValueError):
        calculate_qaly([1.5, 0.7], [1, 1], 0.03)
    
    with pytest.raises(ValueError):
        calculate_qaly([-0.1, 0.7], [1, 1], 0.03)
    
    # Negative discount rate
    with pytest.raises(ValueError):
        calculate_qaly([0.8, 0.7], [1, 1], -0.01)


def test_discount_qaly():
    """Test QALY discounting."""
    discounted = discount_qaly(10.0, 5, 0.03)
    expected = 10.0 / (1.03 ** 5)
    assert abs(discounted - expected) < 1e-10


def test_aggregate_qaly_over_time():
    """Test aggregation of QALYs over time for multiple strategies."""
    utility_trajectories = {
        "standard_care": np.array([0.8, 0.75, 0.7, 0.65, 1.0]),  # Death state = 1.0 utility
        "new_treatment": np.array([0.85, 0.82, 0.8, 0.78, 1.0])
    }
    time_points = np.array([0, 1, 2, 3, 4])
    
    results = aggregate_qaly_over_time(utility_trajectories, time_points, 0.03)
    
    assert "standard_care" in results
    assert "new_treatment" in results
    assert results["new_treatment"] > results["standard_care"]


def test_markov_cohort_model():
    """Test Markov cohort model simulation."""
    # Simple 3-state model: Healthy, Sick, Dead
    transition_matrix = np.array([
        [0.9, 0.05, 0.05],  # Healthy -> Healthy, Sick, Dead
        [0.1, 0.8, 0.1],    # Sick -> Healthy, Sick, Dead
        [0.0, 0.0, 1.0]     # Dead -> Dead (absorbing state)
    ])
    
    initial_state = np.array([1.0, 0.0, 0.0])  # All start healthy
    
    trajectories = markov_cohort_model(transition_matrix, initial_state, 10)
    
    # Should have 11 time points (0-10)
    assert trajectories.shape == (11, 3)
    
    # Initial state should be correct
    assert np.allclose(trajectories[0], [1.0, 0.0, 0.0])
    
    # All rows should sum to 1 (probability distribution)
    for row in trajectories:
        assert abs(np.sum(row) - 1.0) < 1e-10


def test_disease_progression_model():
    """Test disease progression model."""
    # Simple model with 3 health states
    base_transitions = {
        "healthy": {"healthy": 0.9, "sick": 0.1},
        "sick": {"healthy": 0.05, "sick": 0.8, "dead": 0.15},
        "dead": {"dead": 1.0}
    }
    
    transition_matrix = disease_progression_model(base_transitions)
    
    # Should be 3x3 matrix
    assert transition_matrix.shape == (3, 3)
    
    # Rows should sum to 1
    for i in range(3):
        assert abs(np.sum(transition_matrix[i]) - 1.0) < 1e-10


def test_disease_progression_model_with_covariates():
    """Test disease progression model with covariates."""
    base_transitions = {
        "healthy": {"healthy": 0.9, "sick": 0.1},
        "sick": {"healthy": 0.05, "sick": 0.8, "dead": 0.15},
        "dead": {"dead": 1.0}
    }
    
    covariates = {"treatment": 1.0}
    covariate_effects = {
        "treatment": {
            ("sick", "healthy"): 0.5,  # Treatment increases recovery rate
            ("healthy", "sick"): -0.3   # Treatment decreases sickness rate
        }
    }
    
    transition_matrix = disease_progression_model(
        base_transitions, covariates, covariate_effects
    )
    
    # Should still be valid transition matrix
    assert transition_matrix.shape == (3, 3)
    for i in range(3):
        assert abs(np.sum(transition_matrix[i]) - 1.0) < 1e-10


if __name__ == "__main__":
    test_calculate_qaly()
    test_calculate_qaly_variable_utilities()
    test_calculate_qaly_invalid_inputs()
    test_discount_qaly()
    test_aggregate_qaly_over_time()
    test_markov_cohort_model()
    test_disease_progression_model()
    test_disease_progression_model_with_covariates()
    print("All healthcare utilities tests passed!")