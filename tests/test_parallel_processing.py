"""Tests for parallel processing utilities in Value of Information analysis."""

import numpy as np
import xarray as xr

from voiage.parallel.monte_carlo import (
    _monte_carlo_worker,
    parallel_bootstrap_sampling,
    parallel_evsi_calculation,
    parallel_monte_carlo_simulation,
)
from voiage.schema import DecisionOption, ParameterSet, TrialDesign, ValueArray


def simple_model_func(params):
    """Model simple economic scenario for testing."""
    # Extract parameters
    if hasattr(params, 'parameters'):
        mean_treatment = params.parameters.get('mean_treatment', np.array([0.0]))
        mean_control = params.parameters.get('mean_control', np.array([0.0]))
    else:
        # Handle case where params might be a dict or other structure
        mean_treatment = params.get('mean_treatment', np.array([0.0]))
        mean_control = params.get('mean_control', np.array([0.0]))

    # Ensure we're working with arrays
    if not isinstance(mean_treatment, np.ndarray):
        mean_treatment = np.array([mean_treatment])
    if not isinstance(mean_control, np.ndarray):
        mean_control = np.array([mean_control])

    # Calculate net benefits (simplified)
    # Assuming treatment is better when mean is higher
    nb_treatment = mean_treatment
    nb_control = mean_control

    # Stack to create ValueArray-compatible structure
    nb_values = np.column_stack([nb_control, nb_treatment]).astype(np.float64)
    return ValueArray.from_numpy(nb_values)


def mean_statistic(sample):
    """Calculate simple statistic for bootstrap testing."""
    return np.mean(sample)


def test_monte_carlo_worker():
    """Test the Monte Carlo worker function."""
    # Create simple parameter set
    params_data = {
        'mean_treatment': np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        'mean_control': np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        'sd_outcome': np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data['mean_treatment']))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create simple trial design
    trial_design = TrialDesign(arms=[
        DecisionOption(name="Treatment", sample_size=50),
        DecisionOption(name="Control", sample_size=50)
    ])

    # Test the worker function
    expected_max_nb, n_processed = _monte_carlo_worker(
        worker_id=0,
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=10,
        seed_offset=0
    )

    # Check that we got valid results
    assert isinstance(expected_max_nb, float)
    assert n_processed == 10
    assert expected_max_nb >= 0


def test_parallel_monte_carlo_simulation():
    """Test parallel Monte Carlo simulation."""
    # Create parameter set
    params_data = {
        'mean_treatment': np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        'mean_control': np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        'sd_outcome': np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data['mean_treatment']))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create trial design
    trial_design = TrialDesign(arms=[
        DecisionOption(name="Treatment", sample_size=50),
        DecisionOption(name="Control", sample_size=50)
    ])

    # Test parallel Monte Carlo simulation
    result = parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=20,
        n_workers=2,
        use_processes=True
    )

    # Check that we got a valid result
    assert isinstance(result, float)
    assert result >= 0


def test_parallel_evsi_calculation():
    """Test parallel EVSI calculation."""
    # Create parameter set
    params_data = {
        'mean_treatment': np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        'mean_control': np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        'sd_outcome': np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data['mean_treatment']))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create trial design
    trial_design = TrialDesign(arms=[
        DecisionOption(name="Treatment", sample_size=50),
        DecisionOption(name="Control", sample_size=50)
    ])

    # Test parallel EVSI calculation
    evsi_result = parallel_evsi_calculation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        population=1000,
        discount_rate=0.03,
        time_horizon=10,
        n_simulations=20,
        n_workers=2,
        use_processes=True
    )

    # Check that we got a valid result
    assert isinstance(evsi_result, float)
    assert evsi_result >= 0


def test_parallel_bootstrap_sampling():
    """Test parallel bootstrap sampling."""
    # Create test data
    data = np.random.normal(10, 2, 100).astype(np.float64)

    # Test parallel bootstrap sampling
    result = parallel_bootstrap_sampling(
        data=data,
        statistic_func=mean_statistic,
        n_bootstrap_samples=100,
        n_workers=2,
        use_processes=True
    )

    # Check that we got valid results
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'std' in result
    assert 'percentile_2.5' in result
    assert 'percentile_97.5' in result
    assert 'samples' in result
    assert isinstance(result['samples'], np.ndarray)
    assert len(result['samples']) == 100


def test_parallel_monte_carlo_with_threads():
    """Test parallel Monte Carlo simulation using threads instead of processes."""
    # Create parameter set
    params_data = {
        'mean_treatment': np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        'mean_control': np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        'sd_outcome': np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data['mean_treatment']))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create trial design
    trial_design = TrialDesign(arms=[
        DecisionOption(name="Treatment", sample_size=50),
        DecisionOption(name="Control", sample_size=50)
    ])

    # Test parallel Monte Carlo simulation with threads
    result = parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=20,
        n_workers=2,
        use_processes=False  # Use threads
    )

    # Check that we got a valid result
    assert isinstance(result, float)
    assert result >= 0


def test_parallel_bootstrap_with_threads():
    """Test parallel bootstrap sampling using threads instead of processes."""
    # Create test data
    data = np.random.normal(10, 2, 100).astype(np.float64)

    # Test parallel bootstrap sampling with threads
    result = parallel_bootstrap_sampling(
        data=data,
        statistic_func=mean_statistic,
        n_bootstrap_samples=100,
        n_workers=2,
        use_processes=False  # Use threads
    )

    # Check that we got valid results
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'std' in result
    assert 'percentile_2.5' in result
    assert 'percentile_97.5' in result
    assert 'samples' in result
    assert isinstance(result['samples'], np.ndarray)
    assert len(result['samples']) == 100


if __name__ == "__main__":
    test_monte_carlo_worker()
    test_parallel_monte_carlo_simulation()
    test_parallel_evsi_calculation()
    test_parallel_bootstrap_sampling()
    test_parallel_monte_carlo_with_threads()
    test_parallel_bootstrap_with_threads()
    print("All parallel processing tests passed!")
