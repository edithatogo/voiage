"""Tests for edge cases and extreme parameter values."""

import numpy as np
import pytest
import xarray as xr

from voiage.analysis import DecisionAnalysis
from voiage.methods.basic import evppi
from voiage.schema import ParameterSet, ValueArray


def test_extreme_large_numbers():
    """Test with extremely large numbers to check for overflow issues."""
    # Create net benefits with very large values
    large_values = np.array([
        [1e100, 1e100 + 1e90],  # Strategy A and B for sample 1
        [1e100 + 1e80, 1e100 + 2e90],  # Strategy A and B for sample 2
        [1e100 + 2e80, 1e100 + 1.5e90],  # Strategy A and B for sample 3
    ])

    value_array = ValueArray.from_numpy(large_values)
    analysis = DecisionAnalysis(value_array)

    # This should not raise an exception
    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))
    assert evpi_result >= 0  # EVPI should always be non-negative


def test_extreme_small_numbers():
    """Test with extremely small numbers to check for underflow issues."""
    # Create net benefits with very small values
    small_values = np.array([
        [1e-100, 2e-100],  # Strategy A and B for sample 1
        [1.5e-100, 1.8e-100],  # Strategy A and B for sample 2
        [2e-100, 1.2e-100],  # Strategy A and B for sample 3
    ])

    value_array = ValueArray.from_numpy(small_values)
    analysis = DecisionAnalysis(value_array)

    # This should not raise an exception
    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))
    assert evpi_result >= 0  # EVPI should always be non-negative


def test_mixed_extreme_values():
    """Test with a mix of very large and very small values."""
    mixed_values = np.array([
        [1e100, 1e-100],  # Strategy A and B for sample 1
        [1e-100, 1e100],  # Strategy A and B for sample 2
    ])

    value_array = ValueArray.from_numpy(mixed_values)
    analysis = DecisionAnalysis(value_array)

    # This should not raise an exception
    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))
    assert evpi_result >= 0  # EVPI should always be non-negative


def test_single_parameter_degenerate_case():
    """Test with a single parameter (degenerate case)."""
    # Simple net benefits with correct dtype
    nb_array = np.array([
        [100, 150],  # Strategy A and B for sample 1
        [120, 130],  # Strategy A and B for sample 2
        [80, 100],   # Strategy A and B for sample 3
    ], dtype=np.float64)

    # Single parameter
    param_dict = {"single_param": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
    dataset = xr.Dataset(
        {k: ("n_samples", v) for k, v in param_dict.items()},
        coords={"n_samples": np.arange(3)}
    )
    parameter_set = ParameterSet(dataset=dataset)

    value_array = ValueArray.from_numpy(nb_array)
    analysis = DecisionAnalysis(value_array, parameter_set)

    # Test EVPI
    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))
    assert evpi_result >= 0

    # Test EVPPI with single parameter
    evppi_result = evppi(value_array, parameter_set, ["single_param"])
    assert isinstance(evppi_result, (int, float))
    assert evppi_result >= 0
    assert evppi_result <= evpi_result + 1e-10  # EVPPI should not exceed EVPI


def test_identical_strategies_degenerate_case():
    """Test with identical strategies (degenerate case)."""
    # Identical strategies with correct dtype
    identical_values = np.array([
        [100, 100],  # Strategy A and B for sample 1
        [150, 150],  # Strategy A and B for sample 2
        [120, 120],  # Strategy A and B for sample 3
    ], dtype=np.float64)

    value_array = ValueArray.from_numpy(identical_values)
    analysis = DecisionAnalysis(value_array)

    # EVPI should be zero for identical strategies
    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))
    assert abs(evpi_result) < 1e-10  # Should be effectively zero


def test_single_strategy_degenerate_case():
    """Test with a single strategy (degenerate case)."""
    # Single strategy with correct dtype
    single_strategy_values = np.array([
        [100],  # Only Strategy A for sample 1
        [150],  # Only Strategy A for sample 2
        [120],  # Only Strategy A for sample 3
    ], dtype=np.float64)

    value_array = ValueArray.from_numpy(single_strategy_values)
    analysis = DecisionAnalysis(value_array)

    # EVPI should be zero for single strategy
    evpi_result = analysis.evpi()
    assert isinstance(evpi_result, (int, float))
    assert evpi_result == 0


def test_nan_values():
    """Test handling of NaN values."""
    # Net benefits with NaN
    nan_values = np.array([
        [100, 150],    # Strategy A and B for sample 1
        [np.nan, 130], # Strategy A and B for sample 2
        [80, np.nan],  # Strategy A and B for sample 3
    ], dtype=np.float64)

    value_array = ValueArray.from_numpy(nan_values)
    analysis = DecisionAnalysis(value_array)

    # The function should handle NaN gracefully and return a valid result
    evpi_result = analysis.evpi()
    # The result should be a valid float (likely 0.0 due to NaN handling)
    assert isinstance(evpi_result, (int, float))


def test_infinite_values():
    """Test handling of infinite values."""
    # Net benefits with infinity
    inf_values = np.array([
        [100, 150],      # Strategy A and B for sample 1
        [np.inf, 130],   # Strategy A and B for sample 2
        [80, -np.inf],   # Strategy A and B for sample 3
    ], dtype=np.float64)

    value_array = ValueArray.from_numpy(inf_values)
    analysis = DecisionAnalysis(value_array)

    # The function should handle infinity gracefully and return a valid result
    evpi_result = analysis.evpi()
    # The result should be a valid float (likely 0.0 due to inf handling)
    assert isinstance(evpi_result, (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
