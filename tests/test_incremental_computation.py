"""Tests for incremental computation support in DecisionAnalysis."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.schema import ParameterSet, ValueArray


def test_incremental_evpi():
    """Test incremental EVPI computation with chunking."""
    # Create large test data
    np.random.seed(42)
    net_benefits = np.random.randn(1000, 3) * 100
    # Make sure strategy 0 is optimal on average
    net_benefits[:, 0] += 50

    value_array = ValueArray.from_numpy(net_benefits)
    analysis = DecisionAnalysis(value_array)

    # Calculate EVPI with and without chunking
    evpi_standard = analysis.evpi()
    evpi_chunked = analysis.evpi(chunk_size=100)

    # Results should be very close (allowing for small numerical differences)
    assert abs(evpi_standard - evpi_chunked) < 1e-10


def test_incremental_evppi():
    """Test incremental EVPPI computation with chunking."""
    # Create test data with parameters
    np.random.seed(42)
    n_samples = 500
    net_benefits = np.random.randn(n_samples, 2) * 100
    # Make strategy 0 better when param1 is small
    net_benefits[:, 0] += np.random.randn(n_samples) * 50

    # Create parameter samples
    param_dict = {
        "param1": np.random.randn(n_samples) * 10,
        "param2": np.random.randn(n_samples) * 5
    }
    parameter_set = ParameterSet.from_numpy_or_dict(param_dict)

    value_array = ValueArray.from_numpy(net_benefits)
    analysis = DecisionAnalysis(value_array, parameter_set)

    # Calculate EVPPI with and without chunking
    try:
        evppi_standard = analysis.evppi()
        evppi_chunked = analysis.evppi(chunk_size=50)

        # Results should be very close (allowing for small numerical differences)
        assert abs(evppi_standard - evppi_chunked) < 1e-10
    except ImportError:
        # If sklearn is not available, skip this test
        pytest.skip("scikit-learn not available")


def test_incremental_computation_edge_cases():
    """Test edge cases for incremental computation."""
    # Test with small dataset and large chunk size
    net_benefits = np.array([[100.0, 120.0], [90.0, 130.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    analysis = DecisionAnalysis(value_array)

    # Should work even when chunk size is larger than dataset
    evpi_result = analysis.evpi(chunk_size=100)
    assert isinstance(evpi_result, float)
    assert evpi_result >= 0


def test_incremental_computation_single_strategy():
    """Test incremental computation with single strategy."""
    # Single strategy case
    net_benefits = np.array([[100.0], [90.0], [110.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)
    analysis = DecisionAnalysis(value_array)

    # Should return 0 for single strategy
    evpi_result = analysis.evpi(chunk_size=2)
    assert evpi_result == 0.0


if __name__ == "__main__":
    test_incremental_evpi()
    test_incremental_evppi()
    test_incremental_computation_edge_cases()
    test_incremental_computation_single_strategy()
    print("All incremental computation tests passed!")
