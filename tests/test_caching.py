"""Tests for caching mechanisms in DecisionAnalysis."""

from unittest.mock import patch

import numpy as np

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray


def test_caching_enabled():
    """Test that caching works when enabled."""
    # Create test data where different strategies are optimal for different samples
    # This will produce a positive EVPI
    net_benefits = np.array([[120.0, 100.0], [110.0, 90.0], [80.0, 130.0], [90.0, 140.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)

    # Create analysis with caching enabled
    analysis = DecisionAnalysis(value_array, enable_caching=True)

    # First call should compute the result
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result1 = analysis.evpi()
        mock_evpi.assert_called_once()

    # Second call should use cached result
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result2 = analysis.evpi()
        mock_evpi.assert_not_called()  # Should not be called again

    # Results should be identical and positive
    assert result1 == result2
    assert result1 > 0  # Should be positive


def test_caching_disabled():
    """Test that caching is disabled when not enabled."""
    # Create test data where different strategies are optimal for different samples
    net_benefits = np.array([[120.0, 100.0], [110.0, 90.0], [80.0, 130.0], [90.0, 140.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)

    # Create analysis with caching disabled (default)
    analysis = DecisionAnalysis(value_array)

    # Both calls should compute the result
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result1 = analysis.evpi()
        mock_evpi.assert_called_once()

    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result2 = analysis.evpi()
        mock_evpi.assert_called_once()  # Should be called again


def test_cache_invalidation():
    """Test that cache is invalidated when data changes."""
    # Create initial test data where different strategies are optimal for different samples
    net_benefits = np.array([[120.0, 100.0], [110.0, 90.0], [80.0, 130.0], [90.0, 140.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)

    # Create analysis with caching enabled
    analysis = DecisionAnalysis(value_array, enable_caching=True)

    # First call should compute the result
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result1 = analysis.evpi()
        mock_evpi.assert_called_once()

    # Second call should use cached result
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result2 = analysis.evpi()
        mock_evpi.assert_not_called()

    # Verify we have a positive result
    assert result1 > 0
    assert result1 == result2

    # Change the data to something with a different EVPI
    new_net_benefits = np.array([[150.0, 80.0], [160.0, 70.0], [70.0, 160.0], [60.0, 170.0]], dtype=np.float64)
    new_value_array = ValueArray.from_numpy(new_net_benefits)
    analysis.update_with_new_data(new_value_array)

    # Third call should compute the result again because data changed
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result3 = analysis.evpi()
        mock_evpi.assert_called_once()

    # Results 1 and 2 should be identical (cached)
    assert result1 == result2

    # Result 3 should be different (new data)
    assert result3 != result1


def test_caching_different_parameters():
    """Test that caching works with different parameter combinations."""
    # Create test data where different strategies are optimal for different samples
    net_benefits = np.array([[120.0, 100.0], [110.0, 90.0], [80.0, 130.0], [90.0, 140.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)

    # Create analysis with caching enabled
    analysis = DecisionAnalysis(value_array, enable_caching=True)

    # Call with one set of parameters
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result1 = analysis.evpi(population=1000, time_horizon=10)
        mock_evpi.assert_called_once()

    # Call with same parameters should use cache
    with patch.object(analysis.backend, 'evpi', wraps=analysis.backend.evpi) as mock_evpi:
        result2 = analysis.evpi(population=1000, time_horizon=10)
        mock_evpi.assert_not_called()

    # Results should be identical
    assert result1 == result2
    assert result1 > 0  # Should be positive


def test_caching_with_chunking():
    """Test that caching works with chunked computations."""
    # Create test data where different strategies are optimal for different samples
    net_benefits = np.array([[120.0, 100.0], [110.0, 90.0], [80.0, 130.0], [90.0, 140.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(net_benefits)

    # Create analysis with caching enabled
    analysis = DecisionAnalysis(value_array, enable_caching=True)

    # Call with chunking
    with patch.object(analysis, '_incremental_evpi', wraps=analysis._incremental_evpi) as mock_chunked:
        result1 = analysis.evpi(chunk_size=2)
        mock_chunked.assert_called_once()

    # Call with same chunking should use cache
    with patch.object(analysis, '_incremental_evpi', wraps=analysis._incremental_evpi) as mock_chunked:
        result2 = analysis.evpi(chunk_size=2)
        mock_chunked.assert_not_called()

    # Results should be identical and positive
    assert result1 == result2
    assert result1 > 0  # Should be positive


if __name__ == "__main__":
    test_caching_enabled()
    test_caching_disabled()
    test_cache_invalidation()
    test_caching_different_parameters()
    test_caching_with_chunking()
    print("All caching tests passed!")
