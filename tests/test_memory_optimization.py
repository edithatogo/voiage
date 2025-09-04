"""Tests for memory optimization utilities in Value of Information analysis."""

import numpy as np
import pytest
import xarray as xr

from voiage.core.memory_optimization import (
    MemoryOptimizer,
    optimize_value_array,
    optimize_parameter_set,
    chunked_computation,
    memory_efficient_evpi_computation
)
from voiage.schema import ValueArray, ParameterSet


def test_memory_optimizer_initialization():
    """Test MemoryOptimizer initialization."""
    # Test with default memory limit
    optimizer = MemoryOptimizer()
    assert optimizer.memory_limit_bytes > 0
    
    # Test with custom memory limit
    optimizer = MemoryOptimizer(memory_limit_mb=100)
    assert optimizer.memory_limit_bytes == 100 * 1024 * 1024


def test_memory_optimizer_memory_usage():
    """Test memory usage functions."""
    optimizer = MemoryOptimizer()
    
    # Test get_memory_usage
    usage = optimizer.get_memory_usage()
    assert isinstance(usage, (float, int))
    assert usage >= 0
    
    # Test get_available_memory
    available = optimizer.get_available_memory()
    assert isinstance(available, (float, int))
    assert available >= 0
    
    # Test is_memory_available
    result = optimizer.is_memory_available(1024)  # 1KB
    assert isinstance(result, bool)


def test_optimize_array_dtype():
    """Test array dtype optimization."""
    optimizer = MemoryOptimizer()
    
    # Test float64 to float32 conversion when possible
    float64_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    optimized = optimizer.optimize_array_dtype(float64_array)
    # This might stay as float64 or convert to float32 depending on system
    
    # Test integer optimization
    int64_array = np.array([1, 2, 3], dtype=np.int64)
    optimized = optimizer.optimize_array_dtype(int64_array)
    # Should convert to smaller integer type if possible
    
    # Test array that can't be optimized
    small_int_array = np.array([1, 2, 3], dtype=np.int8)
    optimized = optimizer.optimize_array_dtype(small_int_array)
    assert optimized.dtype == np.int8


def test_chunk_large_array():
    """Test chunking large arrays."""
    optimizer = MemoryOptimizer()
    
    # Create test array
    test_array = np.random.randn(100, 3).astype(np.float64)
    
    # Test with chunk size larger than array
    chunks = optimizer.chunk_large_array(test_array, 200)
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], test_array)
    
    # Test with chunk size smaller than array
    chunks = optimizer.chunk_large_array(test_array, 30)
    assert len(chunks) == 4  # 100/30 = 3.33, so 4 chunks
    # Check that all chunks combined equal original array
    combined = np.vstack(chunks)
    np.testing.assert_array_equal(combined, test_array)


def test_estimate_memory_usage():
    """Test memory usage estimation."""
    optimizer = MemoryOptimizer()
    
    # Test numpy array
    test_array = np.random.randn(100, 3).astype(np.float64)
    estimated = optimizer.estimate_memory_usage(test_array)
    actual = test_array.nbytes
    assert estimated == actual
    
    # Test dictionary
    test_dict = {"a": np.array([1, 2, 3]), "b": np.array([4.0, 5.0])}
    estimated = optimizer.estimate_memory_usage(test_dict)
    # Should be at least the sum of array sizes
    assert estimated >= (test_dict["a"].nbytes + test_dict["b"].nbytes)
    
    # Test list
    test_list = [np.array([1, 2]), np.array([3.0, 4.0])]
    estimated = optimizer.estimate_memory_usage(test_list)
    # Should be at least the sum of array sizes
    assert estimated >= (test_list[0].nbytes + test_list[1].nbytes)


def test_optimize_value_array():
    """Test ValueArray optimization."""
    # Create test ValueArray
    test_data = np.random.randn(100, 3).astype(np.float64)
    value_array = ValueArray.from_numpy(test_data)
    
    # Optimize
    optimized = optimize_value_array(value_array)
    assert isinstance(optimized, ValueArray)
    # Values should be the same or optimized
    assert optimized.values.shape == value_array.values.shape


def test_optimize_parameter_set():
    """Test ParameterSet optimization."""
    # Create test ParameterSet
    params_data = {
        'param1': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'param2': np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data['param1']))},
    )
    parameter_set = ParameterSet(dataset=dataset)
    
    # Optimize
    optimized = optimize_parameter_set(parameter_set)
    assert isinstance(optimized, ParameterSet)
    # Parameters should be the same or optimized
    assert len(optimized.parameters) == len(parameter_set.parameters)


def test_chunked_computation():
    """Test chunked computation."""
    # Create test data
    test_data = np.random.randn(100, 3).astype(np.float64)
    
    # Define a simple function to apply
    def sum_function(chunk):
        return np.sum(chunk)
    
    # Test with automatic chunk size
    results = chunked_computation(sum_function, test_data)
    assert isinstance(results, list)
    assert len(results) >= 1
    
    # Test with specific chunk size
    results = chunked_computation(sum_function, test_data, chunk_size=30)
    assert isinstance(results, list)
    assert len(results) == 4  # 100/30 = 3.33, so 4 chunks


def test_memory_efficient_evpi_computation():
    """Test memory-efficient EVPI computation."""
    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(100, 3).astype(np.float64)
    
    # Test with automatic chunk size
    evpi_result = memory_efficient_evpi_computation(test_data)
    assert isinstance(evpi_result, float)
    assert evpi_result >= 0
    
    # Test with specific chunk size
    evpi_result = memory_efficient_evpi_computation(test_data, chunk_size=30)
    assert isinstance(evpi_result, float)
    assert evpi_result >= 0
    
    # Test with small data (should compute directly)
    small_data = np.random.randn(10, 2).astype(np.float64)
    evpi_result = memory_efficient_evpi_computation(small_data)
    assert isinstance(evpi_result, float)
    assert evpi_result >= 0


def test_memory_efficient_evpi_edge_cases():
    """Test memory-efficient EVPI computation with edge cases."""
    # Test with single strategy (EVPI should be 0)
    single_strategy_data = np.random.randn(100, 1).astype(np.float64)
    evpi_result = memory_efficient_evpi_computation(single_strategy_data)
    assert isinstance(evpi_result, float)
    assert evpi_result == 0.0
    
    # Test with identical strategies (EVPI should be 0)
    identical_strategies_data = np.ones((100, 3)).astype(np.float64)
    evpi_result = memory_efficient_evpi_computation(identical_strategies_data)
    assert isinstance(evpi_result, float)
    assert evpi_result == 0.0


if __name__ == "__main__":
    test_memory_optimizer_initialization()
    test_memory_optimizer_memory_usage()
    test_optimize_array_dtype()
    test_chunk_large_array()
    test_estimate_memory_usage()
    test_optimize_value_array()
    test_optimize_parameter_set()
    test_chunked_computation()
    test_memory_efficient_evpi_computation()
    test_memory_efficient_evpi_edge_cases()
    print("All memory optimization tests passed!")