"""Comprehensive tests for memory optimization module to improve coverage."""

from unittest.mock import patch

import numpy as np
import pytest

from voiage.core.memory_optimization import (
    MemoryOptimizer,
    chunked_computation,
    memory_efficient_evpi_computation,
    optimize_parameter_set,
    optimize_value_array,
)
from voiage.schema import ParameterSet, ValueArray


class TestMemoryOptimizationComprehensive:
    """Comprehensive tests for memory optimization module."""

    def test_memory_optimizer_init_default(self):
        """Test MemoryOptimizer initialization with default parameters."""
        optimizer = MemoryOptimizer()

        # Check that memory limit was set to a reasonable default
        assert isinstance(optimizer.memory_limit_bytes, (int, float))
        assert optimizer.memory_limit_bytes > 0
        assert optimizer.current_memory_usage == 0.0

    def test_memory_optimizer_init_custom_limit(self):
        """Test MemoryOptimizer initialization with custom memory limit."""
        custom_limit_mb = 512.0
        optimizer = MemoryOptimizer(memory_limit_mb=custom_limit_mb)

        expected_bytes = custom_limit_mb * 1024 * 1024
        assert optimizer.memory_limit_bytes == expected_bytes
        assert optimizer.current_memory_usage == 0.0

    def test_get_memory_usage(self):
        """Test get_memory_usage method."""
        optimizer = MemoryOptimizer()

        usage = optimizer.get_memory_usage()
        assert isinstance(usage, (int, float))
        assert usage >= 0

    def test_get_available_memory(self):
        """Test get_available_memory method."""
        optimizer = MemoryOptimizer()

        available = optimizer.get_available_memory()
        assert isinstance(available, (int, float))
        assert available >= 0

        # Available memory should be less than or equal to total limit
        assert available <= optimizer.memory_limit_bytes

    def test_is_memory_available_true(self):
        """Test is_memory_available when memory is available."""
        optimizer = MemoryOptimizer(memory_limit_mb=100.0)  # 100 MB limit

        # Request small amount of memory
        is_avail = optimizer.is_memory_available(1024.0)  # 1 KB
        assert is_avail is True

    def test_is_memory_available_false(self):
        """Test is_memory_available when memory is not available."""
        optimizer = MemoryOptimizer(memory_limit_mb=1.0)  # Very small limit

        # Request large amount of memory
        is_avail = optimizer.is_memory_available(1e9)  # 1 GB
        assert is_avail is False

    def test_optimize_array_dtype_float_conversion(self):
        """Test optimize_array_dtype with float array conversion."""
        # Create float64 array
        float64_arr = np.array([[100.123456789, 150.987654321], [90.111111111, 140.222222222]], dtype=np.float64)
        optimizer = MemoryOptimizer()

        # Optimize array dtype (may convert to float32 if precision allows)
        optimized = optimizer.optimize_array_dtype(float64_arr)

        assert isinstance(optimized, np.ndarray)
        # Either same dtype or converted to float32 without significant loss
        assert optimized.shape == float64_arr.shape

    def test_optimize_array_dtype_no_change_needed(self):
        """Test optimize_array_dtype when no conversion is beneficial."""
        # Create float32 array
        float32_arr = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float32)
        optimizer = MemoryOptimizer()

        # Optimize array dtype (should remain float32)
        optimized = optimizer.optimize_array_dtype(float32_arr)

        assert isinstance(optimized, np.ndarray)
        assert optimized.dtype == np.float32
        assert optimized.shape == float32_arr.shape
        np.testing.assert_array_equal(optimized, float32_arr)

    def test_optimize_array_dtype_integer_conversion(self):
        """Test optimize_array_dtype with integer array conversion."""
        # Create array with small integers that fit in int8
        int64_arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        optimizer = MemoryOptimizer()

        # Optimize array dtype
        optimized = optimizer.optimize_array_dtype(int64_arr)

        assert isinstance(optimized, np.ndarray)
        # Should be converted to smaller integer type
        assert optimized.dtype in [np.int8, np.int16, np.int32]
        assert optimized.shape == int64_arr.shape
        np.testing.assert_array_equal(optimized, int64_arr)

    def test_optimize_array_dtype_integer_conversion_medium(self):
        """Test optimize_array_dtype with medium-sized integers."""
        # Create array with integers that need int16
        int64_arr = np.array([[100, 200], [300, 400]], dtype=np.int64)
        optimizer = MemoryOptimizer()

        # Optimize array dtype
        optimized = optimizer.optimize_array_dtype(int64_arr)

        assert isinstance(optimized, np.ndarray)
        # Should be converted to int16 or int32
        assert optimized.dtype in [np.int16, np.int32]
        assert optimized.shape == int64_arr.shape
        np.testing.assert_array_equal(optimized, int64_arr)

    def test_chunk_large_array_small_array(self):
        """Test chunk_large_array with array smaller than chunk size."""
        # Create small array
        small_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # 2x3 array
        optimizer = MemoryOptimizer()

        # Chunk with large size (bigger than array)
        chunks = optimizer.chunk_large_array(small_arr, max_chunk_size=10)

        assert isinstance(chunks, list)
        assert len(chunks) == 1  # Should remain as single chunk
        np.testing.assert_array_equal(chunks[0], small_arr)

    def test_chunk_large_array_exact_chunk_size(self):
        """Test chunk_large_array with exact chunk size."""
        # Create larger array
        large_arr = np.array([
            [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]
        ], dtype=np.float64)  # 5x3 array
        optimizer = MemoryOptimizer()

        # Chunk with exact size
        chunks = optimizer.chunk_large_array(large_arr, max_chunk_size=2)

        assert isinstance(chunks, list)
        assert len(chunks) == 3  # Should be split into 3 chunks: 2+2+1

        # Verify chunks contain the correct data
        np.testing.assert_array_equal(chunks[0], large_arr[0:2])  # First 2 rows
        np.testing.assert_array_equal(chunks[1], large_arr[2:4])  # Next 2 rows
        np.testing.assert_array_equal(chunks[2], large_arr[4:5])  # Last row

    def test_estimate_memory_usage_numpy_array(self):
        """Test estimate_memory_usage for numpy array."""
        test_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        optimizer = MemoryOptimizer()

        usage = optimizer.estimate_memory_usage(test_arr)

        assert isinstance(usage, (int, float))
        assert usage == test_arr.nbytes  # Should match exact nbytes

    def test_estimate_memory_usage_value_array(self):
        """Test estimate_memory_usage for ValueArray."""
        test_data = np.array([[100.0, 150.0], [90.0, 140.0]], dtype=np.float64)
        value_array = ValueArray.from_numpy(test_data, ["Strategy A", "Strategy B"])
        optimizer = MemoryOptimizer()

        usage = optimizer.estimate_memory_usage(value_array)

        assert isinstance(usage, (int, float))
        # The usage should be >= 0 (though exact value depends on internal xarray structure)

    def test_estimate_memory_usage_dict(self):
        """Test estimate_memory_usage for dictionary."""
        test_dict = {
            "array1": np.array([1.0, 2.0, 3.0]),
            "array2": np.array([4.0, 5.0, 6.0]),
            "nested": {"inner": np.array([7.0, 8.0])}
        }
        optimizer = MemoryOptimizer()

        usage = optimizer.estimate_memory_usage(test_dict)

        assert isinstance(usage, (int, float))
        assert usage >= 0

    def test_estimate_memory_usage_list(self):
        """Test estimate_memory_usage for list."""
        test_list = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            "some string"
        ]
        optimizer = MemoryOptimizer()

        usage = optimizer.estimate_memory_usage(test_list)

        assert isinstance(usage, (int, float))
        assert usage >= 0

    def test_force_garbage_collection(self):
        """Test force_garbage_collection method."""
        optimizer = MemoryOptimizer()

        # This should not raise an exception
        optimizer.force_garbage_collection()

        # Verify memory usage is still reasonable after GC
        usage = optimizer.get_memory_usage()
        assert isinstance(usage, (int, float))
        assert usage >= 0

    def test_monitor_memory_usage_low_usage(self):
        """Test monitor_memory_usage with low memory usage."""
        optimizer = MemoryOptimizer(memory_limit_mb=100.0)  # Set a reasonable limit

        # With low memory usage, this shouldn't trigger a warning
        optimizer.monitor_memory_usage(warning_threshold=0.9)  # High threshold

    def test_optimize_value_array(self):
        """Test optimize_value_array function."""
        # Create test ValueArray with large dtype
        test_values = np.array([[100.123456, 150.654321], [90.111111, 140.222222]], dtype=np.float64)
        value_array = ValueArray.from_numpy(test_values, ["Strategy A", "Strategy B"])

        # Optimize the ValueArray
        optimized = optimize_value_array(value_array)

        assert isinstance(optimized, ValueArray)
        assert optimized.values.shape == value_array.values.shape

    def test_optimize_parameter_set(self):
        """Test optimize_parameter_set function."""
        # Create test ParameterSet
        param_dict = {
            "param1": np.array([0.123456789, 0.987654321], dtype=np.float64),
            "param2": np.array([10.111111111, 20.222222222], dtype=np.float64)
        }
        param_set = ParameterSet.from_numpy_or_dict(param_dict)

        # Optimize the ParameterSet
        optimized = optimize_parameter_set(param_set)

        assert isinstance(optimized, ParameterSet)
        # Should have same number of parameters
        assert len(optimized.parameters) == len(param_set.parameters)

    def test_optimize_parameter_set_integer_parameters(self):
        """Test optimize_parameter_set with integer parameters."""
        # Create test ParameterSet with large integer arrays
        param_dict = {
            "param1": np.array([100, 200, 300], dtype=np.int64),
            "param2": np.array([1000, 2000, 3000], dtype=np.int64)
        }
        param_set = ParameterSet.from_numpy_or_dict(param_dict)

        # Optimize the ParameterSet
        optimized = optimize_parameter_set(param_set)

        assert isinstance(optimized, ParameterSet)

    def test_chunked_computation_with_callable(self):
        """Test chunked_computation with a simple function."""
        def simple_sum(arr):
            return np.sum(arr, axis=0)

        # Create test data
        test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

        # Run chunked computation
        results = chunked_computation(simple_sum, test_data, chunk_size=2)

        assert isinstance(results, list)
        assert len(results) == 2  # 4 rows divided into chunks of 2

        # Verify results
        expected_chunk1 = np.array([1.0+3.0, 2.0+4.0])  # [4.0, 6.0]
        expected_chunk2 = np.array([5.0+7.0, 6.0+8.0])  # [12.0, 14.0]

        np.testing.assert_array_almost_equal(results[0], expected_chunk1)
        np.testing.assert_array_almost_equal(results[1], expected_chunk2)

    def test_chunked_computation_no_chunk_size(self):
        """Test chunked_computation without specifying chunk_size."""
        def simple_mean(arr):
            return np.mean(arr, axis=0)

        # Create small test data
        test_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        # Run chunked computation without chunk_size (should process directly)
        results = chunked_computation(simple_mean, test_data, chunk_size=None)

        assert isinstance(results, list)
        assert len(results) >= 1  # Should return at least one result

    def test_memory_efficient_evpi_computation_simple(self):
        """Test memory_efficient_evpi_computation with simple data."""
        # Create test net benefit array
        nb_array = np.array([
            [100.0, 150.0, 120.0],  # Sample 1
            [90.0, 140.0, 130.0],   # Sample 2
            [110.0, 130.0, 140.0]   # Sample 3
        ], dtype=np.float64)

        # Test with default chunk_size
        result = memory_efficient_evpi_computation(nb_array)

        assert isinstance(result, float)
        assert result >= 0  # EVPI should be non-negative

    def test_memory_efficient_evpi_computation_with_chunk_size(self):
        """Test memory_efficient_evpi_computation with specified chunk_size."""
        # Create larger test net benefit array
        nb_array = np.random.rand(10, 3).astype(np.float64) * 1000

        # Test with specific chunk_size
        result = memory_efficient_evpi_computation(nb_array, chunk_size=5)

        assert isinstance(result, float)
        assert result >= 0  # EVPI should be non-negative

    def test_memory_efficient_evpi_computation_single_strategy(self):
        """Test memory_efficient_evpi_computation with single strategy."""
        # Create array with only one strategy (should give EVPI = 0)
        nb_array = np.array([[100.0], [90.0], [110.0]], dtype=np.float64)

        result = memory_efficient_evpi_computation(nb_array)

        assert isinstance(result, float)
        # EVPI for single strategy should be 0 (within floating point tolerance)
        assert abs(result) < 1e-9

    def test_memory_efficient_evpi_computation_identical_strategies(self):
        """Test memory_efficient_evpi_computation with identical strategies."""
        # Create array where all strategies have identical values
        nb_array = np.array([
            [100.0, 100.0, 100.0],  # All same
            [110.0, 110.0, 110.0],  # All same
            [120.0, 120.0, 120.0]   # All same
        ], dtype=np.float64)

        result = memory_efficient_evpi_computation(nb_array)

        assert isinstance(result, float)
        # EVPI for identical strategies should be 0 (within floating point tolerance)
        assert abs(result) < 1e-9

    def test_memory_optimizer_with_warning_monitoring(self):
        """Test memory optimizer monitoring with simulated high usage."""
        # Create a MemoryOptimizer with a small limit
        optimizer = MemoryOptimizer(memory_limit_mb=0.001)  # Very small limit in MB

        # Patch virtual_memory to simulate high usage
        with patch('psutil.virtual_memory') as mock_vm:
            # Create a mock that simulates high memory usage
            mock_memory_info = type('MemoryInfo', (), {})()
            mock_memory_info.used = 999999999  # Simulate very high memory usage
            mock_memory_info.total = 1000000000  # Total memory

            mock_vm.return_value = mock_memory_info

            # This should trigger a warning due to high memory usage
            with pytest.warns(ResourceWarning):
                optimizer.monitor_memory_usage(warning_threshold=0.1)  # Low threshold

    def test_memory_efficient_evpi_computation_empty_array(self):
        """Test memory_efficient_evpi_computation with empty array."""
        # Create empty array
        empty_array = np.array([], dtype=np.float64).reshape(0, 2)  # 0 samples, 2 strategies

        # Should handle gracefully
        result = memory_efficient_evpi_computation(empty_array)

        assert isinstance(result, float)
        assert result == 0.0  # EVPI of empty array should be 0

    def test_chunk_large_array_very_large_array(self):
        """Test chunk_large_array with a very large array."""
        # Create a larger array to test chunking
        large_arr = np.random.rand(20, 3).astype(np.float64)
        optimizer = MemoryOptimizer()

        # Chunk into smaller pieces
        chunks = optimizer.chunk_large_array(large_arr, max_chunk_size=7)

        assert isinstance(chunks, list)
        assert len(chunks) == 3  # 20 // 7 = 2 remainder 6, so 3 chunks (7 + 7 + 6)

        # Verify that chunks reconstruct the original array
        reconstructed = np.vstack(chunks)
        np.testing.assert_array_equal(reconstructed, large_arr)

    def test_estimate_memory_usage_various_objects(self):
        """Test estimate_memory_usage with various object types."""
        optimizer = MemoryOptimizer()

        # Test with different types of objects
        test_objects = [
            np.array([1, 2, 3, 4, 5]),  # numpy array
            [1, 2, 3, 4, 5],  # list
            {"a": 1, "b": 2},  # dict
            (1, 2, 3, 4, 5),  # tuple
            "a string",  # string
            123,  # integer
            1.23,  # float
            True,  # boolean
        ]

        for obj in test_objects:
            usage = optimizer.estimate_memory_usage(obj)
            assert isinstance(usage, (int, float))
            assert usage >= 0
