"""Tests for GPU acceleration utilities in Value of Information analysis."""

import numpy as np
import pytest

from voiage.core.gpu_acceleration import (
    get_gpu_backend,
    is_gpu_available,
    array_to_gpu,
    array_to_cpu,
    gpu_jit_compile,
    gpu_vectorize,
    gpu_parallelize,
    GPUAcceleratedEVPI
)


def test_get_gpu_backend():
    """Test getting the GPU backend."""
    backend = get_gpu_backend()
    assert isinstance(backend, str)
    assert backend in ['jax', 'cupy', 'torch', 'none']


def test_is_gpu_available():
    """Test checking if GPU is available."""
    available = is_gpu_available()
    assert isinstance(available, bool)
    # Should match the result from get_gpu_backend
    assert available == (get_gpu_backend() != 'none')


def test_array_transfer():
    """Test transferring arrays between CPU and GPU."""
    # Create sample data
    cpu_array = np.random.randn(10, 3).astype(np.float64)
    
    # Get the available backend
    backend = get_gpu_backend()
    
    if backend == 'none':
        # If no GPU backend is available, these functions should raise exceptions
        with pytest.raises(RuntimeError):
            array_to_gpu(cpu_array)
        # But array_to_cpu should work with regular NumPy arrays
        result = array_to_cpu(cpu_array)
        np.testing.assert_array_equal(result, cpu_array)
    else:
        # Test transferring to GPU
        gpu_array = array_to_gpu(cpu_array, backend)
        assert gpu_array is not None
        
        # Test transferring back to CPU
        result = array_to_cpu(gpu_array, backend)
        np.testing.assert_array_equal(result, cpu_array)


def test_gpu_jit_compile():
    """Test JIT compilation with GPU backends."""
    def simple_function(x):
        return x * 2
    
    # Test with auto-detect backend
    compiled_func = gpu_jit_compile(simple_function)
    assert callable(compiled_func)
    
    # Test with specific backends
    for backend in ['jax', 'torch', 'none']:
        compiled_func = gpu_jit_compile(simple_function, backend)
        assert callable(compiled_func)


def test_gpu_vectorize():
    """Test vectorization with GPU backends."""
    def simple_function(x):
        return x * 2
    
    # Test with auto-detect backend
    vectorized_func = gpu_vectorize(simple_function)
    assert callable(vectorized_func)
    
    # Test with specific backends
    for backend in ['jax', 'cupy', 'torch', 'none']:
        vectorized_func = gpu_vectorize(simple_function, backend)
        assert callable(vectorized_func)


def test_gpu_parallelize():
    """Test parallelization with GPU backends."""
    def simple_function(x):
        return x * 2
    
    # Test with auto-detect backend
    parallelized_func = gpu_parallelize(simple_function)
    assert callable(parallelized_func)
    
    # Test with specific backends
    for backend in ['jax', 'torch', 'none']:
        parallelized_func = gpu_parallelize(simple_function, backend)
        assert callable(parallelized_func)


def test_gpu_accelerated_evpi():
    """Test GPU-accelerated EVPI calculation."""
    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3).astype(np.float64)
    
    # Get the available backend
    backend = get_gpu_backend()
    
    if backend == 'none':
        # If no GPU backend is available, this should raise an exception
        with pytest.raises(RuntimeError):
            GPUAcceleratedEVPI(backend)
    else:
        # Test with auto-detect backend
        evpi_calculator = GPUAcceleratedEVPI()
        evpi_result = evpi_calculator.calculate_evpi(net_benefit_array)
        assert isinstance(evpi_result, float)
        assert evpi_result >= 0
        
        # Test with specific backend
        evpi_calculator = GPUAcceleratedEVPI(backend)
        evpi_result = evpi_calculator.calculate_evpi(net_benefit_array)
        assert isinstance(evpi_result, float)
        assert evpi_result >= 0


def test_gpu_accelerated_evpi_edge_cases():
    """Test GPU-accelerated EVPI calculation with edge cases."""
    # Get the available backend
    backend = get_gpu_backend()
    
    if backend != 'none':
        evpi_calculator = GPUAcceleratedEVPI(backend)
        
        # Test with single strategy (EVPI should be 0)
        single_strategy_array = np.random.randn(100, 1).astype(np.float64)
        evpi_result = evpi_calculator.calculate_evpi(single_strategy_array)
        assert isinstance(evpi_result, float)
        assert evpi_result == 0.0
        
        # Test with identical strategies (EVPI should be 0)
        identical_strategies_array = np.ones((100, 3)).astype(np.float64)
        evpi_result = evpi_calculator.calculate_evpi(identical_strategies_array)
        assert isinstance(evpi_result, float)
        assert evpi_result == 0.0


if __name__ == "__main__":
    test_get_gpu_backend()
    test_is_gpu_available()
    test_array_transfer()
    test_gpu_jit_compile()
    test_gpu_vectorize()
    test_gpu_parallelize()
    test_gpu_accelerated_evpi()
    test_gpu_accelerated_evpi_edge_cases()
    print("All GPU acceleration tests passed!")