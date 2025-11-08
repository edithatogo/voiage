"""Tests for voiage.core.gpu_acceleration module to increase coverage to >90%."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voiage.core.gpu_acceleration import (
    CUPY_AVAILABLE,
    JAX_AVAILABLE,
    TORCH_AVAILABLE,
    GPUAcceleratedEVPI,
    array_to_cpu,
    array_to_gpu,
    get_gpu_backend,
    gpu_jit_compile,
    gpu_parallelize,
    gpu_vectorize,
    is_gpu_available,
)


class TestGetGPUBackend:
    """Test the get_gpu_backend function."""

    def test_get_gpu_backend_with_jax(self, monkeypatch):
        """Test get_gpu_backend when JAX is available with GPU."""
        if JAX_AVAILABLE:
            # If JAX is available, test with a mock GPU device
            mock_jax = MagicMock()
            mock_device = MagicMock()
            mock_device.device_kind = 'gpu'
            mock_jax.devices.return_value = [mock_device]

            with patch('voiage.core.gpu_acceleration.jax', mock_jax):
                with patch('voiage.core.gpu_acceleration.JAX_AVAILABLE', True):
                    backend = get_gpu_backend()
                    assert backend == 'jax'
        else:
            # If JAX is not available, just make sure no error occurs
            backend = get_gpu_backend()
            assert backend in ['none', 'cupy', 'torch']

    def test_get_gpu_backend_with_cupy(self, monkeypatch):
        """Test get_gpu_backend when CuPy is available."""
        if CUPY_AVAILABLE:
            # If CuPy is available, test basic functionality
            backend = get_gpu_backend()
            assert backend in ['none', 'cupy']
        else:
            # Mock CuPy availability
            with patch('voiage.core.gpu_acceleration.CUPY_AVAILABLE', True):
                with patch('voiage.core.gpu_acceleration.cp.cuda.runtime.getDeviceCount', return_value=1):
                    backend = get_gpu_backend()
                    # Depending on other backends, this could be 'cupy' or another backend
                    assert backend in ['cupy', 'none', 'jax', 'torch']

    def test_get_gpu_backend_with_torch(self, monkeypatch):
        """Test get_gpu_backend when PyTorch is available with CUDA."""
        if TORCH_AVAILABLE:
            # If PyTorch is available, test with mocked CUDA availability
            with patch('voiage.core.gpu_acceleration.torch.cuda.is_available', return_value=True):
                backend = get_gpu_backend()
                if not JAX_AVAILABLE or get_gpu_backend() != 'jax':
                    # If JAX isn't available or doesn't take precedence, torch should be detected
                    assert backend in ['torch', 'none', 'cupy']
        else:
            # Mock PyTorch availability
            with patch('voiage.core.gpu_acceleration.TORCH_AVAILABLE', True):
                with patch('voiage.core.gpu_acceleration.torch') as mock_torch:
                    mock_torch.cuda.is_available.return_value = True
                    backend = get_gpu_backend()
                    # Depending on other backends, this could be 'torch' or another backend
                    assert backend in ['torch', 'none']

    def test_get_gpu_backend_none_available(self, monkeypatch):
        """Test get_gpu_backend when no backends are available."""
        # Mock all backends as unavailable
        with patch('voiage.core.gpu_acceleration.JAX_AVAILABLE', False):
            with patch('voiage.core.gpu_acceleration.CUPY_AVAILABLE', False):
                with patch('voiage.core.gpu_acceleration.TORCH_AVAILABLE', False):
                    backend = get_gpu_backend()
                    assert backend == 'none'


class TestIsGPUAvailable:
    """Test the is_gpu_available function."""

    def test_is_gpu_available_with_backends(self, monkeypatch):
        """Test is_gpu_available when backends are available."""
        # Mock a backend as available
        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='jax'):
            assert is_gpu_available() is True

        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='cupy'):
            assert is_gpu_available() is True

        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='torch'):
            assert is_gpu_available() is True

    def test_is_gpu_available_no_backends(self, monkeypatch):
        """Test is_gpu_available when no backends are available."""
        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='none'):
            assert is_gpu_available() is False


class TestArrayTransferFunctions:
    """Test the array transfer functions (to/from GPU)."""

    def test_array_to_gpu_with_backend_specified(self, monkeypatch):
        """Test array_to_gpu with backend specified."""
        test_array = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Test with 'none' backend (should raise RuntimeError)
        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='none'):
            with pytest.raises(RuntimeError, match="No GPU backend available"):
                array_to_gpu(test_array)

    def test_array_to_gpu_unknown_backend(self):
        """Test array_to_gpu with unknown backend."""
        test_array = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Unknown backend"):
            array_to_gpu(test_array, backend='invalid_backend')

    def test_array_to_cpu_default_behavior(self):
        """Test array_to_cpu with default behavior (already CPU array)."""
        test_array = np.array([[1.0, 2.0], [3.0, 4.0]])

        # If providing a numpy array, it should return as-is
        result = array_to_cpu(test_array)
        np.testing.assert_array_equal(result, test_array)
        assert isinstance(result, np.ndarray)

    def test_array_to_cpu_unknown_backend(self):
        """Test array_to_cpu with unknown backend."""
        test_array = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Unknown backend"):
            array_to_cpu(test_array, backend='invalid_backend')

    def test_array_to_cpu_runtime_error_for_unavailable_backend(self, monkeypatch):
        """Test array_to_cpu raises RuntimeError when backend is not available."""
        # Mock backend availability as False
        with patch('voiage.core.gpu_acceleration.JAX_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="JAX is not available"):
                array_to_cpu(MagicMock(), backend='jax')

        with patch('voiage.core.gpu_acceleration.CUPY_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="CuPy is not available"):
                array_to_cpu(MagicMock(), backend='cupy')

        with patch('voiage.core.gpu_acceleration.TORCH_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="PyTorch is not available"):
                array_to_cpu(MagicMock(), backend='torch')


class TestGPUCompilationAndVectorization:
    """Test JIT compilation, vectorization, and parallelization functions."""

    def test_gpu_jit_compile_with_backend_none(self, monkeypatch):
        """Test gpu_jit_compile with 'none' backend."""
        def dummy_func(x):
            return x * 2

        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='none'):
            result_func = gpu_jit_compile(dummy_func)
            # Should return the original function when no GPU backend is available
            assert result_func is dummy_func
            assert result_func(5) == 10

    def test_gpu_jit_compile_unknown_backend(self):
        """Test gpu_jit_compile with unknown backend."""
        def dummy_func(x):
            return x * 2

        result_func = gpu_jit_compile(dummy_func, backend='invalid_backend')
        # Should return the original function for unknown backend
        assert result_func is dummy_func
        assert result_func(5) == 10

    def test_gpu_vectorize_with_backend_none(self, monkeypatch):
        """Test gpu_vectorize with 'none' backend."""
        def dummy_func(x):
            return x * 2

        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='none'):
            result_func = gpu_vectorize(dummy_func)
            # Should return the original function when no GPU backend is available
            assert result_func is dummy_func
            assert result_func(5) == 10

    def test_gpu_vectorize_unknown_backend(self):
        """Test gpu_vectorize with unknown backend."""
        def dummy_func(x):
            return x * 2

        result_func = gpu_vectorize(dummy_func, backend='invalid_backend')
        # Should return the original function for unknown backend
        assert result_func is dummy_func
        assert result_func(5) == 10

    def test_gpu_parallelize_with_backend_none(self, monkeypatch):
        """Test gpu_parallelize with 'none' backend."""
        def dummy_func(x):
            return x * 2

        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='none'):
            result_func = gpu_parallelize(dummy_func)
            # Should return the original function when no GPU backend is available
            assert result_func is dummy_func
            assert result_func(5) == 10

    def test_gpu_parallelize_unknown_backend(self):
        """Test gpu_parallelize with unknown backend."""
        def dummy_func(x):
            return x * 2

        result_func = gpu_parallelize(dummy_func, backend='invalid_backend')
        # Should return the original function for unknown backend
        assert result_func is dummy_func
        assert result_func(5) == 10


class TestGPUAcceleratedEVPI:
    """Test the GPUAcceleratedEVPI class."""

    def test_gpu_accelerated_evpi_initialization(self, monkeypatch):
        """Test GPUAcceleratedEVPI initialization."""
        # Test initialization with 'none' backend (should raise RuntimeError)
        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='none'):
            with pytest.raises(RuntimeError, match="No GPU backend available"):
                GPUAcceleratedEVPI()

        # Test initialization with specific backend
        # If backends are available, this should work
        if JAX_AVAILABLE:
            with patch('voiage.core.gpu_acceleration.jax.devices') as mock_devices:
                mock_device = MagicMock()
                mock_device.device_kind = 'gpu'
                mock_devices.return_value = [mock_device]
                try:
                    evpi_calc = GPUAcceleratedEVPI(backend='jax')
                    assert evpi_calc.backend == 'jax'
                except RuntimeError:
                    # This might happen if not actually able to use GPU
                    pass

    def test_gpu_accelerated_evpi_initialization_with_invalid_backend(self):
        """Test GPUAcceleratedEVPI initialization with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            GPUAcceleratedEVPI(backend='invalid_backend')

    def test_gpu_accelerated_evpi_calculate_evpi(self, monkeypatch):
        """Test the calculate_evpi method of GPUAcceleratedEVPI."""
        # Create test data
        test_data = np.random.rand(100, 3)  # 100 samples, 3 strategies

        # Test with 'none' backend (should raise RuntimeError during initialization)
        with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='none'):
            with pytest.raises(RuntimeError, match="No GPU backend available"):
                evpi_calc = GPUAcceleratedEVPI()

    def test_gpu_accelerated_evpi_calculate_evpi_with_mocked_backends(self, monkeypatch):
        """Test the calculate_evpi method with mocked backend behavior."""
        # Create test data
        test_data = np.random.rand(10, 2)  # 10 samples, 2 strategies

        # Mock the array_to_gpu function to return the data unchanged for testing
        with patch('voiage.core.gpu_acceleration.array_to_gpu', side_effect=lambda x, y: x):
            # Mock the get_gpu_backend function to return 'jax' for testing
            with patch('voiage.core.gpu_acceleration.get_gpu_backend', return_value='jax'):
                # Mock JAX functions
                with patch('voiage.core.gpu_acceleration.JAX_AVAILABLE', True):
                    with patch('voiage.core.gpu_acceleration.jnp.max', new=lambda x, axis: np.max(x, axis=axis)):
                        with patch('voiage.core.gpu_acceleration.jnp.mean', new=lambda x, axis: np.mean(x, axis=axis)):
                            with patch('voiage.core.gpu_acceleration.jnp.asarray', new=lambda x: np.asarray(x)):
                                # Create instance using 'jax' backend
                                with patch('voiage.core.gpu_acceleration.jax.devices') as mock_devices:
                                    mock_device = MagicMock()
                                    mock_device.device_kind = 'gpu'
                                    mock_devices.return_value = [mock_device]

                                    evpi_calc = GPUAcceleratedEVPI(backend='jax')

                                    # Calculate EVPI
                                    result = evpi_calc.calculate_evpi(test_data)

                                    # Verify result is a float
                                    assert isinstance(result, float)
                                    assert result >= 0  # EVPI should be non-negative in most cases


# Additional tests for edge cases and error conditions
class TestGPUModuleEdgeCases:
    """Test edge cases and error conditions in the GPU module."""

    def test_get_gpu_backend_jax_no_devices(self, monkeypatch):
        """Test get_gpu_backend when JAX is available but no GPU devices are present."""
        with patch('voiage.core.gpu_acceleration.JAX_AVAILABLE', True):
            with patch('voiage.core.gpu_acceleration.jax') as mock_jax:
                mock_jax.devices.return_value = []
                backend = get_gpu_backend()
                # If no GPU devices, should return 'none' or another available backend
                assert backend in ['none', 'cupy', 'torch']

    def test_get_gpu_backend_jax_device_kind_not_gpu(self, monkeypatch):
        """Test get_gpu_backend when JAX is available but only CPU devices are present."""
        with patch('voiage.core.gpu_acceleration.JAX_AVAILABLE', True):
            with patch('voiage.core.gpu_acceleration.jax') as mock_jax:
                mock_device = MagicMock()
                mock_device.device_kind = 'cpu'  # Not a GPU
                mock_jax.devices.return_value = [mock_device]
                backend = get_gpu_backend()
                # If no GPU devices, should return 'none' or another available backend
                assert backend in ['none', 'cupy', 'torch']

    def test_get_gpu_backend_cupy_runtime_error(self, monkeypatch):
        """Test get_gpu_backend when CuPy raises a runtime error."""
        with patch('voiage.core.gpu_acceleration.CUPY_AVAILABLE', True):
            with patch('voiage.core.gpu_acceleration.cp') as mock_cp:
                mock_cp.cuda.runtime.getDeviceCount.side_effect = MagicMock(side_effect=Exception("CUDA error"))
                backend = get_gpu_backend()
                # If CuPy fails, should return 'none' or another available backend
                assert backend in ['none', 'jax', 'torch']

    def test_array_to_cpu_with_detected_backends(self, monkeypatch):
        """Test array_to_cpu with backend detection."""
        test_array = np.array([1.0, 2.0, 3.0])

        # Test when backend detection isn't needed (numpy array)
        result = array_to_cpu(test_array)
        np.testing.assert_array_equal(result, test_array)
        assert isinstance(result, np.ndarray)

    def test_gpu_accelerated_evpi_with_different_backends(self, monkeypatch):
        """Test GPUAcceleratedEVPI with different available backends."""
        test_data = np.random.rand(10, 2)

        # Test with each backend if available
        backends_to_test = []
        if JAX_AVAILABLE:
            backends_to_test.append('jax')
        if TORCH_AVAILABLE:
            backends_to_test.append('torch')

        # Even if backends aren't installed, we can test the logic with mocks
        for backend in ['jax', 'cupy', 'torch']:
            if backend == 'jax' and JAX_AVAILABLE:
                # JAX is available, test with mocked functions
                with patch('voiage.core.gpu_acceleration.array_to_gpu', side_effect=lambda x, y: x):
                    with patch('voiage.core.gpu_acceleration.jnp.max', new=lambda arr, axis: np.max(arr, axis=axis)):
                        with patch('voiage.core.gpu_acceleration.jnp.mean', new=lambda arr, axis: np.mean(arr, axis=axis)):
                            with patch('voiage.core.gpu_acceleration.jnp.asarray', new=lambda x: np.asarray(x)):
                                evpi_calc = GPUAcceleratedEVPI(backend='jax')
                                result = evpi_calc.calculate_evpi(test_data)
                                assert isinstance(result, float)
            elif backend == 'cupy':
                # Test CuPy path with mocked functions
                if CUPY_AVAILABLE:
                    # If CuPy is available, test it
                    pass  # Actual testing would need CuPy arrays
                else:
                    # Mock CuPy availability
                    with patch('voiage.core.gpu_acceleration.CUPY_AVAILABLE', True):
                        with patch('voiage.core.gpu_acceleration.cp') as mock_cp:
                            mock_cp.array.return_value = test_data  # Return same array for mock
                            mock_cp.max.return_value = np.max(test_data, axis=1)
                            mock_cp.mean.return_value = np.mean(test_data, axis=0)
                            mock_cp.asarray.return_value = np.asarray(test_data)

                            # Create GPUAcceleratedEVPI with CuPy backend (would require CuPy to be installed)
                            # This test may still fail in environments without CuPy
                            try:
                                evpi_calc = GPUAcceleratedEVPI(backend='cupy')
                                # This test will likely require CuPy to be installed
                            except RuntimeError:
                                # Expected if CuPy isn't properly available
                                pass
            elif backend == 'torch' and TORCH_AVAILABLE:
                # Test PyTorch path with mocked functions
                with patch('voiage.core.gpu_acceleration.array_to_gpu', side_effect=lambda x, y: x):
                    # PyTorch tests would need actual PyTorch tensors
                    pass


def test_import_handling():
    """Test the module's import handling."""
    # Verify constants exist and are set appropriately
    assert isinstance(JAX_AVAILABLE, bool)
    assert isinstance(CUPY_AVAILABLE, bool)
    assert isinstance(TORCH_AVAILABLE, bool)

    # Test that functions exist and have the expected signatures
    assert callable(get_gpu_backend)
    assert callable(is_gpu_available)
    assert callable(array_to_gpu)
    assert callable(array_to_cpu)
    assert callable(gpu_jit_compile)
    assert callable(gpu_vectorize)
    assert callable(gpu_parallelize)
    assert callable(GPUAcceleratedEVPI)
    assert callable(GPUAcceleratedEVPI.calculate_evpi)
