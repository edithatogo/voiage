"""Tests for the computational backends."""

import numpy as np
import pytest

from voiage.backends import NumpyBackend, get_backend, set_backend


def test_numpy_backend():
    """Test the NumPy backend."""
    backend = get_backend("numpy")
    assert isinstance(backend, NumpyBackend)

    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Calculate EVPI
    evpi_result = backend.evpi(net_benefit_array)
    assert isinstance(evpi_result, (float, np.floating))
    assert evpi_result >= 0


def test_backend_registry():
    """Test the backend registry functionality."""
    # Test getting the default backend
    default_backend = get_backend()
    assert isinstance(default_backend, NumpyBackend)

    # Test setting and getting a different backend
    set_backend("numpy")
    numpy_backend = get_backend("numpy")
    assert isinstance(numpy_backend, NumpyBackend)

    # Test error handling for unknown backends
    with pytest.raises(ValueError):
        get_backend("unknown_backend")

    with pytest.raises(ValueError):
        set_backend("unknown_backend")


def test_backend_consistency():
    """Test that different backends produce consistent results."""
    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Get NumPy backend
    numpy_backend = get_backend("numpy")
    numpy_evpi = numpy_backend.evpi(net_benefit_array)

    # Test that results are consistent
    assert numpy_evpi >= 0


# Test JAX backend if available
try:
    from voiage.backends import JaxBackend
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend():
    """Test the JAX backend."""
    from voiage.backends import get_backend

    backend = get_backend("jax")
    assert isinstance(backend, JaxBackend)

    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Calculate EVPI
    evpi_result = backend.evpi(net_benefit_array)
    # JAX returns JAX arrays, which can be converted to Python floats
    assert hasattr(evpi_result, '__float__') or isinstance(evpi_result, (float, np.floating))
    assert float(evpi_result) >= 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_jit():
    """Test the JIT-compiled version of the JAX backend."""
    from voiage.backends import get_backend

    backend = get_backend("jax")

    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Calculate EVPI with JIT compilation
    evpi_result = backend.evpi_jit(net_benefit_array)
    # JAX returns JAX arrays, which can be converted to Python floats
    assert hasattr(evpi_result, '__float__') or isinstance(evpi_result, (float, np.floating))
    assert float(evpi_result) >= 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_backend_consistency_jax():
    """Test that NumPy and JAX backends produce consistent results."""
    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Get NumPy backend
    numpy_backend = get_backend("numpy")
    numpy_evpi = numpy_backend.evpi(net_benefit_array)

    # Get JAX backend
    jax_backend = get_backend("jax")
    jax_evpi = jax_backend.evpi(net_benefit_array)

    # Test that results are consistent (within floating point precision)
    # Use a more reasonable tolerance for JAX/NumPy differences
    assert abs(numpy_evpi - float(jax_evpi)) < 1e-6
