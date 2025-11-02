"""Tests for the DecisionAnalysis class."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray as NetBenefitArray


def test_evpi_method():
    """Test the EVPI method of DecisionAnalysis."""
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    value_array = NetBenefitArray.from_numpy(data)

    analysis = DecisionAnalysis(value_array)
    evpi_result = analysis.evpi()
    assert evpi_result >= 0


def test_evppi_method():
    """Test the EVPPI method of DecisionAnalysis."""
    # This test requires parameter samples, which would need more setup
    pass


def test_backend_selection():
    """Test that backend selection works correctly."""
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    value_array = NetBenefitArray.from_numpy(data)

    # Test with default (NumPy) backend
    analysis_numpy = DecisionAnalysis(value_array)
    evpi_numpy = analysis_numpy.evpi()
    assert evpi_numpy >= 0

    # Test with explicit NumPy backend
    analysis_numpy_explicit = DecisionAnalysis(value_array, backend="numpy")
    evpi_numpy_explicit = analysis_numpy_explicit.evpi()
    assert evpi_numpy_explicit >= 0

    # Verify they're the same
    assert evpi_numpy == evpi_numpy_explicit


# Test JAX backend if available
try:
    from voiage.backends import JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend():
    """Test that JAX backend works correctly."""
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    value_array = NetBenefitArray.from_numpy(data)

    # Test with JAX backend
    analysis_jax = DecisionAnalysis(value_array, backend="jax")
    evpi_jax = analysis_jax.evpi()
    assert evpi_jax >= 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_backend_consistency():
    """Test that NumPy and JAX backends produce consistent results."""
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    value_array = NetBenefitArray.from_numpy(data)

    # Test with NumPy backend
    analysis_numpy = DecisionAnalysis(value_array, backend="numpy")
    evpi_numpy = analysis_numpy.evpi()

    # Test with JAX backend
    analysis_jax = DecisionAnalysis(value_array, backend="jax")
    evpi_jax = analysis_jax.evpi()

    # Verify they're consistent (within floating point precision)
    # Use a more reasonable tolerance for JAX/NumPy differences
    assert abs(evpi_numpy - float(evpi_jax)) < 1e-6


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jit_compilation():
    """Test that JIT compilation works correctly."""
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(100, 3)
    value_array = NetBenefitArray.from_numpy(data)

    # Test with JAX backend without JIT
    analysis_jax_no_jit = DecisionAnalysis(value_array, backend="jax", use_jit=False)
    evpi_jax_no_jit = analysis_jax_no_jit.evpi()

    # Test with JAX backend with JIT
    analysis_jax_jit = DecisionAnalysis(value_array, backend="jax", use_jit=True)
    evpi_jax_jit = analysis_jax_jit.evpi()

    # Verify they're consistent (within floating point precision)
    assert abs(float(evpi_jax_no_jit) - float(evpi_jax_jit)) < 1e-10
