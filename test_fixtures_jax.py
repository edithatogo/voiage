#!/usr/bin/env python3
"""
JAX Test Fixtures for voiage
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# JAX fixtures
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

@pytest.fixture(scope="session")
def jax_devices():
    """Provide JAX devices information."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    devices = jax.devices()
    return {
        "cpu": [d for d in devices if "cpu" in str(d).lower()],
        "gpu": [d for d in devices if "gpu" in str(d).lower()],
        "all": devices
    }

@pytest.fixture
def jax_key():
    """Provide a JAX random key for reproducible tests."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    return jrandom.key(42)

@pytest.fixture
def jax_test_arrays(jax_key):
    """Provide JAX test arrays of different sizes."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")
    
    def create_arrays(size):
        return jrandom.normal(jax_key, (size, size), dtype=jnp.float64)
    
    return {
        "small": create_arrays(10),
        "medium": create_arrays(100),
        "large": create_arrays(1000)
    }

@pytest.fixture
def numpy_test_arrays():
    """Provide NumPy test arrays for comparison."""
    np.random.seed(42)
    
    def create_arrays(size):
        return np.random.randn(size, size).astype(np.float64)
    
    return {
        "small": create_arrays(10),
        "medium": create_arrays(100),
        "large": create_arrays(1000)
    }

@pytest.fixture
def comparison_data(jax_test_arrays, numpy_test_arrays):
    """Provide paired JAX and NumPy arrays for comparison testing."""
    return {
        "small": {
            "jax": jax_test_arrays["small"],
            "numpy": numpy_test_arrays["small"]
        },
        "medium": {
            "jax": jax_test_arrays["medium"],
            "numpy": numpy_test_arrays["medium"]
        },
        "large": {
            "jax": jax_test_arrays["large"],
            "numpy": numpy_test_arrays["large"]
        }
    }

@pytest.fixture
def evpi_test_data():
    """Provide test data for EVPI calculations."""
    np.random.seed(42)
    
    # Small dataset
    small_nb = np.random.randn(100, 4) * 100 + 1000
    
    # Medium dataset  
    medium_nb = np.random.randn(1000, 6) * 100 + 1000
    
    return {
        "small": small_nb,
        "medium": medium_nb,
        "option_names": ["Option A", "Option B", "Option C", "Option D", "Option E", "Option F"],
        "parameter_names": [f"param_{i}" for i in range(5)]
    }

class JaxTestHelper:
    """Helper class for JAX testing."""
    
    def __init__(self, jax_key):
        self.key = jax_key
    
    def create_test_array(self, shape, dtype=jnp.float64):
        """Create a test array with the given shape and dtype."""
        return jrandom.normal(self.key, shape, dtype=dtype)
    
    def assert_array_close(self, jax_array, numpy_array, rtol=1e-6, atol=1e-6):
        """Assert that JAX and NumPy arrays are close."""
        if JAX_AVAILABLE:
            jax_result = jnp.asarray(jax_array)
            np_result = numpy_array
        else:
            jax_result = jax_array
            np_result = numpy_array
        
        np.testing.assert_allclose(jax_result, np_result, rtol=rtol, atol=atol)
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function with timing."""
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        return {
            "result": result,
            "time": end_time - start_time
        }

@pytest.fixture
def jax_helper(jax_key):
    """Provide a JAX test helper instance."""
    return JaxTestHelper(jax_key)

# Skip JAX tests if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
