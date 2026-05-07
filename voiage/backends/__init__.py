# voiage backends package
"""Advanced JAX backends and utilities for voiage."""

# Import from the main_backends.py file
from ..main_backends import (
    JAX_AVAILABLE,
    Backend,
    JaxBackend,
    NumpyBackend,
    get_backend,
    set_backend,
)

# Import advanced backends
from .advanced_jax_regression import JaxAdvancedRegression
from .gpu_acceleration import GpuAcceleration
from .performance_profiler import JaxPerformanceProfiler

__all__ = [
    "JAX_AVAILABLE",
    "Backend",
    "GpuAcceleration",
    "JaxAdvancedRegression",
    "JaxBackend",
    "JaxPerformanceProfiler",
    "NumpyBackend",
    "get_backend",
    "set_backend",
]
