# voiage backends package
"""Advanced JAX backends and utilities for voiage."""

# Import from the main_backends.py file
from ..main_backends import (
    get_backend,
    set_backend,
    JAX_AVAILABLE,
    JaxBackend,
    NumpyBackend,
    Backend
)

# Import advanced backends
from .advanced_jax_regression import JaxAdvancedRegression
from .gpu_acceleration import GpuAcceleration
from .performance_profiler import JaxPerformanceProfiler

__all__ = [
    'get_backend',
    'set_backend', 
    'JAX_AVAILABLE', 
    'JaxBackend', 
    'NumpyBackend', 
    'Backend',
    'JaxAdvancedRegression',
    'GpuAcceleration', 
    'JaxPerformanceProfiler'
]
