# voiage backends package
"""Advanced JAX backends and utilities for voiage."""

# Import from the main_backends.py file
from ..main_backends import (
    JAX_AVAILABLE,
    AppleMetalBackend,
    Backend,
    JaxBackend,
    NumpyBackend,
    compile_phase_3_handoff_packet,
    benchmark_mps_vs_cpu,
    benchmark_evpi,
    benchmark_memory_throughput,
    get_backend,
    set_backend,
)

# Import advanced backends
from .advanced_jax_regression import JaxAdvancedRegression
from .gpu_acceleration import GpuAcceleration
from .performance_profiler import JaxPerformanceProfiler

__all__ = [
    "JAX_AVAILABLE",
    "AppleMetalBackend",
    "Backend",
    "GpuAcceleration",
    "JaxAdvancedRegression",
    "JaxBackend",
    "JaxPerformanceProfiler",
    "NumpyBackend",
    "benchmark_evpi",
    "compile_phase_3_handoff_packet",
    "benchmark_mps_vs_cpu",
    "benchmark_memory_throughput",
    "get_backend",
    "set_backend",
]
