# voiage backends package
"""Core backends plus lazily loaded optional JAX utilities."""

from importlib import import_module

# Import from the main_backends.py file
from ..main_backends import (
    JAX_AVAILABLE,
    AppleMetalBackend,
    Backend,
    JaxBackend,
    NumpyBackend,
    benchmark_evpi,
    benchmark_memory_throughput,
    benchmark_mps_vs_cpu,
    compile_phase_3_handoff_packet,
    get_backend,
    set_backend,
)

# Import advanced backends
from .advanced_jax_regression import JaxAdvancedRegression
from .performance_profiler import JaxPerformanceProfiler


def __getattr__(name: str) -> object:
    """Load the GPU utility only when the JAX extra is available."""
    if name != "GpuAcceleration":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = import_module(".gpu_acceleration", __name__).GpuAcceleration
    except ModuleNotFoundError as error:
        if not (error.name or "").startswith("jax") and not str(error).startswith(
            "JAX intentionally absent"
        ):
            raise
        from ..exceptions import raise_optional_dependency_error

        raise_optional_dependency_error(
            "GpuAcceleration requires JAX; install it with `pip install 'voiage[jax]'`."
        )
    globals()[name] = value
    return value


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
    "benchmark_memory_throughput",
    "benchmark_mps_vs_cpu",
    "compile_phase_3_handoff_packet",
    "get_backend",
    "set_backend",
]
