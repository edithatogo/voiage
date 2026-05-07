"""Integration of advanced features with the main JAX backend."""

from collections.abc import Mapping, Sequence

import numpy as np

from .enhanced_jax_backend import EnhancedJaxBackend
from .gpu_acceleration import GpuAcceleration
from .performance_profiler import JaxPerformanceProfiler


class JaxAdvancedBackend(EnhancedJaxBackend):
    """Extended JAX backend with advanced features."""

    def __init__(self) -> None:
        super().__init__()
        self.gpu_utils = GpuAcceleration()
        self.profiler = JaxPerformanceProfiler()

    def evppi_advanced(
        self,
        net_benefit_array: np.ndarray,
        parameter_samples: np.ndarray | Mapping[str, np.ndarray],
        parameters_of_interest: Sequence[str],
        method: str = "polynomial",
        degree: int = 2,
        cv_folds: int = 5,
        regularization: float = 1e-6,
    ) -> float:
        """Advanced EVPPI calculation with enhanced regression."""
        return super().evppi_advanced(
            net_benefit_array,
            parameter_samples,
            parameters_of_interest,
            method,
            degree,
            cv_folds,
            regularization,
        )

    def get_gpu_info(self) -> object:
        """Get GPU information for optimization."""
        return self.gpu_utils.get_memory_info()

    def profile_evppi(
        self,
        net_benefit_array: np.ndarray,
        parameter_samples: np.ndarray | Mapping[str, np.ndarray],
        parameters_of_interest: Sequence[str],
    ) -> dict[str, object]:
        """Profile EVPPI calculation performance."""
        return self.profiler.memory_usage_analysis(
            self.evppi_advanced,
            net_benefit_array,
            parameter_samples,
            parameters_of_interest,
        )
