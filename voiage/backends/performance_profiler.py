"""Performance profiling and optimization tools."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class JaxPerformanceProfiler:
    """Profile and optimize JAX computations."""

    def __init__(self) -> None:
        self.profiles: dict[str, list[float]] = {}
        self.timings: dict[str, list[float]] = {}

    def profile_function(self, func: Callable[..., object]) -> Callable[..., object]:
        """Profile function execution time."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            func_name = func.__name__
            if func_name not in self.timings:
                self.timings[func_name] = []
            self.timings[func_name].append(end_time - start_time)

            return result

        return wrapper

    def compare_implementations(
        self,
        numpy_func: Callable[..., object],
        jax_func: Callable[..., object],
        test_data: tuple[object, ...],
        n_runs: int = 10,
    ) -> dict[str, list[float]]:
        """Compare NumPy vs JAX implementations."""
        results: dict[str, list[float]] = {
            "numpy_times": [],
            "jax_times": [],
            "speedups": [],
        }

        # Warm up JAX
        jax_func(*test_data)

        for _i in range(n_runs):
            # Time NumPy
            start = time.time()
            _ = numpy_func(*test_data)
            numpy_time = time.time() - start
            results["numpy_times"].append(numpy_time)

            # Time JAX
            start = time.time()
            _ = jax_func(*test_data)
            jax_time = time.time() - start
            results["jax_times"].append(jax_time)

            # Calculate speedup
            speedup = numpy_time / jax_time if jax_time > 0 else 0
            results["speedups"].append(speedup)

        return results

    def memory_usage_analysis(
        self,
        func: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        """Analyze memory usage of a function."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run function
        result = func(*args, **kwargs)

        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_increase_mb": memory_after - memory_before,
            "result": result,
        }

    def get_performance_report(self) -> dict[str, dict[str, object]]:
        """Generate performance report."""
        report: dict[str, dict[str, object]] = {}
        for func_name, times in self.timings.items():
            report[func_name] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "calls": len(times),
            }
        return report
