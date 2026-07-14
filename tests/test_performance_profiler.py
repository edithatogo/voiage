"""Tests for performance profiling utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voiage.backends.performance_profiler import JaxPerformanceProfiler


def test_profile_function_records_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrapped functions should preserve results and store elapsed time."""
    profiler = JaxPerformanceProfiler()
    clock_values = iter([10.0, 10.25])

    monkeypatch.setattr(
        "voiage.backends.performance_profiler.time.time",
        lambda: next(clock_values),
    )

    @profiler.profile_function
    def add_one(value: int) -> int:
        return value + 1

    assert add_one(4) == 5
    assert profiler.timings["add_one"] == [0.25]


def test_compare_implementations_collects_times_and_speedups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison should collect NumPy and JAX timings for each run."""
    profiler = JaxPerformanceProfiler()
    clock_values = iter([1.0, 1.4, 2.0, 2.2, 3.0, 3.6, 4.0, 4.3])
    warmup_calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        "voiage.backends.performance_profiler.time.time",
        lambda: next(clock_values),
    )

    def numpy_func(*args: object) -> object:
        return args

    def jax_func(*args: object) -> object:
        warmup_calls.append(args)
        return args

    results = profiler.compare_implementations(
        numpy_func=numpy_func,
        jax_func=jax_func,
        test_data=(1, 2, 3),
        n_runs=2,
    )

    assert warmup_calls == [(1, 2, 3), (1, 2, 3), (1, 2, 3)]
    assert results["numpy_times"] == pytest.approx([0.4, 0.6])
    assert results["jax_times"] == pytest.approx([0.2, 0.3])
    assert results["speedups"] == pytest.approx([2.0, 2.0])


def test_compare_implementations_handles_zero_jax_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero JAX runtime should produce a speedup of zero instead of dividing."""
    profiler = JaxPerformanceProfiler()
    clock_values = iter([5.0, 5.5, 6.0, 6.0])

    monkeypatch.setattr(
        "voiage.backends.performance_profiler.time.time",
        lambda: next(clock_values),
    )

    results = profiler.compare_implementations(
        numpy_func=lambda value: value,
        jax_func=lambda value: value,
        test_data=(7,),
        n_runs=1,
    )

    assert results["numpy_times"] == pytest.approx([0.5])
    assert results["jax_times"] == pytest.approx([0.0])
    assert results["speedups"] == [0]


def test_memory_usage_analysis_returns_memory_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory usage analysis should report before, after, and function result."""
    profiler = JaxPerformanceProfiler()

    class FakeProcess:
        def __init__(self) -> None:
            self._rss_values = iter([10 * 1024 * 1024, 16 * 1024 * 1024])

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=next(self._rss_values))

    monkeypatch.setattr("os.getpid", lambda: 1234)
    monkeypatch.setattr("psutil.Process", lambda pid: FakeProcess())

    result = profiler.memory_usage_analysis(lambda x, y: x + y, 2, 5)

    assert result == {
        "memory_before_mb": 10.0,
        "memory_after_mb": 16.0,
        "memory_increase_mb": 6.0,
        "result": 7,
    }


def test_get_performance_report_summarizes_timings() -> None:
    """Performance report should summarize each tracked function."""
    profiler = JaxPerformanceProfiler()
    profiler.timings = {
        "alpha": [0.1, 0.2, 0.3],
        "beta": [1.0],
    }

    report = profiler.get_performance_report()

    assert report["alpha"]["calls"] == 3
    assert report["alpha"]["mean_time"] == pytest.approx(0.2)
    assert report["alpha"]["std_time"] == pytest.approx(0.0816496580927726)
    assert report["alpha"]["min_time"] == pytest.approx(0.1)
    assert report["alpha"]["max_time"] == pytest.approx(0.3)
    assert report["beta"] == {
        "mean_time": pytest.approx(1.0),
        "std_time": pytest.approx(0.0),
        "min_time": pytest.approx(1.0),
        "max_time": pytest.approx(1.0),
        "calls": 1,
    }
