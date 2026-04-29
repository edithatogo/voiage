"""Regression tests for backend GPU acceleration helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from voiage.backends.advanced_integration import JaxAdvancedBackend
from voiage.backends.gpu_acceleration import GpuAcceleration


class _FakeDevice:
    def __init__(self, device_kind: str) -> None:
        self.device_kind = device_kind


def test_detect_gpu_filters_gpu_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("jax")

    monkeypatch.setattr(
        "voiage.backends.gpu_acceleration.jax.devices",
        lambda: [_FakeDevice("CPU"), _FakeDevice("NVIDIA GPU"), _FakeDevice("gpu-tpu")],
    )

    gpu_utils = GpuAcceleration()

    devices = gpu_utils.detect_gpu()

    assert [device.device_kind for device in devices] == ["NVIDIA GPU", "gpu-tpu"]


def test_get_memory_info_reports_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("jax")

    monkeypatch.setattr(
        GpuAcceleration,
        "detect_gpu",
        staticmethod(lambda: [_FakeDevice("GPU 0"), _FakeDevice("GPU 1")]),
    )

    gpu_utils = GpuAcceleration()

    info = gpu_utils.get_memory_info()

    assert info == {
        "gpu_available": True,
        "gpu_count": 2,
        "memory_info": "Available via jax.lib.xla_bridge",
    }


def test_get_memory_info_handles_detection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_: GpuAcceleration) -> list[_FakeDevice]:
        raise RuntimeError("boom")

    monkeypatch.setattr(GpuAcceleration, "detect_gpu", _raise)

    gpu_utils = GpuAcceleration()

    info = gpu_utils.get_memory_info()

    assert info == {
        "gpu_available": False,
        "gpu_count": 0,
        "memory_info": "Unable to query memory info",
    }


def test_optimize_for_gpu_returns_device_arrays_unchanged() -> None:
    pytest.importorskip("jax")

    gpu_utils = GpuAcceleration()
    device_array = SimpleNamespace(device_buffer=object())

    optimized = gpu_utils.optimize_for_gpu(device_array)

    assert optimized is device_array


def test_optimize_for_gpu_converts_cpu_inputs_to_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("jax")

    captured: dict[str, object] = {}

    def _asarray(data: object, dtype: object) -> np.ndarray:
        captured["data"] = data
        captured["dtype"] = dtype
        return np.asarray(data, dtype=np.float32)

    monkeypatch.setattr("voiage.backends.gpu_acceleration.jnp.asarray", _asarray)

    gpu_utils = GpuAcceleration()
    values = [1, 2, 3]

    optimized = gpu_utils.optimize_for_gpu(values)

    assert captured == {"data": values, "dtype": np.float32}
    assert isinstance(optimized, np.ndarray)
    assert optimized.dtype == np.float32
    assert optimized.tolist() == [1.0, 2.0, 3.0]


def test_memory_efficient_batch_process_flushes_when_threshold_is_exceeded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gpu_utils = GpuAcceleration()
    calls: list[list[np.ndarray]] = []

    def process_func(batch: list[np.ndarray]) -> list[np.ndarray]:
        calls.append(list(batch))
        return list(batch)

    batches = [np.ones((256, 768), dtype=np.float32) for _ in range(3)]

    result: object = gpu_utils.memory_efficient_batch_process(
        batches,
        process_func,
        max_memory_mb=1,
    )

    captured = capsys.readouterr()

    assert captured.out.splitlines() == [
        "Processing batch 1 to free memory",
        "Processing batch 2 to free memory",
    ]
    assert [len(batch) for batch in calls] == [1, 2, 3]
    assert all(
        np.array_equal(left, right)
        for batch in calls
        for left, right in zip(
            batch,
            batches[: len(batch)],
            strict=False,
        )
    )
    assert all(
        left is right
        for left, right in zip(
            cast("list[np.ndarray]", result),
            batches,
            strict=False,
        )
    )


def test_jax_advanced_backend_delegates_gpu_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("jax")

    backend = JaxAdvancedBackend()
    expected = {"gpu_available": True, "gpu_count": 1, "memory_info": "ok"}
    monkeypatch.setattr(backend.gpu_utils, "get_memory_info", lambda: expected)

    assert backend.get_gpu_info() == expected


def test_jax_advanced_backend_delegates_profile_evppi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("jax")

    backend = JaxAdvancedBackend()
    expected = {"analysis": "profiled"}
    captured: dict[str, object] = {}

    def _memory_usage_analysis(
        func: object, *args: object, **kwargs: object
    ) -> dict[str, str]:
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        backend.profiler, "memory_usage_analysis", _memory_usage_analysis
    )

    parameters_of_interest = ["parameter"]

    outcome = backend.profile_evppi(1, 2, parameters_of_interest)

    assert outcome == expected
    assert captured["func"] == backend.evppi_advanced
    assert captured["args"] == (1, 2, parameters_of_interest)
    assert captured["kwargs"] == {}
