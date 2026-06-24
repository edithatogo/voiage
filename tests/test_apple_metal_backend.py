"""Tests for the optional Apple Metal backend."""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from voiage import main_backends
from voiage.backends import (
    benchmark_evpi,
    benchmark_memory_throughput,
    benchmark_mps_vs_cpu,
)


class _FakeTensor:
    def __init__(self, array: np.ndarray, device: object | None = None) -> None:
        self.array = np.asarray(array, dtype=float)
        self.device = device

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def item(self) -> float:
        return float(np.asarray(self.array).item())

    def __sub__(self, other: object) -> "_FakeTensor":
        other_array = other.array if isinstance(other, _FakeTensor) else other
        return _FakeTensor(
            self.array - np.asarray(other_array, dtype=float), self.device
        )


class _FakeMPSBackend:
    @staticmethod
    def is_built() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True


class _FakeTorch:
    float32 = np.float32

    class Backends:
        mps = _FakeMPSBackend()

    backends = Backends()

    @staticmethod
    def device(name: str) -> SimpleNamespace:
        return SimpleNamespace(type=name)

    @staticmethod
    def as_tensor(
        array: np.ndarray, dtype: object | None = None, device: object | None = None
    ) -> _FakeTensor:
        del dtype
        return _FakeTensor(np.asarray(array, dtype=float), device=device)

    @staticmethod
    def max(tensor: _FakeTensor, dim: int | None = None) -> object:
        if dim is None:
            return _FakeTensor(np.max(tensor.array), device=tensor.device)
        return SimpleNamespace(
            values=_FakeTensor(np.max(tensor.array, axis=dim), device=tensor.device)
        )

    @staticmethod
    def mean(tensor: _FakeTensor, dim: int | None = None) -> _FakeTensor:
        return _FakeTensor(np.mean(tensor.array, axis=dim), device=tensor.device)


@contextmanager
def _fake_apple_metal_backend() -> Iterator[None]:
    original_torch = main_backends.torch
    original_platform = main_backends.sys.platform
    try:
        main_backends.torch = _FakeTorch()
        main_backends.sys.platform = "darwin"
        yield
    finally:
        main_backends.torch = original_torch
        main_backends.sys.platform = original_platform


def test_apple_metal_backend_requires_torch() -> None:
    """The backend should raise when PyTorch/MPS is unavailable."""
    original_torch = main_backends.torch
    try:
        main_backends.torch = None
        with pytest.raises(ImportError):
            main_backends.get_backend("apple_metal")
    finally:
        main_backends.torch = original_torch


def test_apple_metal_backend_evpi_and_enbs_simple() -> None:
    """The backend should compute the same EVPI envelope on a fake MPS device."""
    with _fake_apple_metal_backend():
        backend = main_backends.get_backend("apple_metal")

        assert isinstance(backend, main_backends.AppleMetalBackend)
        assert backend.device.type == "mps"

        net_benefit_array = np.array([[10.0, 1.0], [2.0, 8.0]])

        assert backend.evpi(net_benefit_array) == pytest.approx(3.0)
        assert backend.enbs_simple(
            net_benefit_array, research_cost=1.0
        ) == pytest.approx(2.0)
        assert backend.enbs_simple_jit(
            net_benefit_array, research_cost=1.0
        ) == pytest.approx(2.0)


def test_benchmark_evpi_reports_device_and_throughput() -> None:
    """The benchmark helper should report a stable measurement envelope."""
    with _fake_apple_metal_backend():
        backend = main_backends.get_backend("apple_metal")
        payload = benchmark_evpi(
            backend,
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=10,
            warmup_runs=2,
        )

        assert payload["backend"] == "AppleMetalBackend"
        assert payload["device"] == "mps"
        assert payload["repeats"] == 10
        assert payload["warmup_runs"] == 2
        assert payload["result"] == pytest.approx(3.0)
        assert payload["mean_latency_ns"] > 0
        assert payload["throughput_ops_per_sec"] > 0


def test_benchmark_evpi_rejects_non_positive_repeats() -> None:
    """The benchmark helper should validate its repeat count."""
    backend = main_backends.NumpyBackend()

    with pytest.raises(ValueError):
        benchmark_evpi(backend, np.array([[10.0, 1.0], [2.0, 8.0]]), repeats=0)


def test_benchmark_memory_throughput_reports_samples_and_summary() -> None:
    """The memory-throughput helper should mirror the committed artifact shape."""
    with _fake_apple_metal_backend():
        backend = main_backends.get_backend("apple_metal")
        payload = benchmark_memory_throughput(
            backend,
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=10,
            warmup_runs=1,
        )

        assert payload["backend"] == "AppleMetalBackend"
        assert payload["device"] == "mps"
        assert payload["repeats"] == 10
        assert payload["warmup_runs"] == 1

        samples = payload["samples"]
        assert len(samples) == 3
        assert [sample["phase"] for sample in samples] == ["cold", "warm", "warm"]
        assert [sample["iteration"] for sample in samples] == [0, 1, 2]
        assert all(sample["latency_ns"] > 0 for sample in samples)
        assert all(sample["throughput_ops_per_sec"] > 0 for sample in samples)

        summary = payload["summary"]
        assert summary["evpi"] == pytest.approx(3.0)
        assert summary["cold_start_latency_ns"] > 0
        assert summary["warm_start_latency_ns"] > 0
        assert summary["mean_latency_ns"] > 0
        assert summary["throughput_ops_per_sec"] > 0
        assert payload["mean_latency_ns"] == summary["mean_latency_ns"]
        assert payload["throughput_ops_per_sec"] == summary["throughput_ops_per_sec"]


def test_benchmark_memory_throughput_works_on_cpu_backend() -> None:
    """The memory-throughput helper should remain CPU-fallback safe."""
    payload = benchmark_memory_throughput(
        main_backends.NumpyBackend(),
        np.array([[10.0, 1.0], [2.0, 8.0]]),
        repeats=10,
        warmup_runs=1,
    )

    assert payload["backend"] == "NumpyBackend"
    assert payload["device"] is None
    assert len(payload["samples"]) == 3
    assert payload["summary"]["evpi"] == pytest.approx(3.0)


def test_benchmark_mps_vs_cpu_memory_reports_available_comparison() -> None:
    """The comparison workflow should include CPU and Apple Metal payloads when available."""
    with _fake_apple_metal_backend():
        payload = benchmark_mps_vs_cpu(
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=10,
            warmup_runs=1,
            benchmark=benchmark_memory_throughput,
        )

        assert payload["backend"] == "apple_metal_vs_cpu"
        assert payload["workflow"] == "apple_metal_vs_cpu"
        assert payload["comparison"]["enabled"] is True
        assert payload["comparison"]["apple_metal_backend"] == "AppleMetalBackend"
        assert payload["comparison"]["apple_metal_device"] == "mps"
        assert payload["cpu"]["device"] is None
        assert payload["apple_metal"]["device"] == "mps"

        assert payload["comparison"]["result_delta"] == pytest.approx(0.0)
        assert payload["comparison"]["mean_latency_speedup"] >= 0.0
        assert payload["comparison"]["throughput_speedup"] >= 0.0
        assert payload["payload_version"] == "1.0.0"
        assert payload["workflow"] == "apple_metal_vs_cpu"
        assert payload["workload"]["shape"] == [2, 2]
        assert len(payload["workload"]["sha256"]) == 64
        assert payload["runtime"]["backend"]["torch"] is None
        assert payload["review"]["status"] == "device_comparison_available"
        assert payload["review"]["phase"] == "phase_3"
        assert "required_fields" in payload["review"]
        assert payload["review"]["required_fields"] == [
            "backend",
            "device",
            "workload.shape",
            "workload.sha256",
            "repeats",
            "warmup_runs",
            "mean_latency_ns",
            "throughput_ops_per_sec",
        ]


def test_benchmark_mps_vs_cpu_uses_generic_benchmark_hook() -> None:
    """Any benchmark hook should be swappable for the same comparison workflow."""
    with _fake_apple_metal_backend():
        payload = benchmark_mps_vs_cpu(
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=8,
            warmup_runs=1,
            benchmark=benchmark_evpi,
        )

        assert payload["benchmark"] == "benchmark_evpi"
        assert payload["comparison"]["enabled"] is True
        assert "mean_latency_ns" in payload["cpu"]
        assert payload["cpu"]["backend"] == "NumpyBackend"
        assert payload["apple_metal"]["backend"] == "AppleMetalBackend"


def test_memory_benchmark_payload_includes_required_flat_fields() -> None:
    """Memory throughput payloads should expose required fields at the top level."""
    with _fake_apple_metal_backend():
        payload = benchmark_memory_throughput(
            main_backends.get_backend("apple_metal"),
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=10,
            warmup_runs=1,
        )

    assert payload["mean_latency_ns"] == payload["summary"]["mean_latency_ns"]
    assert (
        payload["throughput_ops_per_sec"]
        == payload["summary"]["throughput_ops_per_sec"]
    )


def test_benchmark_mps_vs_cpu_reports_unavailable_without_torch() -> None:
    """The workflow should remain CPU-only when Apple Metal cannot be constructed."""
    original_torch = main_backends.torch
    try:
        main_backends.torch = None
        payload = benchmark_mps_vs_cpu(
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=10,
            warmup_runs=1,
            benchmark=benchmark_memory_throughput,
        )

        assert payload["comparison"]["enabled"] is False
        assert payload["apple_metal"] is None
        assert payload["comparison"]["apple_metal_backend"] is None
        assert payload["comparison"]["apple_metal_device"] is None
        assert payload["comparison"]["result_delta"] == 0.0
        assert payload["comparison"]["mean_latency_speedup"] == 0.0
        assert payload["comparison"]["throughput_speedup"] == 0.0
        assert payload["apple_metal_error"] == (
            "PyTorch is required for the Apple Metal backend"
        )
        assert payload["review"]["status"] == "cpu_reference_only"
        assert payload["review"]["phase"] == "phase_3"
        assert payload["review"]["required_fields"] == [
            "backend",
            "device",
            "workload.shape",
            "workload.sha256",
            "repeats",
            "warmup_runs",
            "mean_latency_ns",
            "throughput_ops_per_sec",
        ]
        assert payload["runtime"]["backend"]["torch"] is None
    finally:
        main_backends.torch = original_torch


def test_benchmark_mps_vs_cpu_reports_payload_checksum() -> None:
    """Workload hashing and runtime metadata must be present for handoff packets."""
    original_torch = main_backends.torch
    try:
        main_backends.torch = None
        payload = benchmark_mps_vs_cpu(
            np.array([[10.0, 1.0], [2.0, 8.0]], dtype=float),
            repeats=10,
            warmup_runs=1,
            benchmark=benchmark_evpi,
        )

        workload = payload["workload"]
        expected_workload = np.array([[10.0, 1.0], [2.0, 8.0]], dtype=float)
        expected_digest = hashlib.sha256(expected_workload.tobytes()).hexdigest()
        assert workload["shape"] == [2, 2]
        assert workload["dtype"] == "float64"
        assert workload["size"] == 4
        assert workload["nbytes"] == expected_workload.nbytes
        assert workload["sha256"] == expected_digest
        assert workload["min"] == 1.0
        assert workload["max"] == 10.0

        runtime = payload["runtime"]
        assert "platform" in runtime
        assert isinstance(runtime["system"], str)
    finally:
        main_backends.torch = original_torch


def test_benchmark_mps_vs_cpu_layout_invariant_signature() -> None:
    """The workload signature should be stable regardless of input memory layout."""
    original_torch = main_backends.torch
    try:
        main_backends.torch = None
        base = np.array([[10.0, 1.0], [2.0, 8.0]], dtype=np.float32, order="F")
        c_array = np.ascontiguousarray(base)
        f_array = np.array(base, order="F", copy=True)

        c_payload = benchmark_mps_vs_cpu(
            c_array, repeats=4, warmup_runs=0, benchmark=benchmark_evpi
        )
        f_payload = benchmark_mps_vs_cpu(
            f_array, repeats=4, warmup_runs=0, benchmark=benchmark_evpi
        )

        assert c_payload["workload"]["sha256"] == f_payload["workload"]["sha256"]
        assert c_payload["workload"]["dtype"] == "float64"
        assert f_payload["workload"]["dtype"] == "float64"
    finally:
        main_backends.torch = original_torch


def test_phase_3_handoff_compiles_scalar_and_memory_review_packets() -> None:
    """The Phase-3 handoff helper should combine scalar and memory review packets."""
    with _fake_apple_metal_backend():
        payload = main_backends.compile_phase_3_handoff_packet(
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=10,
            warmup_runs=1,
        )

        assert isinstance(payload, dict)
        payload = cast("dict[str, object]", payload)

        assert payload["payload_version"] == "1.0.0"
        assert payload["workflow"] == "apple_metal_phase_3_handoff"
        assert payload["review_phase"] == "phase_3"
        assert payload["review_context"] == "apple_metal_vs_cpu"
        assert payload["review"]["phase"] == "phase_3"
        assert payload["review"]["status"] == "device_comparison_available"

        for field in main_backends._PHASE_3_HARDENED_REQUIRED_FIELDS:
            assert field in payload

        benchmarks = cast("dict[str, object]", payload["benchmarks"])
        assert set(benchmarks.keys()) == {"scalar", "memory"}

        scalar_payload = cast("dict[str, object]", benchmarks["scalar"])
        memory_payload = cast("dict[str, object]", benchmarks["memory"])

        assert scalar_payload["review"]["phase"] == "phase_3"
        assert memory_payload["review"]["phase"] == "phase_3"
        assert scalar_payload["benchmark"] == "benchmark_evpi"
        assert memory_payload["benchmark"] == "benchmark_memory_throughput"
        assert set(scalar_payload["review"]["required_fields"]) == {
            "backend",
            "device",
            "workload.shape",
            "workload.sha256",
            "repeats",
            "warmup_runs",
            "mean_latency_ns",
            "throughput_ops_per_sec",
        }


def test_phase_3_handoff_can_emit_optional_json() -> None:
    """The helper should optionally emit a JSON representation."""
    payload = main_backends.compile_phase_3_handoff_packet(
        np.array([[10.0, 1.0], [2.0, 8.0]]),
        repeats=10,
        warmup_runs=1,
        as_json=True,
    )
    assert isinstance(payload, str)

    decoded = json.loads(payload)
    if (
        decoded["apple_metal_error"]["scalar"] is None
        and decoded["apple_metal_error"]["memory"] is None
    ):
        assert decoded["review"]["status"] == "device_comparison_available"
    else:
        assert decoded["review"]["status"] == "cpu_reference_only"
    assert decoded["benchmarks"]["memory"]["apple_metal_error"] is None
    assert decoded["benchmarks"]["scalar"]["apple_metal_error"] is None


def test_phase_3_handoff_review_has_strict_required_fields() -> None:
    """The review payload should include a strict required-fields declaration."""
    with _fake_apple_metal_backend():
        payload = main_backends.compile_phase_3_handoff_packet(
            np.array([[10.0, 1.0], [2.0, 8.0]]),
            repeats=10,
            warmup_runs=1,
        )

        required_fields = payload["review"]["required_fields"]
        assert isinstance(required_fields, list)
        assert len(required_fields) == len(set(required_fields))
        assert required_fields == [
            "payload_version",
            "workflow",
            "review.phase",
            "review.status",
            "review.required_fields",
            "runtime.platform",
            "runtime.system",
            "workload.shape",
            "workload.sha256",
            "benchmarks.scalar.payload_version",
            "benchmarks.scalar.workflow",
            "benchmarks.memory.payload_version",
            "benchmarks.memory.workflow",
            "benchmarks.scalar.repeats",
            "benchmarks.scalar.warmup_runs",
            "benchmarks.memory.repeats",
            "benchmarks.memory.warmup_runs",
        ]
