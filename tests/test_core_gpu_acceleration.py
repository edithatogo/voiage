"""Focused tests for core GPU acceleration helpers without requiring GPU hardware."""

from types import SimpleNamespace

import numpy as np
import pytest

from voiage.core import gpu_acceleration as gpu


class _FakeJaxDevice:
    def __init__(self, device_kind: str) -> None:
        self.device_kind = device_kind


class _FakeTorchTensor:
    def __init__(self, values: np.ndarray, is_cuda: bool = True) -> None:
        self.values = values
        self.is_cuda = is_cuda

    def cpu(self) -> "_FakeTorchTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeCuPyArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def get(self) -> np.ndarray:
        return self.values


def test_gpu_backend_detection_prefers_available_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend detection should handle JAX, CuPy, Torch, and no-GPU cases."""
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", True)
    monkeypatch.setattr(
        gpu, "jax", SimpleNamespace(devices=lambda: [_FakeJaxDevice("gpu")])
    )
    assert gpu.get_gpu_backend() == "jax"

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1))
        ),
    )
    assert gpu.get_gpu_backend() == "cupy"

    monkeypatch.setattr(
        gpu.cp.cuda.runtime,
        "getDeviceCount",
        lambda: (_ for _ in ()).throw(RuntimeError("cuda unavailable")),
    )
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert gpu.get_gpu_backend() == "torch"

    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", False)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    assert gpu.get_gpu_backend() == "none"
    assert gpu.is_gpu_available() is False


def test_array_to_gpu_validates_and_dispatches_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array transfer should validate backend names and dispatch to backend APIs."""
    arr = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Unknown backend"):
        gpu.array_to_gpu(arr, backend="bad")

    with pytest.raises(RuntimeError, match="No GPU backend available"):
        gpu.array_to_gpu(arr, backend="none")

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="JAX is not available"):
        gpu.array_to_gpu(arr, backend="jax")

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", True)
    monkeypatch.setattr(gpu, "jnp", SimpleNamespace(array=lambda value: ("jax", value)))
    assert gpu.array_to_gpu(arr, backend="jax") == ("jax", arr)

    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(gpu, "cp", SimpleNamespace(array=lambda value: ("cupy", value)))
    assert gpu.array_to_gpu(arr, backend="cupy") == ("cupy", arr)

    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(tensor=lambda value, device: ("torch", value, device)),
    )
    assert gpu.array_to_gpu(arr, backend="torch") == ("torch", arr, "cuda")


def test_array_to_cpu_detects_and_dispatches_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Array transfer back to CPU should handle explicit and auto-detected inputs."""
    arr = np.array([1.0, 2.0])

    assert np.array_equal(gpu.array_to_cpu(arr), arr)

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", True)
    assert np.array_equal(gpu.array_to_cpu(arr, backend="jax"), arr)

    cupy_array = _FakeCuPyArray(arr)
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)
    assert np.array_equal(gpu.array_to_cpu(cupy_array, backend="cupy"), arr)

    torch_tensor = _FakeTorchTensor(arr)
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    assert np.array_equal(gpu.array_to_cpu(torch_tensor, backend="torch"), arr)

    with pytest.raises(ValueError, match="Unknown backend"):
        gpu.array_to_cpu(arr, backend="bad")


def test_gpu_compile_vectorize_and_parallelize_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compilation helpers should fall back cleanly or use supplied backend hooks."""

    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "none")
    assert gpu.gpu_jit_compile(_double)(3) == 6
    assert gpu.gpu_vectorize(_double)(3) == 6
    assert gpu.gpu_parallelize(_double)(3) == 6

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", True)
    monkeypatch.setattr(gpu, "jit", lambda func: lambda value: func(value) + 1)
    monkeypatch.setattr(gpu, "vmap", lambda func: lambda value: func(value) + 2)
    monkeypatch.setattr(gpu, "pmap", lambda func: lambda value: func(value) + 3)
    assert gpu.gpu_jit_compile(_double, backend="jax")(3) == 7
    assert gpu.gpu_vectorize(_double, backend="jax")(3) == 8
    assert gpu.gpu_parallelize(_double, backend="jax")(3) == 9

    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    torch_wrapper = gpu.gpu_jit_compile(_double, backend="torch")
    assert torch_wrapper(3) == 6


def test_gpu_accelerated_evpi_jax_backend() -> None:
    """The JAX EVPI calculator should work without a physical GPU when requested."""
    pytest.importorskip("jax")
    values = np.array([[10.0, 1.0], [2.0, 8.0]], dtype=np.float64)

    calculator = gpu.GPUAcceleratedEVPI(backend="jax")

    assert calculator.calculate_evpi(values) == pytest.approx(3.0)


def test_example_gpu_acceleration_no_backend_prints_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The example helper should return early when no GPU backend is available."""
    monkeypatch.setattr(gpu, "is_gpu_available", lambda: False)

    gpu.example_gpu_acceleration()

    assert capsys.readouterr().out == "No GPU backend available\n"
