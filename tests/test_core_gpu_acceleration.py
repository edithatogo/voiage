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


def test_array_to_gpu_backend_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.array([1.0, 2.0])

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "cupy")
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="CuPy is not available"):
        gpu.array_to_gpu(arr, backend="cupy")

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "torch")
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="PyTorch is not available"):
        gpu.array_to_gpu(arr, backend="torch")


def test_array_to_gpu_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.array([1.0, 2.0])
    monkeypatch.setattr(gpu, "_validate_backend", lambda backend: backend)
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        gpu.array_to_gpu(arr, backend="invalid")


def test_array_to_cpu_auto_detect_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = np.array([1.0, 2.0])

    class _FakeJaxArray:
        def __array__(self) -> np.ndarray:
            return expected

    class _FakeCuPyNdArray:
        def get(self) -> np.ndarray:
            return expected

    class _FakeTorchTensorWithIsCuda:
        is_cuda = True

        def cpu(self) -> "_FakeTorchTensorWithIsCuda":
            return self

        def numpy(self) -> np.ndarray:
            return expected

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", True)
    monkeypatch.setattr(gpu, "jax", SimpleNamespace(Array=_FakeJaxArray))
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(gpu, "cp", SimpleNamespace(ndarray=_FakeCuPyNdArray))
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu, "torch", SimpleNamespace(Tensor=_FakeTorchTensorWithIsCuda)
    )

    assert np.array_equal(gpu.array_to_cpu(_FakeJaxArray()), expected)
    assert np.array_equal(gpu.array_to_cpu(_FakeCuPyNdArray()), expected)
    assert np.array_equal(gpu.array_to_cpu(_FakeTorchTensorWithIsCuda()), expected)


def test_array_to_cpu_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.array([1.0, 2.0])
    monkeypatch.setattr(gpu, "_validate_backend", lambda backend: backend)
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        gpu.array_to_cpu(arr, backend="invalid")


def test_gpu_jit_compile_none_jax_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "none")
    assert gpu.gpu_jit_compile(_double)(3) == 6

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    assert gpu.gpu_jit_compile(_double, backend="jax")(3) == 6


def test_gpu_vectorize_none_jax_cupy_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "none")
    assert gpu.gpu_vectorize(_double)(3) == 6

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    assert gpu.gpu_vectorize(_double, backend="jax")(3) == 6

    assert gpu.gpu_vectorize(_double, backend="cupy")(3) == 6
    assert gpu.gpu_vectorize(_double, backend="torch")(3) == 6


def test_gpu_parallelize_torch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", False)
    assert gpu.gpu_parallelize(_double, backend="torch")(3) == 6

    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: 2),
            nn=SimpleNamespace(DataParallel=lambda f: f),
        ),
    )
    assert gpu.gpu_parallelize(_double, backend="torch")(3) == 6


def test_gpu_accelerated_evpi_jax_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)

    calculator = gpu.GPUAcceleratedEVPI()
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)
    with pytest.raises(RuntimeError, match="JAX is not available"):
        calculator.calculate_evpi(np.array([[1.0]]))


def test_gpu_accelerated_evpi_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "cupy")
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)

    class FakeCupy:
        def max(self, arr, axis=None):
            return FakeCupyArray(
                np.max([a.val if hasattr(a, "val") else a for a in arr], axis=axis)
                if axis is not None
                else np.max(
                    [
                        a.val if hasattr(a, "val") else a
                        for a in np.atleast_1d(arr).flatten()
                    ]
                )
            )

        def mean(self, arr, axis=None):
            return FakeCupyArray(
                np.mean([a.val if hasattr(a, "val") else a for a in arr], axis=axis)
                if axis is not None
                else np.mean(
                    [
                        a.val if hasattr(a, "val") else a
                        for a in np.atleast_1d(arr).flatten()
                    ]
                )
            )

    class FakeCupyArray:
        def __init__(self, val):
            self.val = val

        def get(self):
            return self.val

        def __sub__(self, other):
            return FakeCupyArray(self.val - other.val)

    monkeypatch.setattr(gpu, "cp", FakeCupy())
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)

    calculator = gpu.GPUAcceleratedEVPI()
    values = np.array([[10.0, 1.0], [2.0, 8.0]], dtype=np.float64)
    assert calculator.calculate_evpi(values) == pytest.approx(3.0)


def test_gpu_accelerated_evpi_cupy_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "cupy")
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", False)

    calculator = gpu.GPUAcceleratedEVPI()
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)
    with pytest.raises(RuntimeError, match="CuPy is not available"):
        calculator.calculate_evpi(np.array([[1.0]]))


def test_gpu_accelerated_evpi_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "torch")
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)

    class FakeTorchMaxRes:
        def __init__(self, val):
            self.values = FakeTorchTensor(val)

    class FakeTorch:
        def max(self, arr, dim=None):
            if dim is not None:
                return FakeTorchMaxRes(
                    np.max([a.val if hasattr(a, "val") else a for a in arr], axis=dim)
                )
            return FakeTorchTensor(
                np.max(
                    [
                        a.val if hasattr(a, "val") else a
                        for a in np.atleast_1d(arr).flatten()
                    ]
                )
            )

        def mean(self, arr, dim=None):
            return FakeTorchTensor(
                np.mean([a.val if hasattr(a, "val") else a for a in arr], axis=dim)
                if dim is not None
                else np.mean(
                    [
                        a.val if hasattr(a, "val") else a
                        for a in np.atleast_1d(arr).flatten()
                    ]
                )
            )

    class FakeTorchTensor:
        def __init__(self, val):
            self.val = val

        def cpu(self):
            return self

        def item(self):
            return float(self.val)

        def __sub__(self, other):
            return FakeTorchTensor(self.val - other.val)

    monkeypatch.setattr(gpu, "torch", FakeTorch())
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)

    calculator = gpu.GPUAcceleratedEVPI()
    values = np.array([[10.0, 1.0], [2.0, 8.0]], dtype=np.float64)
    assert calculator.calculate_evpi(values) == pytest.approx(3.0)


def test_gpu_accelerated_evpi_torch_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "torch")
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", False)

    calculator = gpu.GPUAcceleratedEVPI()
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)
    with pytest.raises(RuntimeError, match="PyTorch is not available"):
        calculator.calculate_evpi(np.array([[1.0]]))


def test_example_gpu_acceleration_full(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gpu, "is_gpu_available", lambda: True)
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)
    monkeypatch.setattr(gpu, "array_to_cpu", lambda arr, b: arr)

    class FakeCalculator:
        def calculate_evpi(self, data):
            return 42.0

    monkeypatch.setattr(gpu, "GPUAcceleratedEVPI", lambda backend: FakeCalculator())

    gpu.example_gpu_acceleration()
    out = capsys.readouterr().out
    assert "Using backend: jax" in out
    assert "EVPI calculated using GPU: 42.0" in out


def test_imports_handle_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    jax_orig = sys.modules.get("jax")
    cupy_orig = sys.modules.get("cupy")
    torch_orig = sys.modules.get("torch")

    sys.modules["jax"] = None
    sys.modules["cupy"] = None
    sys.modules["torch"] = None

    try:
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)
        assert not g.JAX_AVAILABLE
    finally:
        if jax_orig is not None:
            sys.modules["jax"] = jax_orig
        else:
            del sys.modules["jax"]

        if cupy_orig is not None:
            sys.modules["cupy"] = cupy_orig
        else:
            del sys.modules["cupy"]

        if torch_orig is not None:
            sys.modules["torch"] = torch_orig
        else:
            del sys.modules["torch"]

        import voiage.core.gpu_acceleration as g

        importlib.reload(g)


def test_calculate_evpi_unknown_backend_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Need to bypass validation but also not be one of the known ones
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    calculator = gpu.GPUAcceleratedEVPI(backend="jax")
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)

    # We set self.backend to "unknown"
    calculator.backend = "unknown"
    values = np.array([[10.0, 1.0], [2.0, 8.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        calculator.calculate_evpi(values)


def test_gpu_parallelize_torch_1_device(monkeypatch: pytest.MonkeyPatch) -> None:
    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: 1),
            nn=SimpleNamespace(DataParallel=lambda func: lambda value: func(value) + 1),
        ),
    )

    assert gpu.gpu_parallelize(_double, backend="torch")(3) == 6


def test_imports_handle_success_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    cupy_orig = sys.modules.get("cupy")
    sys.modules["cupy"] = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 1,
                CUDARuntimeError=RuntimeError,
            )
        )
    )

    try:
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)
        assert g.CUPY_AVAILABLE
    finally:
        if cupy_orig is not None:
            sys.modules["cupy"] = cupy_orig
        else:
            del sys.modules["cupy"]
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)


def test_imports_handle_success_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    torch_orig = sys.modules.get("torch")
    sys.modules["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
        )
    )

    try:
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)
        assert g.TORCH_AVAILABLE
    finally:
        if torch_orig is not None:
            sys.modules["torch"] = torch_orig
        else:
            del sys.modules["torch"]
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)


def test_array_to_gpu_raise_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.array([1.0, 2.0])
    # This is a bit of a trick, since we can't legitimately hit line 132 unless _validate_backend is bypassed
    monkeypatch.setattr(gpu, "_validate_backend", lambda backend: backend)
    with pytest.raises(ValueError, match="Unknown backend: bad_backend"):
        gpu.array_to_gpu(arr, backend="bad_backend")


def test_array_to_cpu_raise_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    arr = np.array([1.0, 2.0])
    monkeypatch.setattr(gpu, "_validate_backend", lambda backend: backend)
    with pytest.raises(ValueError, match="Unknown backend: bad_backend"):
        gpu.array_to_cpu(arr, backend="bad_backend")


def test_gpu_accelerated_evpi_raise_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu, "_validate_backend", lambda backend: backend)
    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", True)

    calculator = gpu.GPUAcceleratedEVPI(backend="jax")
    calculator.backend = "bad_backend"
    monkeypatch.setattr(gpu, "array_to_gpu", lambda arr, b: arr)

    with pytest.raises(ValueError, match="Unknown backend: bad_backend"):
        calculator.calculate_evpi(np.array([[1.0]]))


def test_gpu_jit_compile_auto_detect_not_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    assert gpu.gpu_jit_compile(_double)(3) == 6


def test_gpu_vectorize_auto_detect_not_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "jax")
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    assert gpu.gpu_vectorize(_double)(3) == 6


def test_get_gpu_backend_cupy_raises_and_torch_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CuPy throws exception, fallback to none if Torch is not available
    def _raise_runtime_error() -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=_raise_runtime_error)
            )
        ),
    )
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert gpu.get_gpu_backend() == "torch"


def test_get_gpu_backend_cupy_count_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # Testing branch 69->74: CuPy available, getDeviceCount <= 0, no exception
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
        ),
    )
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert gpu.get_gpu_backend() == "torch"


def test_get_gpu_backend_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test edge cases in get_gpu_backend detection."""
    # JAX available but no devices
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", True)
    monkeypatch.setattr(gpu, "jax", SimpleNamespace(devices=lambda: []))
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", False)
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", False)
    assert gpu.get_gpu_backend() == "none"

    # JAX available but no GPU devices
    monkeypatch.setattr(
        gpu, "jax", SimpleNamespace(devices=lambda: [_FakeJaxDevice("cpu")])
    )
    assert gpu.get_gpu_backend() == "none"

    # CuPy available but getDeviceCount returns 0
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
        ),
    )
    assert gpu.get_gpu_backend() == "none"

    # CuPy throws exception, fallback to Torch
    def _raise_runtime_error() -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        gpu,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=_raise_runtime_error)
            )
        ),
    )
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert gpu.get_gpu_backend() == "torch"

    # CuPy throws exception, fallback to none if Torch is not available
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", False)
    assert gpu.get_gpu_backend() == "none"

    # CuPy available but returns 0, fallback to Torch
    monkeypatch.setattr(
        gpu,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
        ),
    )
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        gpu,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    assert gpu.get_gpu_backend() == "torch"


def test_imports_handle_success_cupy_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # We test that imports can be mocked by changing sys.modules
    import importlib
    import sys

    # Save original modules
    cupy_orig = sys.modules.get("cupy")
    torch_orig = sys.modules.get("torch")

    # Force ImportError
    sys.modules["cupy"] = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 0,
                CUDARuntimeError=RuntimeError,
            )
        )
    )
    sys.modules["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
        )
    )

    try:
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)
        assert g.CUPY_AVAILABLE
        assert g.TORCH_AVAILABLE
    finally:
        if cupy_orig is not None:
            sys.modules["cupy"] = cupy_orig
        else:
            del sys.modules["cupy"]

        if torch_orig is not None:
            sys.modules["torch"] = torch_orig
        else:
            del sys.modules["torch"]

        import voiage.core.gpu_acceleration as g

        importlib.reload(g)


def test_imports_handle_success_jax_import2(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys
    from types import ModuleType

    jax_orig = sys.modules.get("jax")
    jnp_orig = sys.modules.get("jax.numpy")

    fake_jax = ModuleType("jax")
    fake_jax.jit = lambda func: func  # type: ignore[attr-defined]
    fake_jax.pmap = lambda func: func  # type: ignore[attr-defined]
    fake_jax.vmap = lambda func: func  # type: ignore[attr-defined]
    fake_jax.devices = lambda: [_FakeJaxDevice("cpu")]  # type: ignore[attr-defined]
    fake_jnp = ModuleType("jax.numpy")
    fake_jax.numpy = fake_jnp  # type: ignore[attr-defined]
    sys.modules["jax"] = fake_jax
    sys.modules["jax.numpy"] = fake_jnp

    try:
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)
        assert g.JAX_AVAILABLE
    finally:
        if jax_orig is not None:
            sys.modules["jax"] = jax_orig
        else:
            del sys.modules["jax"]

        if jnp_orig is not None:
            sys.modules["jax.numpy"] = jnp_orig
        else:
            del sys.modules["jax.numpy"]
        import voiage.core.gpu_acceleration as g

        importlib.reload(g)
