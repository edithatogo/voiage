"""GPU acceleration utilities for Value of Information analysis."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np

from voiage.exceptions import raise_runtime_error, raise_value_error

# Try to import JAX for GPU acceleration
try:
    import jax
    from jax import jit, pmap, vmap
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    jit = None
    vmap = None
    pmap = None

# Try to import CuPy for GPU acceleration (alternative to JAX)
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 0,
                CUDARuntimeError=RuntimeError,
            )
        )
    )

# Try to import PyTorch for GPU acceleration
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def get_gpu_backend() -> str:
    """
    Get the available GPU backend.

    Returns
    -------
        str: Name of the available GPU backend ('jax', 'cupy', 'torch', or 'none')
    """
    if (
        JAX_AVAILABLE
        and jax.devices()
        and any(device.device_kind == "gpu" for device in jax.devices())
    ):
        return "jax"
    if CUPY_AVAILABLE:
        try:
            # Check if CUDA is available
            if cp.cuda.runtime.getDeviceCount() > 0:
                return "cupy"
        except Exception:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "torch"
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "torch"
    return "none"


def _validate_backend(backend: str) -> str:
    """Validate a backend name."""
    valid_backends = {"jax", "cupy", "torch", "none"}
    if backend not in valid_backends:
        raise_value_error(f"Unknown backend: {backend}")
    return backend


def is_gpu_available() -> bool:
    """
    Check if any GPU backend is available.

    Returns
    -------
        bool: True if GPU acceleration is available, False otherwise
    """
    return get_gpu_backend() != "none"


def array_to_gpu(arr: np.ndarray, backend: str | None = None) -> object:
    """
    Transfer a NumPy array to GPU memory using the specified or default backend.

    Args:
        arr: NumPy array to transfer to GPU
        backend: GPU backend to use ('jax', 'cupy', 'torch', or None for auto-detect)

    Returns
    -------
        Array on GPU (type depends on backend)

    Raises
    ------
        RuntimeError: If no GPU backend is available
    """
    if backend is None:
        backend = get_gpu_backend()
    backend = _validate_backend(backend)
    if backend == "none":
        raise_runtime_error("No GPU backend available")

    if backend == "jax":
        if not JAX_AVAILABLE:
            raise_runtime_error("JAX is not available")
        return jnp.array(arr)
    if backend == "cupy":
        if not CUPY_AVAILABLE:
            raise_runtime_error("CuPy is not available")
        return cp.array(arr)
    if backend == "torch":
        if not TORCH_AVAILABLE:
            raise_runtime_error("PyTorch is not available")
        return torch.tensor(arr, device="cuda")
    return raise_value_error(f"Unknown backend: {backend}")


def array_to_cpu(arr: object, backend: str | None = None) -> np.ndarray:
    """
    Transfer an array from GPU memory back to CPU (NumPy).

    Args:
        arr: Array on GPU (type depends on backend)
        backend: GPU backend that was used ('jax', 'cupy', 'torch', or None for auto-detect)

    Returns
    -------
        NumPy array on CPU
    """
    if backend is None:
        # Try to detect the backend from the array type
        if JAX_AVAILABLE and isinstance(arr, jax.Array):
            backend = "jax"
        elif CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
            backend = "cupy"
        elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor) and arr.is_cuda:
            backend = "torch"
        else:
            # If we can't detect the backend, assume it's already a NumPy array
            return np.asarray(arr)
    else:
        backend = _validate_backend(backend)

    if backend == "jax":
        if not JAX_AVAILABLE:
            raise_runtime_error("JAX is not available")
        return np.asarray(arr)
    if backend == "cupy":
        if not CUPY_AVAILABLE:
            raise_runtime_error("CuPy is not available")
        return np.asarray(cast("object", arr).get())  # type: ignore[union-attr]
    if backend == "torch":
        if not TORCH_AVAILABLE:
            raise_runtime_error("PyTorch is not available")
        return np.asarray(cast("object", arr).cpu().numpy())  # type: ignore[union-attr]
    return raise_value_error(f"Unknown backend: {backend}")


def gpu_jit_compile(
    func: Callable[..., object], backend: str | None = None
) -> Callable[..., object]:
    """
    JIT compile a function using the specified or default GPU backend.

    Args:
        func: Function to compile
        backend: GPU backend to use ('jax', 'torch', or None for auto-detect)

    Returns
    -------
        JIT compiled function
    """
    if backend is None:
        backend = get_gpu_backend()
        if backend == "none":
            # Return the original function if no GPU backend is available
            return func

    if backend == "jax":
        if not JAX_AVAILABLE:
            return func
        return cast("Callable[..., object]", jit(func))
    if backend == "torch":
        if not TORCH_AVAILABLE:
            return func

        # For PyTorch, we need to return a function that can be traced later
        # We can't trace without example inputs, so we'll return a wrapper
        def torch_jit_wrapper(*args: object, **kwargs: object) -> object:
            # This is a simplified approach - in practice, you'd want to trace with actual inputs
            return func(*args, **kwargs)

        return torch_jit_wrapper
    # CuPy doesn't have JIT compilation
    return func


def gpu_vectorize(
    func: Callable[..., object], backend: str | None = None
) -> Callable[..., object]:
    """
    Vectorize a function using the specified or default GPU backend.

    Args:
        func: Function to vectorize
        backend: GPU backend to use ('jax', 'cupy', 'torch', or None for auto-detect)

    Returns
    -------
        Vectorized function
    """
    if backend is None:
        backend = get_gpu_backend()
        if backend == "none":
            # Return the original function if no GPU backend is available
            return func

    if backend == "jax":
        if not JAX_AVAILABLE:
            return func
        return cast("Callable[..., object]", vmap(func))
    if backend in {"cupy", "torch"}:
        # For CuPy and PyTorch, we can use their native vectorization
        return func
    return func


def gpu_parallelize(
    func: Callable[..., object], backend: str | None = None
) -> Callable[..., object]:
    """
    Parallelize a function across multiple GPUs using the specified or default backend.

    Args:
        func: Function to parallelize
        backend: GPU backend to use ('jax', 'torch', or None for auto-detect)

    Returns
    -------
        Parallelized function
    """
    parallelized_func: Callable[..., object] = func

    if backend is None:
        backend = get_gpu_backend()
    if backend == "none":
        # Return the original function if no GPU backend is available
        return parallelized_func
    if backend == "jax":
        if not JAX_AVAILABLE:
            return parallelized_func
        parallelized_func = cast("Callable[..., object]", pmap(func))
    if backend == "torch":
        if not TORCH_AVAILABLE:
            return parallelized_func
        # For PyTorch, we can use DataParallel for simple parallelization
        if torch.cuda.device_count() > 1:
            parallelized_func = cast(
                "Callable[..., object]", torch.nn.DataParallel(func)
            )
    # CuPy doesn't have built-in parallelization across multiple GPUs
    return parallelized_func


class GPUAcceleratedEVPI:
    """GPU-accelerated Expected Value of Perfect Information (EVPI) calculator."""

    def __init__(self, backend: str | None = None) -> None:
        """
        Initialize the GPU-accelerated EVPI calculator.

        Args:
            backend: GPU backend to use ('jax', 'cupy', 'torch', or None for auto-detect)
        """
        self.backend = (
            get_gpu_backend() if backend is None else _validate_backend(backend)
        )
        if self.backend == "none":
            raise_runtime_error("No GPU backend available")

    def calculate_evpi(self, net_benefit_array: np.ndarray) -> float:
        """
        Calculate EVPI using GPU acceleration.

        Args:
            net_benefit_array: 2D array of net benefits (samples x strategies)

        Returns
        -------
            float: Calculated EVPI value
        """
        # Transfer data to GPU
        gpu_nb_array = array_to_gpu(net_benefit_array, self.backend)

        if self.backend == "jax":
            if not JAX_AVAILABLE:
                raise_runtime_error("JAX is not available")

            # Calculate the maximum net benefit for each parameter sample
            max_nb = jnp.max(gpu_nb_array, axis=1)

            # Calculate the expected net benefit for each decision option
            expected_nb_options = jnp.mean(gpu_nb_array, axis=0)

            # Find the maximum expected net benefit
            max_expected_nb = jnp.max(expected_nb_options, axis=0)

            # Calculate the expected maximum net benefit
            expected_max_nb = jnp.mean(max_nb, axis=0)

            # EVPI is the difference
            evpi = expected_max_nb - max_expected_nb

            # Transfer result back to CPU
            return float(np.asarray(evpi))

        if self.backend == "cupy":
            if not CUPY_AVAILABLE:
                raise_runtime_error("CuPy is not available")

            # Calculate the maximum net benefit for each parameter sample
            max_nb = cp.max(gpu_nb_array, axis=1)

            # Calculate the expected net benefit for each decision option
            expected_nb_options = cp.mean(gpu_nb_array, axis=0)

            # Find the maximum expected net benefit
            max_expected_nb = cp.max(expected_nb_options)

            # Calculate the expected maximum net benefit
            expected_max_nb = cp.mean(max_nb)

            # EVPI is the difference
            evpi = expected_max_nb - max_expected_nb

            # Transfer result back to CPU
            return float(evpi.get())

        if self.backend == "torch":
            if not TORCH_AVAILABLE:
                raise_runtime_error("PyTorch is not available")

            # Calculate the maximum net benefit for each parameter sample
            max_nb = torch.max(gpu_nb_array, dim=1).values

            # Calculate the expected net benefit for each decision option
            expected_nb_options = torch.mean(gpu_nb_array, dim=0)

            # Find the maximum expected net benefit
            max_expected_nb = torch.max(expected_nb_options)

            # Calculate the expected maximum net benefit
            expected_max_nb = torch.mean(max_nb)

            # EVPI is the difference
            evpi = expected_max_nb - max_expected_nb

            # Transfer result back to CPU
            return float(evpi.cpu().item())

        return raise_value_error(f"Unknown backend: {self.backend}")


# Example usage function
def example_gpu_acceleration() -> None:
    """Use GPU acceleration utilities."""
    if not is_gpu_available():
        print("No GPU backend available")
        return

    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(1000, 3).astype(np.float64)

    # Get the available backend
    backend = get_gpu_backend()
    print(f"Using backend: {backend}")

    # Transfer data to GPU
    gpu_array = array_to_gpu(net_benefit_array, backend)
    print(f"Array transferred to GPU: {type(gpu_array)}")

    # Transfer data back to CPU
    cpu_array = array_to_cpu(gpu_array, backend)
    print(f"Array transferred back to CPU: {type(cpu_array)}")

    # Calculate EVPI using GPU acceleration
    evpi_calculator = GPUAcceleratedEVPI(backend)
    evpi_result = evpi_calculator.calculate_evpi(net_benefit_array)
    print(f"EVPI calculated using GPU: {evpi_result}")


if __name__ == "__main__":  # pragma: no cover
    example_gpu_acceleration()
