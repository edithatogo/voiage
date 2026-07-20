"""Execution adapter abstractions for local and distributed CPU scheduling."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from threading import Thread
from typing import TYPE_CHECKING, Protocol, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

_Result = TypeVar("_Result")


class ExecutionAdapter(Protocol):
    """A scheduler-facing adapter that can provide an executor."""

    name: str

    def create_executor(self, n_workers: int) -> Executor:
        """Create an executor for a worker pool."""


@dataclass(frozen=True, slots=True)
class LocalProcessAdapter:
    """Use local process pools for CPU fan-out."""

    name: str = "local-process"

    def create_executor(self, n_workers: int) -> Executor:
        """Create a local process-pool executor."""
        return ProcessPoolExecutor(max_workers=n_workers)


@dataclass(frozen=True, slots=True)
class LocalThreadAdapter:
    """Use local thread pools for CPU fan-out."""

    name: str = "local-thread"

    def create_executor(self, n_workers: int) -> Executor:
        """Create a local thread-pool executor."""
        return ThreadPoolExecutor(max_workers=n_workers)


@dataclass(frozen=True, slots=True)
class DaskClusterAdapter:
    """Use a Dask distributed client if the optional dependency is available."""

    scheduler_address: str | None = None
    local_cluster: bool = True
    name: str = "dask-cluster"

    def create_executor(self, n_workers: int) -> Executor:
        """Create an executor backed by Dask."""
        try:
            from dask.distributed import Client, LocalCluster
        except Exception as exc:  # pragma: no cover - soft dependency gate
            raise ImportError(
                "dask.distributed is required for the Dask cluster adapter"
            ) from exc

        if self.scheduler_address:
            client = Client(self.scheduler_address)
        else:
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
        return _DaskExecutor(client)


class _DaskExecutor:
    """A tiny Executor-compatible wrapper around a Dask client."""

    def __init__(self, client: object) -> None:
        self._client = client

    def submit(self, fn: Callable[..., _Result], /, *args: object, **kwargs: object):
        return self._client.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


@dataclass(frozen=True, slots=True)
class RayClusterAdapter:
    """Use a Ray cluster if the optional dependency is available."""

    address: str | None = None
    name: str = "ray-cluster"

    def create_executor(self, n_workers: int) -> Executor:
        """Create an executor backed by Ray."""
        try:
            import ray
        except Exception as exc:  # pragma: no cover - soft dependency gate
            raise ImportError("ray is required for the Ray cluster adapter") from exc

        if not ray.is_initialized():
            init_kwargs: dict[str, object] = {}
            if self.address is not None:
                init_kwargs["address"] = self.address
            else:
                init_kwargs["num_cpus"] = n_workers
            ray.init(**init_kwargs)
        return _RayExecutor(ray)


class _RayExecutor:
    """A tiny Executor-compatible wrapper around Ray."""

    def __init__(self, ray_module: object) -> None:
        self._ray = ray_module

    def submit(self, fn: Callable[..., _Result], /, *args: object, **kwargs: object):
        from concurrent.futures import Future

        future: Future[_Result] = Future()

        @self._ray.remote  # type: ignore[attr-defined]
        def _run() -> _Result:
            return fn(*args, **kwargs)

        def _resolve() -> None:
            try:
                future.set_result(self._ray.get(_run.remote()))
            except Exception as exc:  # pragma: no cover - runtime path
                future.set_exception(exc)

        Thread(target=_resolve, daemon=True).start()
        return future

    def shutdown(self, wait: bool = True) -> None:
        if hasattr(self._ray, "shutdown"):
            self._ray.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


@dataclass(frozen=True, slots=True)
class FpgaClusterAdapter:
    """Explicit FPGA execution lane placeholder.

    The repository does not currently ship an FPGA runtime or toolchain
    integration. This adapter exists so the execution layer can expose a
    stable, explicit failure mode until a real backend is implemented.
    """

    name: str = "fpga-cluster"

    def create_executor(self, n_workers: int) -> Executor:
        """Raise until a real FPGA executor exists."""
        raise NotImplementedError(
            "FPGA execution is not implemented in voiage yet; "
            "the lane remains evidence-gated."
        )


@dataclass(frozen=True, slots=True)
class AsicClusterAdapter:
    """Explicit ASIC/custom-circuit execution lane placeholder.

    The repository does not currently ship an ASIC/custom-circuit runtime.
    This adapter keeps the contract visible without claiming unsupported
    hardware availability.
    """

    name: str = "asic-cluster"

    def create_executor(self, n_workers: int) -> Executor:
        """Raise until a real ASIC/custom-circuit executor exists."""
        raise NotImplementedError(
            "ASIC/custom-circuit execution is not implemented in voiage yet; "
            "the lane remains evidence-gated."
        )


def get_execution_adapter(name: str) -> ExecutionAdapter:
    """Resolve a named execution adapter."""
    normalized = name.strip().lower()
    if normalized in {"process", "processes", "local-process"}:
        return LocalProcessAdapter()
    if normalized in {"thread", "threads", "local-thread"}:
        return LocalThreadAdapter()
    if normalized in {"dask", "dask-cluster", "distributed"}:
        return DaskClusterAdapter()
    if normalized in {"ray", "ray-cluster"}:
        return RayClusterAdapter()
    if normalized in {"fpga", "fpga-cluster"}:
        return FpgaClusterAdapter()
    if normalized in {"asic", "asic-cluster"}:
        return AsicClusterAdapter()
    raise ValueError(f"Unknown execution adapter: {name}")


def available_execution_adapters() -> tuple[str, ...]:
    """Return the public scheduler names understood by the adapter layer."""
    return (
        "process",
        "thread",
        "dask",
        "ray",
        "fpga",
        "asic",
    )


def is_placeholder_execution_adapter(name: str) -> bool:
    """Return whether a scheduler name currently maps to a placeholder backend."""
    normalized = name.strip().lower()
    return normalized in {"fpga", "fpga-cluster", "asic", "asic-cluster"}


def build_executor_factory(
    adapter: ExecutionAdapter,
) -> Callable[[int, bool], Executor]:
    """Adapt a scheduler-facing adapter to the distributed executor factory signature."""

    def factory(n_workers: int, use_processes: bool) -> Executor:
        if use_processes:
            try:
                return adapter.create_executor(n_workers)
            except (BrokenProcessPool, OSError, RuntimeError):
                pass
        return ThreadPoolExecutor(max_workers=n_workers)

    return factory
