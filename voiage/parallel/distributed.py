"""Distributed and cluster-oriented execution helpers for HPC-style CPU workloads."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import multiprocessing as mp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np


@dataclass(frozen=True, slots=True)
class ClusterExecutionConfig:
    """Describe a CPU cluster or distributed execution topology."""

    n_nodes: int = 1
    workers_per_node: int | None = None
    use_processes: bool = True
    chunk_count: int | None = None
    scheduler: str = "process"
    scheduler_address: str | None = None

    def __post_init__(self) -> None:
        """Validate cluster sizing and scheduler configuration."""
        if self.n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if self.workers_per_node is not None and self.workers_per_node <= 0:
            raise ValueError("workers_per_node must be positive")
        if self.chunk_count is not None and self.chunk_count <= 0:
            raise ValueError("chunk_count must be positive")
        if not self.scheduler.strip():
            raise ValueError("scheduler must not be empty")

    @property
    def total_workers(self) -> int:
        """Return the effective worker count across all nodes."""
        per_node = self.workers_per_node or mp.cpu_count()
        return max(1, self.n_nodes * per_node)


def _default_executor(n_workers: int, use_processes: bool) -> Executor:
    if use_processes:
        try:
            return ProcessPoolExecutor(max_workers=n_workers)
        except (OSError, RuntimeError, BrokenProcessPool):
            pass
    return ThreadPoolExecutor(max_workers=n_workers)


def partition_workload(total_items: int, partition_count: int) -> list[slice]:
    """Split an item range into balanced slices."""
    if total_items < 0:
        raise ValueError("total_items must be non-negative")
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if total_items == 0:
        return []

    partition_count = min(total_items, partition_count)
    base, remainder = divmod(total_items, partition_count)
    slices: list[slice] = []
    start = 0
    for index in range(partition_count):
        stop = start + base + (1 if index < remainder else 0)
        slices.append(slice(start, stop))
        start = stop
    return slices


def distributed_map[Item, ChunkResult](
    items: Sequence[Item] | Iterable[Item],
    worker_func: Callable[[Item], ChunkResult],
    *,
    config: ClusterExecutionConfig | None = None,
    n_workers: int | None = None,
    use_processes: bool | None = None,
    executor_factory: Callable[[int, bool], Executor] | None = None,
) -> list[ChunkResult]:
    """Execute work across a CPU pool or a caller-supplied distributed executor.

    The call preserves input ordering, so the returned list is aligned to the
    incoming item sequence even when the underlying executor completes tasks out
    of order.
    """
    if config is None:
        config = ClusterExecutionConfig()

    worker_total = n_workers or config.total_workers
    process_mode = config.use_processes if use_processes is None else use_processes
    values = list(items)
    if not values:
        return []

    factory = executor_factory or _default_executor
    results: list[ChunkResult] = [None] * len(values)  # type: ignore[list-item]

    try:
        with factory(worker_total, process_mode) as executor:
            futures = [
                (index, executor.submit(worker_func, item))
                for index, item in enumerate(values)
            ]
            for index, future in futures:
                results[index] = future.result()
    except (BrokenProcessPool, OSError, RuntimeError, TypeError):
        for index, item in enumerate(values):
            results[index] = worker_func(item)

    return results


def distributed_reduce[Item, ChunkResult](
    items: Sequence[Item] | Iterable[Item],
    worker_func: Callable[[Item], ChunkResult],
    reducer: Callable[[Sequence[ChunkResult]], ChunkResult],
    *,
    config: ClusterExecutionConfig | None = None,
    n_workers: int | None = None,
    use_processes: bool | None = None,
    executor_factory: Callable[[int, bool], Executor] | None = None,
) -> ChunkResult:
    """Map distributed work and reduce it deterministically."""
    mapped = distributed_map(
        items,
        worker_func,
        config=config,
        n_workers=n_workers,
        use_processes=use_processes,
        executor_factory=executor_factory,
    )
    return reducer(mapped)


def distributed_chunk_map[Item, ChunkResult](
    chunks: Sequence[Sequence[Item]] | Iterable[Sequence[Item]],
    worker_func: Callable[[Sequence[Item]], ChunkResult],
    *,
    config: ClusterExecutionConfig | None = None,
    n_workers: int | None = None,
    use_processes: bool | None = None,
    executor_factory: Callable[[int, bool], Executor] | None = None,
) -> list[ChunkResult]:
    """Execute chunk-oriented work in input order across local or distributed CPU workers."""
    return distributed_map(
        list(chunks),
        worker_func,
        config=config,
        n_workers=n_workers,
        use_processes=use_processes,
        executor_factory=executor_factory,
    )


def partition_counts(total_items: int, partition_count: int) -> list[int]:
    """Return balanced counts for a workload partition."""
    return [s.stop - s.start for s in partition_workload(total_items, partition_count)]


def chunked_numpy_partition(data: np.ndarray, partition_count: int) -> list[np.ndarray]:
    """Partition a 1D numpy array into balanced views for distributed CPU work."""
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")
    return [
        data[slice_.start : slice_.stop]
        for slice_ in partition_workload(len(data), partition_count)
    ]
