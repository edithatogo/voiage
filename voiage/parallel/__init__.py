"""Parallel processing utilities for Value of Information analysis."""

from .monte_carlo import (
    distributed_monte_carlo_simulation,
    parallel_bootstrap_sampling,
    parallel_evsi_calculation,
    parallel_monte_carlo_simulation,
)
from .distributed import (
    ClusterExecutionConfig,
    distributed_chunk_map,
    distributed_map,
    distributed_reduce,
    partition_counts,
    partition_workload,
)
from .adapters import (
    AsicClusterAdapter,
    ExecutionAdapter,
    FpgaClusterAdapter,
    LocalProcessAdapter,
    LocalThreadAdapter,
    available_execution_adapters,
    build_executor_factory,
    get_execution_adapter,
    is_placeholder_execution_adapter,
)

__all__ = [
    "parallel_bootstrap_sampling",
    "parallel_evsi_calculation",
    "parallel_monte_carlo_simulation",
    "distributed_monte_carlo_simulation",
    "ClusterExecutionConfig",
    "distributed_chunk_map",
    "distributed_map",
    "distributed_reduce",
    "partition_counts",
    "partition_workload",
    "AsicClusterAdapter",
    "ExecutionAdapter",
    "FpgaClusterAdapter",
    "LocalProcessAdapter",
    "LocalThreadAdapter",
    "available_execution_adapters",
    "build_executor_factory",
    "get_execution_adapter",
    "is_placeholder_execution_adapter",
]
