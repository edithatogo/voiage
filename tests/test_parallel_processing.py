"""Tests for parallel processing utilities in Value of Information analysis."""

import numpy as np
import pytest
import xarray as xr

from voiage.parallel import monte_carlo
from voiage.parallel.adapters import (
    AsicClusterAdapter,
    DaskClusterAdapter,
    FpgaClusterAdapter,
    LocalProcessAdapter,
    LocalThreadAdapter,
    RayClusterAdapter,
    available_execution_adapters,
    build_executor_factory,
    get_execution_adapter,
    is_placeholder_execution_adapter,
)
from voiage.parallel.distributed import (
    ClusterExecutionConfig,
    distributed_chunk_map,
    distributed_map,
    distributed_reduce,
    partition_counts,
    partition_workload,
)
from voiage.parallel.monte_carlo import (
    _monte_carlo_worker,
    distributed_monte_carlo_simulation,
    parallel_bootstrap_sampling,
    parallel_evsi_calculation,
    parallel_monte_carlo_simulation,
)
from voiage.schema import DecisionOption, ParameterSet, TrialDesign, ValueArray


def simple_model_func(params):
    """Model simple economic scenario for testing."""
    # Extract parameters
    if hasattr(params, "parameters"):
        mean_treatment = params.parameters.get("mean_treatment", np.array([0.0]))
        mean_control = params.parameters.get("mean_control", np.array([0.0]))
    else:
        # Handle case where params might be a dict or other structure
        mean_treatment = params.get("mean_treatment", np.array([0.0]))
        mean_control = params.get("mean_control", np.array([0.0]))

    # Ensure we're working with arrays
    if not isinstance(mean_treatment, np.ndarray):
        mean_treatment = np.array([mean_treatment])
    if not isinstance(mean_control, np.ndarray):
        mean_control = np.array([mean_control])

    # Calculate net benefits (simplified)
    # Assuming treatment is better when mean is higher
    nb_treatment = mean_treatment
    nb_control = mean_control

    # Stack to create ValueArray-compatible structure
    nb_values = np.column_stack([nb_control, nb_treatment]).astype(np.float64)
    return ValueArray.from_numpy(nb_values)


def mean_statistic(sample):
    """Calculate simple statistic for bootstrap testing."""
    return np.mean(sample)


def test_monte_carlo_worker() -> None:
    """Test the Monte Carlo worker function."""
    # Create simple parameter set
    params_data = {
        "mean_treatment": np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        "mean_control": np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        "sd_outcome": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data["mean_treatment"]))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create simple trial design
    trial_design = TrialDesign(
        arms=[
            DecisionOption(name="Treatment", sample_size=50),
            DecisionOption(name="Control", sample_size=50),
        ]
    )

    # Test the worker function
    expected_max_nb, n_processed = _monte_carlo_worker(
        worker_id=0,
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=10,
        seed_offset=0,
    )

    # Check that we got valid results
    assert isinstance(expected_max_nb, float)
    assert n_processed == 10
    assert expected_max_nb >= 0


def test_parallel_monte_carlo_simulation() -> None:
    """Test parallel Monte Carlo simulation."""
    # Create parameter set
    params_data = {
        "mean_treatment": np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        "mean_control": np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        "sd_outcome": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data["mean_treatment"]))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create trial design
    trial_design = TrialDesign(
        arms=[
            DecisionOption(name="Treatment", sample_size=50),
            DecisionOption(name="Control", sample_size=50),
        ]
    )

    # Test parallel Monte Carlo simulation
    result = parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=20,
        n_workers=2,
        use_processes=True,
    )

    # Check that we got a valid result
    assert isinstance(result, float)
    assert result >= 0


def test_parallel_evsi_calculation() -> None:
    """Test parallel EVSI calculation."""
    # Create parameter set
    params_data = {
        "mean_treatment": np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        "mean_control": np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        "sd_outcome": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data["mean_treatment"]))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create trial design
    trial_design = TrialDesign(
        arms=[
            DecisionOption(name="Treatment", sample_size=50),
            DecisionOption(name="Control", sample_size=50),
        ]
    )

    # Test parallel EVSI calculation
    evsi_result = parallel_evsi_calculation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        population=1000,
        discount_rate=0.03,
        time_horizon=10,
        n_simulations=20,
        n_workers=2,
        use_processes=True,
    )

    # Check that we got a valid result
    assert isinstance(evsi_result, float)
    assert evsi_result >= 0


def test_parallel_bootstrap_sampling() -> None:
    """Test parallel bootstrap sampling."""
    # Create test data
    data = np.random.normal(10, 2, 100).astype(np.float64)

    # Test parallel bootstrap sampling
    result = parallel_bootstrap_sampling(
        data=data,
        statistic_func=mean_statistic,
        n_bootstrap_samples=100,
        n_workers=2,
        use_processes=True,
    )

    # Check that we got valid results
    assert isinstance(result, dict)
    assert "mean" in result
    assert "std" in result
    assert "percentile_2.5" in result
    assert "percentile_97.5" in result
    assert "samples" in result
    assert isinstance(result["samples"], np.ndarray)
    assert len(result["samples"]) == 100


def test_parallel_monte_carlo_with_threads() -> None:
    """Test parallel Monte Carlo simulation using threads instead of processes."""
    # Create parameter set
    params_data = {
        "mean_treatment": np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        "mean_control": np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        "sd_outcome": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data["mean_treatment"]))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    # Create trial design
    trial_design = TrialDesign(
        arms=[
            DecisionOption(name="Treatment", sample_size=50),
            DecisionOption(name="Control", sample_size=50),
        ]
    )

    # Test parallel Monte Carlo simulation with threads
    result = parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=20,
        n_workers=2,
        use_processes=False,  # Use threads
    )

    # Check that we got a valid result
    assert isinstance(result, float)
    assert result >= 0


def test_parallel_bootstrap_with_threads() -> None:
    """Test parallel bootstrap sampling using threads instead of processes."""
    # Create test data
    data = np.random.normal(10, 2, 100).astype(np.float64)

    # Test parallel bootstrap sampling with threads
    result = parallel_bootstrap_sampling(
        data=data,
        statistic_func=mean_statistic,
        n_bootstrap_samples=100,
        n_workers=2,
        use_processes=False,  # Use threads
    )

    # Check that we got valid results
    assert isinstance(result, dict)
    assert "mean" in result
    assert "std" in result
    assert "percentile_2.5" in result
    assert "percentile_97.5" in result
    assert "samples" in result
    assert isinstance(result["samples"], np.ndarray)
    assert len(result["samples"]) == 100


def test_parallel_helpers_fallback_to_threads_when_processes_unavailable(
    monkeypatch,
) -> None:
    """Test that process requests fall back cleanly when process pools fail."""

    class FailingProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            raise OSError("process pools are unavailable")

    monkeypatch.setattr(monte_carlo, "ProcessPoolExecutor", FailingProcessPoolExecutor)

    params_data = {
        "mean_treatment": np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        "mean_control": np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        "sd_outcome": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data["mean_treatment"]))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    trial_design = TrialDesign(
        arms=[
            DecisionOption(name="Treatment", sample_size=50),
            DecisionOption(name="Control", sample_size=50),
        ]
    )

    monte_carlo_result = parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=20,
        n_workers=2,
        use_processes=True,
    )

    bootstrap_result = parallel_bootstrap_sampling(
        data=np.random.normal(10, 2, 100).astype(np.float64),
        statistic_func=mean_statistic,
        n_bootstrap_samples=100,
        n_workers=2,
        use_processes=True,
    )

    assert isinstance(monte_carlo_result, float)
    assert monte_carlo_result >= 0
    assert isinstance(bootstrap_result, dict)
    assert isinstance(bootstrap_result["samples"], np.ndarray)
    assert len(bootstrap_result["samples"]) == 100


def test_partition_workload_and_counts() -> None:
    """Test balanced workload partitioning for cluster-oriented execution."""
    slices = partition_workload(10, 3)
    counts = partition_counts(10, 3)

    assert slices == [slice(0, 4), slice(4, 7), slice(7, 10)]
    assert counts == [4, 3, 3]


def test_distributed_map_preserves_order() -> None:
    """Test that distributed mapping keeps input order deterministic."""
    result = distributed_map(
        [3, 1, 2],
        lambda value: value * 2,
        use_processes=False,
    )
    assert result == [6, 2, 4]


def test_distributed_reduce_and_chunk_map() -> None:
    """Test distributed reduction and chunk mapping."""
    mapped = distributed_chunk_map(
        [[1, 2], [3]],
        sum,
        use_processes=False,
    )
    reduced = distributed_reduce(
        [1, 2, 3],
        lambda value: value * 2,
        sum,
        use_processes=False,
    )

    assert mapped == [3, 3]
    assert reduced == 12


def test_distributed_monte_carlo_simulation() -> None:
    """Test the cluster-oriented Monte Carlo wrapper."""
    params_data = {
        "mean_treatment": np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        "mean_control": np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        "sd_outcome": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data["mean_treatment"]))},
    )
    psa_prior = ParameterSet(dataset=dataset)

    trial_design = TrialDesign(
        arms=[
            DecisionOption(name="Treatment", sample_size=50),
            DecisionOption(name="Control", sample_size=50),
        ]
    )

    cluster_config = ClusterExecutionConfig(n_nodes=2, workers_per_node=1)
    result = distributed_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=20,
        cluster_config=cluster_config,
    )

    assert isinstance(result, float)
    assert result >= 0


def test_parallel_and_distributed_monte_carlo_share_scalar_shape() -> None:
    """The local and distributed CPU lanes should both preserve scalar output shape."""
    params_data = {
        "mean_treatment": np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float64),
        "mean_control": np.array([0.5, 0.6, 0.4, 0.5], dtype=np.float64),
        "sd_outcome": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64),
    }

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(len(params_data["mean_treatment"]))},
    )
    psa_prior = ParameterSet(dataset=dataset)
    trial_design = TrialDesign(
        arms=[
            DecisionOption(name="Treatment", sample_size=50),
            DecisionOption(name="Control", sample_size=50),
        ]
    )

    local_result = parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=8,
        n_workers=2,
        use_processes=False,
    )
    distributed_result = distributed_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=8,
        cluster_config=ClusterExecutionConfig(
            n_nodes=1,
            workers_per_node=2,
            use_processes=False,
            scheduler="thread",
        ),
    )

    assert isinstance(local_result, float)
    assert isinstance(distributed_result, float)
    assert local_result >= 0
    assert distributed_result >= 0


def test_execution_adapter_resolution() -> None:
    """Test resolving execution adapters and building executor factories."""
    process_adapter = get_execution_adapter("process")
    thread_adapter = get_execution_adapter("thread")

    assert isinstance(process_adapter, LocalProcessAdapter)
    assert isinstance(thread_adapter, LocalThreadAdapter)
    assert isinstance(get_execution_adapter("ray"), RayClusterAdapter)
    assert isinstance(get_execution_adapter("fpga"), FpgaClusterAdapter)
    assert isinstance(get_execution_adapter("asic"), AsicClusterAdapter)

    factory = build_executor_factory(thread_adapter)
    with factory(2, use_processes=False) as executor:
        future = executor.submit(lambda value: value + 1, 1)
        assert future.result() == 2


def test_dask_cluster_adapter_soft_dependency_gate() -> None:
    """The Dask adapter should fail clearly when the optional dependency is absent."""
    adapter = DaskClusterAdapter()
    with pytest.raises(ImportError, match="dask.distributed"):
        adapter.create_executor(1)


def test_dask_cluster_adapter_with_dependency() -> None:
    """Validate the Dask adapter executor path when the optional dependency exists."""
    pytest = __import__("pytest")
    pytest.importorskip("dask.distributed")

    adapter = DaskClusterAdapter()
    with adapter.create_executor(1) as executor:
        future = executor.submit(lambda value: value + 2, 3)
        assert future.result() == 5


def test_ray_cluster_adapter_soft_dependency_gate() -> None:
    """The Ray adapter should fail clearly when the optional dependency is absent."""
    adapter = RayClusterAdapter()
    with pytest.raises(ImportError, match="ray is required"):
        adapter.create_executor(1)


def test_ray_cluster_adapter_with_dependency() -> None:
    """Validate the Ray adapter executor path when the optional dependency exists."""
    pytest.importorskip("ray")

    adapter = RayClusterAdapter()
    with adapter.create_executor(1) as executor:
        future = executor.submit(lambda value: value + 2, 3)
        assert future.result() == 5


def test_fpga_cluster_adapter_explicit_placeholder() -> None:
    """The FPGA adapter should fail explicitly until a runtime exists."""
    adapter = FpgaClusterAdapter()
    with pytest.raises(NotImplementedError, match="FPGA execution is not implemented"):
        adapter.create_executor(1)


def test_asic_cluster_adapter_explicit_placeholder() -> None:
    """The ASIC adapter should fail explicitly until a runtime exists."""
    adapter = AsicClusterAdapter()
    with pytest.raises(
        NotImplementedError, match="ASIC/custom-circuit execution is not implemented"
    ):
        adapter.create_executor(1)


def test_placeholder_adapters_are_exported() -> None:
    """The parallel package should export the placeholder accelerator adapters."""
    from voiage.parallel import AsicClusterAdapter as ExportedAsicClusterAdapter
    from voiage.parallel import FpgaClusterAdapter as ExportedFpgaClusterAdapter
    from voiage.parallel import (
        available_execution_adapters as exported_available_execution_adapters,
    )

    assert ExportedFpgaClusterAdapter is FpgaClusterAdapter
    assert ExportedAsicClusterAdapter is AsicClusterAdapter
    assert exported_available_execution_adapters is available_execution_adapters


def test_available_execution_adapters_lists_placeholder_names() -> None:
    """The adapter discovery helper should expose the placeholder scheduler names."""
    assert available_execution_adapters() == (
        "process",
        "thread",
        "dask",
        "ray",
        "fpga",
        "asic",
    )


def test_placeholder_execution_adapter_detection() -> None:
    """Placeholder adapter detection should distinguish unsupported accelerator names."""
    assert is_placeholder_execution_adapter("fpga")
    assert is_placeholder_execution_adapter("asic-cluster")
    assert not is_placeholder_execution_adapter("process")
    assert not is_placeholder_execution_adapter("ray")


if __name__ == "__main__":
    test_monte_carlo_worker()
    test_parallel_monte_carlo_simulation()
    test_parallel_evsi_calculation()
    test_parallel_bootstrap_sampling()
    test_parallel_monte_carlo_with_threads()
    test_parallel_bootstrap_with_threads()
    print("All parallel processing tests passed!")
