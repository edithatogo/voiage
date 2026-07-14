"""Tests for parallel processing utilities in Value of Information analysis."""

from concurrent.futures import Future
import sys
import types

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
    _RayExecutor,
    available_execution_adapters,
    build_executor_factory,
    get_execution_adapter,
    is_placeholder_execution_adapter,
)
from voiage.parallel.distributed import (
    ClusterExecutionConfig,
    chunked_numpy_partition,
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


def test_run_work_in_monte_carlo_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the run_work inner function of parallel_monte_carlo_simulation directly."""
    import concurrent.futures
    from unittest.mock import MagicMock

    # We will intercept the call to _execute_parallel_work
    # to capture the run_work callback.
    captured_run_work = []

    def mock_execute_parallel_work(*, n_workers, use_processes, work):
        captured_run_work.append(work)
        return 42.0  # arbitrary return value

    monkeypatch.setattr(
        monte_carlo, "_execute_parallel_work", mock_execute_parallel_work
    )

    # Set up dummy arguments for parallel_monte_carlo_simulation
    params_data = {
        "mean_treatment": np.array([1.0]),
        "mean_control": np.array([0.5]),
        "sd_outcome": np.array([0.1]),
    }
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(1)},
    )
    psa_prior = ParameterSet(dataset=dataset)
    trial_design = TrialDesign(arms=[DecisionOption(name="Treatment", sample_size=50)])

    # Call the simulation to trigger capturing run_work
    n_sims = 10
    parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=n_sims,
        n_workers=2,
    )

    assert len(captured_run_work) == 1
    run_work_fn = captured_run_work[0]

    # Create a dummy executor to pass to run_work
    mock_executor = MagicMock(spec=concurrent.futures.Executor)

    # We need futures that return (expected_max_nb, n_sims_processed)
    # The simulation divides 10 sims over 2 workers -> [5, 5]
    f1 = concurrent.futures.Future()
    f1.set_result((10.0, 5))
    f2 = concurrent.futures.Future()
    f2.set_result((20.0, 5))

    mock_executor.submit.side_effect = [f1, f2]

    result = run_work_fn(mock_executor)

    # (10 * 5 + 20 * 5) / 10 = 15.0
    assert result == 15.0
    assert mock_executor.submit.call_count == 2

    # Test zero simulations case
    mock_executor.reset_mock()
    # If a future returns 0 processed
    f3 = concurrent.futures.Future()
    f3.set_result((10.0, 0))
    f4 = concurrent.futures.Future()
    f4.set_result((20.0, 0))
    mock_executor.submit.side_effect = [f3, f4]

    result_zero = run_work_fn(mock_executor)
    assert result_zero == 0.0


def test_parallel_monte_carlo_simulation_remainder_coverage() -> None:
    """Test parallel Monte Carlo simulation to cover the remainder logic."""
    # Create parameter set
    params_data = {
        "mean_treatment": np.array([1.0]),
        "mean_control": np.array([0.5]),
        "sd_outcome": np.array([0.1]),
    }
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params_data.items()},
        coords={"n_samples": np.arange(1)},
    )
    psa_prior = ParameterSet(dataset=dataset)
    trial_design = TrialDesign(arms=[DecisionOption(name="Treatment", sample_size=50)])

    # 11 simulations, 2 workers means a remainder of 1
    result = parallel_monte_carlo_simulation(
        model_func=simple_model_func,
        psa_prior=psa_prior,
        trial_design=trial_design,
        n_simulations=11,
        n_workers=2,
        use_processes=False,
    )
    assert result >= 0


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


def test_run_work_in_bootstrap_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the run_work inner function of parallel_bootstrap_sampling directly."""
    import concurrent.futures
    from unittest.mock import MagicMock

    captured_run_work = []

    def mock_execute_parallel_work(*, n_workers, use_processes, work):
        captured_run_work.append(work)
        return {"mean": 42.0}  # arbitrary return value

    monkeypatch.setattr(
        monte_carlo, "_execute_parallel_work", mock_execute_parallel_work
    )

    data = np.array([1.0, 2.0, 3.0])

    parallel_bootstrap_sampling(
        data=data,
        statistic_func=mean_statistic,
        n_bootstrap_samples=10,
        n_workers=2,
    )

    assert len(captured_run_work) == 1
    run_work_fn = captured_run_work[0]

    mock_executor = MagicMock(spec=concurrent.futures.Executor)

    f1 = concurrent.futures.Future()
    f1.set_result([1.0, 2.0, 3.0, 4.0, 5.0])
    f2 = concurrent.futures.Future()
    f2.set_result([6.0, 7.0, 8.0, 9.0, 10.0])

    mock_executor.submit.side_effect = [f1, f2]

    result = run_work_fn(mock_executor)

    assert "mean" in result
    assert result["mean"] == 5.5
    assert len(result["samples"]) == 10


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


def test_local_adapters_create_expected_executor_types() -> None:
    """Local adapters should instantiate concrete executor pools."""
    with LocalThreadAdapter().create_executor(1) as executor:
        future = executor.submit(lambda value: value + 1, 1)
        assert future.result() == 2

    process_executor = LocalProcessAdapter().create_executor(1)
    try:
        assert process_executor._max_workers == 1
    finally:
        process_executor.shutdown()


def test_execution_adapter_aliases_and_unknown_name() -> None:
    """All public aliases should resolve while unknown names fail clearly."""
    assert isinstance(get_execution_adapter(" processes "), LocalProcessAdapter)
    assert isinstance(get_execution_adapter("local-process"), LocalProcessAdapter)
    assert isinstance(get_execution_adapter("threads"), LocalThreadAdapter)
    assert isinstance(get_execution_adapter("local-thread"), LocalThreadAdapter)
    assert isinstance(get_execution_adapter("distributed"), DaskClusterAdapter)
    assert isinstance(get_execution_adapter("dask-cluster"), DaskClusterAdapter)
    assert isinstance(get_execution_adapter("ray-cluster"), RayClusterAdapter)
    assert isinstance(get_execution_adapter("fpga-cluster"), FpgaClusterAdapter)
    assert isinstance(get_execution_adapter("asic-cluster"), AsicClusterAdapter)

    with pytest.raises(ValueError, match="Unknown execution adapter"):
        get_execution_adapter("quantum")


def test_build_executor_factory_falls_back_to_threads() -> None:
    """Adapter factory should fall back to threads if process-style creation fails."""

    class FailingAdapter:
        name = "failing"

        def create_executor(self, n_workers: int):
            raise RuntimeError("boom")

    factory = build_executor_factory(FailingAdapter())
    with factory(1, use_processes=True) as executor:
        future = executor.submit(lambda value: value + 1, 1)
        assert future.result() == 2


def _install_fake_dask(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    distributed_module = types.ModuleType("dask.distributed")
    targets: list[object] = []

    class FakeLocalCluster:
        def __init__(self, n_workers: int, threads_per_worker: int) -> None:
            self.n_workers = n_workers
            self.threads_per_worker = threads_per_worker

    class FakeClient:
        def __init__(self, target: object) -> None:
            self.target = target
            self.closed = False
            targets.append(target)

        def submit(self, fn, *args, **kwargs):
            future: Future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

        def close(self) -> None:
            self.closed = True

    distributed_module.Client = FakeClient
    distributed_module.LocalCluster = FakeLocalCluster
    dask_module = types.ModuleType("dask")
    dask_module.distributed = distributed_module

    monkeypatch.setitem(sys.modules, "dask", dask_module)
    monkeypatch.setitem(sys.modules, "dask.distributed", distributed_module)
    return targets


def test_dask_cluster_adapter_with_fake_local_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Dask adapter should wrap a LocalCluster when no address is supplied."""
    fake_targets = _install_fake_dask(monkeypatch)

    with DaskClusterAdapter().create_executor(2) as executor:
        future = executor.submit(lambda value: value + 2, 3)
        assert future.result() == 5

    target = fake_targets[-1]
    assert target.n_workers == 2
    assert target.threads_per_worker == 1


def test_dask_cluster_adapter_with_fake_scheduler_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Dask adapter should pass explicit scheduler addresses to Client."""
    fake_targets = _install_fake_dask(monkeypatch)

    executor = DaskClusterAdapter(
        scheduler_address="tcp://scheduler:8786"
    ).create_executor(4)
    try:
        assert executor.submit(lambda value: value * 2, 4).result() == 8
    finally:
        executor.shutdown()

    assert fake_targets[-1] == "tcp://scheduler:8786"


class _FakeRemoteFunction:
    def __init__(self, fn) -> None:
        self._fn = fn

    def remote(self):
        return self._fn()


class _FakeRayModule:
    def __init__(self, *, initialized: bool = False) -> None:
        self._initialized = initialized
        self.init_kwargs: list[dict[str, object]] = []
        self.shutdown_called = False

    def is_initialized(self) -> bool:
        return self._initialized

    def init(self, **kwargs: object) -> None:
        self.init_kwargs.append(kwargs)
        self._initialized = True

    def remote(self, fn):
        return _FakeRemoteFunction(fn)

    def get(self, value):
        return value

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_ray_cluster_adapter_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ray adapter should initialize and wrap a Ray-like module."""
    fake_ray = _FakeRayModule()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    with RayClusterAdapter().create_executor(3) as executor:
        future = executor.submit(lambda value: value + 7, 5)
        assert future.result(timeout=5) == 12

    assert fake_ray.init_kwargs == [{"num_cpus": 3}]
    assert fake_ray.shutdown_called is True


def test_ray_cluster_adapter_with_fake_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ray adapter should initialize using an explicit cluster address."""
    fake_ray = _FakeRayModule()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    executor = RayClusterAdapter(address="ray://cluster").create_executor(2)
    try:
        assert executor.submit(lambda value: value * 3, 4).result(timeout=5) == 12
    finally:
        executor.shutdown()

    assert fake_ray.init_kwargs == [{"address": "ray://cluster"}]


def test_ray_executor_skips_shutdown_when_unavailable() -> None:
    """Ray-like modules without shutdown should still satisfy the executor API."""

    def remote(fn):
        return _FakeRemoteFunction(fn)

    def get(value):
        return value

    fake_ray = types.SimpleNamespace(
        remote=remote,
        get=get,
    )
    wrapped = _RayExecutor(fake_ray)

    assert wrapped.submit(lambda: "ok").result(timeout=5) == "ok"
    wrapped.shutdown()


def test_distributed_validation_and_fallback_branches() -> None:
    """Distributed helpers should validate inputs and fall back deterministically."""
    for kwargs, match in (
        ({"n_nodes": 0}, "n_nodes"),
        ({"workers_per_node": 0}, "workers_per_node"),
        ({"chunk_count": 0}, "chunk_count"),
        ({"scheduler": " "}, "scheduler"),
    ):
        with pytest.raises(ValueError, match=match):
            ClusterExecutionConfig(**kwargs)

    assert ClusterExecutionConfig(n_nodes=1).total_workers >= 1

    with pytest.raises(ValueError, match="total_items"):
        partition_workload(-1, 1)
    with pytest.raises(ValueError, match="partition_count"):
        partition_workload(1, 0)
    assert partition_workload(0, 3) == []

    def failing_factory(n_workers: int, use_processes: bool):
        raise TypeError("executor unavailable")

    assert distributed_map(
        [1, 2], lambda value: value + 1, executor_factory=failing_factory
    ) == [
        2,
        3,
    ]
    assert distributed_map([], lambda value: value) == []

    chunks = chunked_numpy_partition(np.array([1, 2, 3, 4]), 3)
    assert [chunk.tolist() for chunk in chunks] == [[1, 2], [3], [4]]
    with pytest.raises(ValueError, match="one-dimensional"):
        chunked_numpy_partition(np.array([[1, 2]]), 2)


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
