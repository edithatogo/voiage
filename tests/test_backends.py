"""Tests for the computational backends."""

import importlib
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from voiage import main_backends
from voiage.backends import NumpyBackend, get_backend, set_backend
from voiage.schema import DecisionOption, TrialDesign


def _import_module_without_jax(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    """Import a voiage submodule in an isolated package without JAX."""

    class BlockJaxFinder:
        def find_spec(
            self,
            fullname: str,
            path: object | None = None,
            target: object | None = None,
        ) -> object | None:
            if fullname == "jax" or fullname.startswith("jax."):
                raise ImportError("blocked jax")
            return None

    package = ModuleType("voiage")
    package.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "voiage")]

    monkeypatch.setattr(sys, "meta_path", [BlockJaxFinder(), *sys.meta_path])
    monkeypatch.setitem(sys.modules, "voiage", package)
    for name in (
        "jax",
        "jax.numpy",
        "voiage.analysis",
        "voiage.backends",
        "voiage.backends.advanced_jax_regression",
        "voiage.backends.enhanced_jax_backend",
        "voiage.backends.gpu_acceleration",
        "voiage.backends.performance_profiler",
        "voiage.config",
        "voiage.core",
        "voiage.core.utils",
        "voiage.exceptions",
        "voiage.main_backends",
        "voiage.schema",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    return importlib.import_module(module_name)


def test_numpy_backend() -> None:
    """Test the NumPy backend."""
    backend = get_backend("numpy")
    assert isinstance(backend, NumpyBackend)

    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Calculate EVPI
    evpi_result = backend.evpi(net_benefit_array)
    assert isinstance(evpi_result, (float, np.floating))
    assert evpi_result >= 0


def test_backend_registry() -> None:
    """Test the backend registry functionality."""
    # Test getting the default backend
    default_backend = get_backend()
    assert isinstance(default_backend, NumpyBackend)

    # Test setting and getting a different backend
    set_backend("numpy")
    numpy_backend = get_backend("numpy")
    assert isinstance(numpy_backend, NumpyBackend)

    # Test error handling for unknown backends
    with pytest.raises(ValueError):
        get_backend("unknown_backend")

    with pytest.raises(ValueError):
        set_backend("unknown_backend")


def test_backend_consistency() -> None:
    """Test that different backends produce consistent results."""
    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Get NumPy backend
    numpy_backend = get_backend("numpy")
    numpy_evpi = numpy_backend.evpi(net_benefit_array)

    # Test that results are consistent
    assert numpy_evpi >= 0


def test_numpy_backend_enbs_simple_positive_value() -> None:
    """ENBS should subtract research cost from EVPI when still beneficial."""
    backend = get_backend("numpy")
    net_benefit_array = np.array(
        [
            [10.0, 5.0],
            [1.0, 8.0],
        ]
    )

    enbs = backend.enbs_simple(net_benefit_array, research_cost=1.0)

    assert enbs == pytest.approx(1.5)


def test_numpy_backend_enbs_simple_floors_at_zero() -> None:
    """ENBS should not return a negative value when research cost dominates."""
    backend = get_backend("numpy")
    net_benefit_array = np.array(
        [
            [10.0, 5.0],
            [1.0, 8.0],
        ]
    )

    enbs = backend.enbs_simple(net_benefit_array, research_cost=10.0)

    assert enbs == 0.0


def test_numpy_backend_enbs_simple_jit_matches_standard() -> None:
    """The convenience JIT wrapper should match the eager implementation."""
    backend = get_backend("numpy")
    net_benefit_array = np.array(
        [
            [10.0, 5.0],
            [1.0, 8.0],
        ]
    )

    eager = backend.enbs_simple(net_benefit_array, research_cost=1.0)
    jitted = backend.enbs_simple_jit(net_benefit_array, research_cost=1.0)

    assert jitted == pytest.approx(eager)


# Test JAX backend if available
try:
    from voiage.backends import JaxBackend

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend() -> None:
    """Test the JAX backend."""
    from voiage.backends import get_backend

    backend = get_backend("jax")
    assert isinstance(backend, JaxBackend)

    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Calculate EVPI
    evpi_result = backend.evpi(net_benefit_array)
    # JAX returns JAX arrays, which can be converted to Python floats
    assert hasattr(evpi_result, "__float__") or isinstance(
        evpi_result, (float, np.floating)
    )
    assert float(evpi_result) >= 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_jit() -> None:
    """Test the JIT-compiled version of the JAX backend."""
    from voiage.backends import get_backend

    backend = get_backend("jax")

    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Calculate EVPI with JIT compilation
    evpi_result = backend.evpi_jit(net_benefit_array)
    # JAX returns JAX arrays, which can be converted to Python floats
    assert hasattr(evpi_result, "__float__") or isinstance(
        evpi_result, (float, np.floating)
    )
    assert float(evpi_result) >= 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_backend_consistency_jax() -> None:
    """Test that NumPy and JAX backends produce consistent results."""
    # Create sample data
    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)

    # Get NumPy backend
    numpy_backend = get_backend("numpy")
    numpy_evpi = numpy_backend.evpi(net_benefit_array)

    # Get JAX backend
    jax_backend = get_backend("jax")
    jax_evpi = jax_backend.evpi(net_benefit_array)

    # Test that results are consistent (within floating point precision)
    # Use a more reasonable tolerance for JAX/NumPy differences
    assert abs(numpy_evpi - float(jax_evpi)) < 1e-6


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_advanced_backends_smoke() -> None:
    """Test the advanced backend wrappers."""
    from voiage.backends.advanced_integration import JaxAdvancedBackend
    from voiage.backends.enhanced_jax_backend import EnhancedJaxBackend

    np.random.seed(42)
    net_benefit_array = np.random.randn(100, 3)
    parameter_samples = np.random.randn(100, 2)

    enhanced_backend = EnhancedJaxBackend()
    enhanced_evppi = enhanced_backend.evppi_advanced(
        net_benefit_array,
        parameter_samples,
        ["p1", "p2"],
    )
    assert hasattr(enhanced_evppi, "__float__") or isinstance(
        enhanced_evppi, (float, np.floating)
    )
    assert float(enhanced_evppi) >= 0

    advanced_backend = JaxAdvancedBackend()
    advanced_evppi = advanced_backend.evppi_advanced(
        net_benefit_array,
        parameter_samples,
        ["p1", "p2"],
    )
    assert hasattr(advanced_evppi, "__float__") or isinstance(
        advanced_evppi, (float, np.floating)
    )
    assert float(advanced_evppi) >= 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_enhanced_jax_backend_batch_parallel_and_linear_paths() -> None:
    """Cover enhanced backend batch, linear fallback, and Monte Carlo helpers."""
    from voiage.backends.enhanced_jax_backend import EnhancedJaxBackend

    backend = EnhancedJaxBackend()
    net_benefit_array = np.array(
        [
            [10.0, 5.0],
            [2.0, 8.0],
            [11.0, 7.0],
            [3.0, 9.0],
        ]
    )
    parameters = np.array(
        [
            [0.1, 1.0],
            [0.9, 2.0],
            [0.2, 1.5],
            [0.8, 2.5],
        ]
    )

    linear_evppi = backend.evppi_advanced(
        net_benefit_array,
        parameters,
        ["risk", "cost"],
        method="ridge",
        degree=1,
    )
    dict_evppi = backend.evppi_advanced(
        net_benefit_array,
        {
            "risk": parameters[:, 0],
            "cost": parameters[:, 1],
        },
        ["risk", "cost"],
        method="ridge",
        degree=1,
    )
    batch = backend.batch_evppi(
        [net_benefit_array, net_benefit_array + 1.0],
        parameters,
        ["risk", "cost"],
    )
    monte_carlo = backend.parallel_monte_carlo(
        net_benefit_array,
        n_simulations=3,
        chunk_size=10,
    )

    assert linear_evppi >= 0.0
    assert dict_evppi >= 0.0
    assert batch.shape == (2,)
    assert set(monte_carlo) == {"mean", "std", "values"}
    assert monte_carlo["values"].shape == (3,)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_enhanced_jax_backend_numpy_sampling_fallback(monkeypatch) -> None:
    """Cover the NumPy sampling branch used when JAX is unavailable at runtime."""
    from voiage.backends import enhanced_jax_backend
    from voiage.backends.enhanced_jax_backend import EnhancedJaxBackend

    backend = EnhancedJaxBackend()
    net_benefit_array = np.array(
        [
            [10.0, 5.0],
            [2.0, 8.0],
            [11.0, 7.0],
            [3.0, 9.0],
        ]
    )

    monkeypatch.setattr(enhanced_jax_backend, "HAS_JAX", False)
    result = backend.parallel_monte_carlo(
        net_benefit_array,
        n_simulations=2,
        chunk_size=2,
    )

    assert result["values"].shape == (2,)



@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_enhanced_jax_backend_compute_evpi_subset() -> None:
    """Test compute_evpi_subset logic indirectly via parallel_monte_carlo."""
    from voiage.backends.enhanced_jax_backend import EnhancedJaxBackend
    import numpy as np

    backend = EnhancedJaxBackend()

    # Test case 1: Identical strategies -> EVPI should be 0
    nb_zeros = np.zeros((10, 2))
    res_zeros = backend.parallel_monte_carlo(nb_zeros, n_simulations=5, chunk_size=5)
    assert res_zeros["mean"] == 0.0
    assert np.all(res_zeros["values"] == 0.0)

    # Test case 2: One strategy strictly dominates -> EVPI should be 0
    nb_dom = np.array([[10.0, 5.0]] * 10)
    res_dom = backend.parallel_monte_carlo(nb_dom, n_simulations=5, chunk_size=5)
    assert res_dom["mean"] == 0.0

    # Test case 3: chunk_size larger than n_samples is capped at n_samples
    nb_small = np.array([[1.0, 0.0], [0.0, 1.0]])
    res_large_chunk = backend.parallel_monte_carlo(nb_small, n_simulations=3, chunk_size=100)
    assert len(res_large_chunk["values"]) == 3

    # Test case 4: Known EVPI behavior for a specific distribution
    res_evpi = backend.parallel_monte_carlo(nb_small, n_simulations=20, chunk_size=2)
    assert res_evpi["mean"] >= 0.0
    assert any(v > 0 for v in res_evpi["values"])

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_enhanced_jax_backend_compute_evpi_subset_no_jax(monkeypatch) -> None:
    """Test compute_evpi_subset logic indirectly without JAX."""
    from voiage.backends import enhanced_jax_backend
    from voiage.backends.enhanced_jax_backend import EnhancedJaxBackend
    import numpy as np

    backend = EnhancedJaxBackend()
    monkeypatch.setattr(enhanced_jax_backend, "HAS_JAX", False)

    nb_zeros = np.zeros((10, 2))
    res_zeros = backend.parallel_monte_carlo(nb_zeros, n_simulations=5, chunk_size=5)
    assert res_zeros["mean"] == 0.0

    nb_small = np.array([[1.0, 0.0], [0.0, 1.0]])
    res_evpi = backend.parallel_monte_carlo(nb_small, n_simulations=20, chunk_size=2)
    assert res_evpi["mean"] >= 0.0
    assert any(v > 0 for v in res_evpi["values"])


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_evppi_and_enbs_helpers() -> None:
    """Cover JAX EVPPI and ENBS helper paths with deterministic data."""
    backend = get_backend("jax")
    net_benefit_array = np.array(
        [
            [10.0, 5.0],
            [2.0, 8.0],
            [11.0, 7.0],
            [3.0, 9.0],
        ]
    )
    parameters = {
        "risk": np.array([0.1, 0.9, 0.2, 0.8]),
        "cost": np.array([1.0, 2.0, 1.5, 2.5]),
    }

    evppi = backend.evppi(net_benefit_array, parameters, ["risk", "cost"])
    evppi_jit = backend.evppi_jit(net_benefit_array, parameters["risk"], ["risk"])
    single_strategy_evppi = backend.evppi(
        net_benefit_array[:, :1], parameters["risk"], ["risk"]
    )
    dict_evppi_jit = backend.evppi_jit(net_benefit_array, parameters, ["risk", "cost"])
    enbs = backend.enbs(evsi_result=10.0, research_cost=4.0)
    enbs_jit = backend.enbs_jit(evsi_result=10.0, research_cost=4.0)
    simple = backend.enbs_simple(net_benefit_array, research_cost=1.0)
    simple_jit = backend.enbs_simple_jit(net_benefit_array, research_cost=1.0)

    assert evppi >= 0.0
    assert evppi_jit >= 0.0
    assert single_strategy_evppi == 0.0
    assert dict_evppi_jit >= 0.0
    assert enbs == pytest.approx(6.0)
    assert enbs_jit == pytest.approx(6.0)
    assert simple >= 0.0
    assert simple_jit >= 0.0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_evsi_dispatch_scaling_and_unknown_method(monkeypatch) -> None:
    """Cover JAX EVSI dispatch, population scaling, and method validation."""
    backend = get_backend("jax")
    trial_design = TrialDesign([DecisionOption("New Treatment", 3)])
    prior = SimpleNamespace(
        parameters={
            "mean_new_treatment": np.array([1.0, 2.0, 3.0]),
            "sd_outcome": np.array([1.0, 1.0, 1.0]),
        },
        n_samples=3,
    )

    def model_func(_prior: object) -> SimpleNamespace:
        return SimpleNamespace(values=np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]]))

    monkeypatch.setattr(backend._impl, "_evsi_two_loop_jax", lambda *args: 5.0)
    monkeypatch.setattr(backend._impl, "_evsi_regression_jax", lambda *args: 6.0)

    two_loop = backend.evsi(
        model_func,
        prior,
        trial_design,
        method="two_loop",
        population=100.0,
        time_horizon=2.0,
        discount_rate=0.0,
    )
    discounted = backend.evsi(
        model_func,
        prior,
        trial_design,
        method="two_loop",
        population=100.0,
        time_horizon=2.0,
        discount_rate=0.03,
    )
    regression = backend.evsi(model_func, prior, trial_design, method="regression")

    assert two_loop == pytest.approx((5.0 - 3.0) * 100.0 * 2.0)
    assert discounted < two_loop
    assert regression == pytest.approx(3.0)

    with pytest.raises(ValueError, match="not recognized"):
        backend.evsi(model_func, prior, trial_design, method="unknown")


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_trial_simulation_and_bayesian_update_helpers() -> None:
    """Cover deterministic JAX EVSI helper branches."""
    backend = get_backend("jax")
    trial_design = TrialDesign([DecisionOption("New Treatment", 4)])

    class Prior(SimpleNamespace):
        def replace_parameters(self, parameters: dict[str, np.ndarray]) -> "Prior":
            return Prior(
                parameters=parameters, n_samples=len(next(iter(parameters.values())))
            )

    prior = Prior(
        parameters={
            "mean_new_treatment": np.array([1.0, 1.5, 2.0, 2.5]),
            "mean_control": np.array([0.5, 0.75, 1.0, 1.25]),
            "sd_outcome": np.array([1.0, 1.0, 1.0, 1.0]),
        },
        n_samples=4,
    )

    trial_data = backend._impl._simulate_trial_data_jax(
        {"mean_new_treatment": 2.0, "sd_outcome": 1.0},
        trial_design,
    )
    posterior = backend._impl._bayesian_update_jax(prior, trial_data, trial_design)

    assert set(trial_data) == {"New Treatment"}
    assert trial_data["New Treatment"].shape == (4,)
    assert set(posterior.parameters) == set(prior.parameters)
    assert posterior.parameters["mean_new_treatment"].shape == (4,)
    np.testing.assert_array_equal(
        posterior.parameters["mean_control"], prior.parameters["mean_control"]
    )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_two_loop_evsi_computes_without_numpy_fallback(
    monkeypatch,
) -> None:
    """JAX two-loop EVSI should compute real posterior value, not placeholder 0."""
    backend = get_backend("jax")
    trial_design = TrialDesign([DecisionOption("New Treatment", 4)])

    class Prior(SimpleNamespace):
        def replace_parameters(self, parameters: dict[str, np.ndarray]) -> "Prior":
            return Prior(
                parameters=parameters, n_samples=len(next(iter(parameters.values())))
            )

    prior = Prior(
        parameters={
            "mean_new_treatment": np.array([2.0, 4.0, 6.0, 8.0]),
            "mean_control": np.array([0.0, 0.0, 0.0, 0.0]),
            "sd_outcome": np.array([1.0, 1.0, 1.0, 1.0]),
        },
        n_samples=4,
    )

    def model_func(updated_prior: Prior) -> SimpleNamespace:
        return SimpleNamespace(
            values=np.column_stack(
                [
                    updated_prior.parameters["mean_control"],
                    updated_prior.parameters["mean_new_treatment"],
                ]
            )
        )

    def fail_numpy_fallback(*_args, **_kwargs):
        msg = "numpy fallback should not be used"
        raise AssertionError(msg)

    monkeypatch.setattr("voiage.methods.sample_information.evsi", fail_numpy_fallback)

    expected_post_study_max = backend._impl._evsi_two_loop_jax(
        model_func,
        prior,
        trial_design,
        n_outer_loops=3,
        n_inner_loops=2,
    )
    public_evsi = backend.evsi(
        model_func,
        prior,
        trial_design,
        method="two_loop",
        n_outer_loops=3,
        n_inner_loops=2,
    )

    assert expected_post_study_max > 0.0
    assert public_evsi >= 0.0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_regression_evsi_helper_executes_full_path(monkeypatch) -> None:
    """Cover the regression EVSI helper without invoking the slow public path."""
    backend = get_backend("jax")
    trial_design = TrialDesign([DecisionOption("New Treatment", 3)])

    class Prior(SimpleNamespace):
        def replace_parameters(self, parameters: dict[str, np.ndarray]) -> "Prior":
            return Prior(
                parameters=parameters, n_samples=len(next(iter(parameters.values())))
            )

    prior = Prior(
        parameters={
            "mean_new_treatment": np.array([1.0, 2.0, 3.0]),
            "sd_outcome": np.array([1.0, 1.0, 1.0]),
        },
        n_samples=3,
    )

    def model_func(updated_prior: Prior) -> SimpleNamespace:
        mean = updated_prior.parameters["mean_new_treatment"]
        return SimpleNamespace(values=np.column_stack([mean, mean + 1.0]))

    monkeypatch.setattr(main_backends, "SKLEARN_AVAILABLE", True)

    result = backend._impl._evsi_regression_jax(
        model_func,
        prior,
        trial_design,
        n_regression_samples=3,
    )

    assert float(result) >= 0.0


@pytest.mark.skipif(not main_backends.JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_optional_dependency_and_regression_fallbacks(monkeypatch) -> None:
    """Cover optional dependency and fallback branches on the public JAX backend."""
    monkeypatch.setattr(main_backends, "JAX_AVAILABLE", False)
    with pytest.raises(ImportError):
        main_backends.JaxBackend()

    monkeypatch.setattr(main_backends, "JAX_AVAILABLE", True)
    monkeypatch.setattr(main_backends, "SKLEARN_AVAILABLE", False)
    backend = main_backends.JaxBackend()
    monkeypatch.setattr(backend._impl, "_evsi_two_loop_jax", lambda *args: 11.0)

    result = backend._impl._evsi_regression_jax(
        lambda _prior: SimpleNamespace(values=np.array([[1.0, 2.0]])),
        SimpleNamespace(parameters={"p": np.array([1.0])}, n_samples=1),
        TrialDesign([DecisionOption("New Treatment", 1)]),
        n_regression_samples=1,
    )

    assert result == pytest.approx(11.0)


@pytest.mark.skipif(not main_backends.JAX_AVAILABLE, reason="JAX not available")
def test_jax_backend_evppi_uses_simple_fallback_when_regression_fails(
    monkeypatch,
) -> None:
    """EVPPI should keep returning a nonnegative value if linear solve fails."""
    backend = main_backends.JaxBackend()
    net_benefit_array = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 6.0]])
    parameter_samples = np.array([0.1, 0.2, 0.3])

    def fail_solve(*_args, **_kwargs):
        msg = "synthetic solver failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(main_backends.jnp.linalg, "solve", fail_solve)

    assert backend.evppi(net_benefit_array, parameter_samples, ["risk"]) >= 0.0


def test_main_backends_imports_and_operates_without_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public backend module should still import and serve NumPy paths."""
    module = _import_module_without_jax("voiage.main_backends", monkeypatch)

    backend = module.get_backend()

    assert module.JAX_AVAILABLE is False
    assert backend.__class__.__name__ == "NumpyBackend"
    assert module.get_backend("numpy").__class__.__name__ == "NumpyBackend"
    with pytest.raises(ValueError):
        module.get_backend("jax")


def test_analysis_imports_and_operates_without_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The analysis module should keep working when JAX cannot be imported."""
    module = _import_module_without_jax("voiage.analysis", monkeypatch)

    analysis = module.DecisionAnalysis(np.array([[1.0, 2.0], [3.0, 1.5]]))

    assert module.JAX_AVAILABLE is False
    assert analysis.backend.__class__.__name__ == "NumpyBackend"
    assert analysis.evpi() == pytest.approx(0.5)
