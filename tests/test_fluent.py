"""Focused tests for the fluent DecisionAnalysis wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from voiage import backends
from voiage.fluent import FluentDecisionAnalysis, create_analysis
from voiage.schema import ParameterSet


def _make_analysis() -> FluentDecisionAnalysis:
    """Build a minimal fluent analysis fixture."""
    nb_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    return FluentDecisionAnalysis(nb_array=nb_array, backend="numpy")


def test_with_parameters_accepts_dict_numpy_and_parameterset() -> None:
    """Parameter inputs should normalize to ParameterSet and chain."""
    analysis = _make_analysis()

    dict_samples = {"alpha": np.array([0.1, 0.2]), "beta": np.array([0.3, 0.4])}
    assert analysis.with_parameters(dict_samples) is analysis
    assert isinstance(analysis.parameter_samples, ParameterSet)
    assert set(analysis.parameter_samples.parameter_names) == {"alpha", "beta"}
    assert np.array_equal(
        analysis.parameter_samples.parameters["alpha"], np.array([0.1, 0.2])
    )

    numpy_samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert analysis.with_parameters(numpy_samples) is analysis
    assert isinstance(analysis.parameter_samples, ParameterSet)
    assert analysis.parameter_samples.n_samples == 2
    assert len(analysis.parameter_samples.parameter_names) == 2

    parameter_set = ParameterSet.from_numpy_or_dict({"gamma": np.array([5.0, 6.0])})
    assert analysis.with_parameters(parameter_set) is analysis
    assert analysis.parameter_samples is parameter_set


def test_with_parameters_rejects_invalid_type() -> None:
    """Invalid parameter inputs should raise TypeError."""
    analysis = _make_analysis()

    with pytest.raises(TypeError, match="parameter_samples"):
        analysis.with_parameters("invalid")  # type: ignore[arg-type]


def test_with_backend_uses_backend_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend selection should use the package-level lookup helper."""
    analysis = _make_analysis()
    sentinel_backend = object()

    monkeypatch.setattr(backends, "get_backend", lambda name: sentinel_backend)

    assert analysis.with_backend("custom") is analysis
    assert analysis.backend is sentinel_backend


def test_configuration_helpers_update_analysis_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JIT, streaming, caching, and data ingestion should chain and mutate state."""
    analysis = _make_analysis()
    init_calls: list[str] = []
    update_calls: list[tuple[np.ndarray, dict[str, np.ndarray] | None]] = []

    monkeypatch.setattr(
        analysis, "_initialize_streaming_buffers", lambda: init_calls.append("init")
    )
    monkeypatch.setattr(
        analysis,
        "update_with_new_data",
        lambda nb_data, parameter_data=None: update_calls.append(
            (nb_data, parameter_data)
        ),
    )

    assert analysis.with_jit() is analysis
    assert analysis.use_jit is True

    assert analysis.with_streaming(32) is analysis
    assert analysis.streaming_window_size == 32
    assert init_calls == ["init"]

    assert analysis.with_caching() is analysis
    assert analysis.enable_caching is True
    assert analysis._cache == {}

    assert analysis.with_caching(False) is analysis
    assert analysis.enable_caching is False
    assert analysis._cache is None

    new_nb_data = np.array([[10.0, 20.0]])
    new_parameter_samples = {"theta": np.array([0.5])}
    assert analysis.add_data(new_nb_data, new_parameter_samples) is analysis
    assert len(update_calls) == 1
    assert np.array_equal(update_calls[0][0], new_nb_data)
    assert update_calls[0][1] == new_parameter_samples


def test_calculation_helpers_store_results_and_getters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fluent calculators should store results for later access."""
    analysis = _make_analysis()
    evpi_calls: list[tuple[float | None, float | None, float | None, int | None]] = []
    evppi_calls: list[
        tuple[
            float | None,
            float | None,
            float | None,
            int | None,
            object | None,
            int | None,
        ]
    ] = []

    def fake_evpi(
        population: float | None = None,
        time_horizon: float | None = None,
        discount_rate: float | None = None,
        chunk_size: int | None = None,
    ) -> float:
        evpi_calls.append((population, time_horizon, discount_rate, chunk_size))
        return 12.5

    def fake_evppi(
        population: float | None = None,
        time_horizon: float | None = None,
        discount_rate: float | None = None,
        n_regression_samples: int | None = None,
        regression_model: object | None = None,
        chunk_size: int | None = None,
    ) -> float:
        evppi_calls.append(
            (
                population,
                time_horizon,
                discount_rate,
                n_regression_samples,
                regression_model,
                chunk_size,
            )
        )
        return 3.25

    monkeypatch.setattr(analysis, "evpi", fake_evpi)
    monkeypatch.setattr(analysis, "evppi", fake_evppi)

    regression_model = object()
    assert (
        analysis.calculate_evpi(
            population=100.0,
            time_horizon=5.0,
            discount_rate=0.03,
            chunk_size=10,
        )
        is analysis
    )
    assert evpi_calls == [(100.0, 5.0, 0.03, 10)]
    assert analysis.get_evpi_result() == pytest.approx(12.5)

    assert (
        analysis.calculate_evppi(
            population=250.0,
            time_horizon=2.0,
            discount_rate=0.01,
            n_regression_samples=50,
            regression_model=regression_model,
            chunk_size=4,
        )
        is analysis
    )
    assert evppi_calls == [(250.0, 2.0, 0.01, 50, regression_model, 4)]
    assert analysis.get_evppi_result() == pytest.approx(3.25)
    assert analysis.get_results() == {"evpi": 12.5, "evppi": 3.25}


def test_context_manager_and_factory_return_fluent_analysis() -> None:
    """Factory and context-manager support should preserve fluent analysis objects."""
    with create_analysis(
        np.array([[1.0, 0.0], [0.5, 1.5]]), backend="numpy"
    ) as analysis:
        assert isinstance(analysis, FluentDecisionAnalysis)
        assert analysis.get_results() == {"evpi": None, "evppi": None}
