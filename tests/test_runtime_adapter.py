"""Focused tests for the private Python-to-Rust runtime adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from voiage import _runtime
from voiage.exceptions import DimensionMismatchError, InputError, SerializationError
from voiage.methods.ceaf import CEAFResult
from voiage.methods.dominance import DominanceResult

if TYPE_CHECKING:
    from collections.abc import Callable


class NativeError(RuntimeError):
    """Fake native error carrying stable cross-language fields."""

    def __init__(self, category: str, diagnostic_code: str) -> None:
        super().__init__("native validation failed")
        self.category = category
        self.diagnostic_code = diagnostic_code


def _ceaf_result() -> CEAFResult:
    return CEAFResult(
        wtp_thresholds=np.array([0.0]),
        optimal_strategy_indices=np.array([1]),
        optimal_strategy_names=["B"],
        acceptability_probabilities=np.array([0.75]),
        probability_lower=np.array([0.5]),
        probability_upper=np.array([0.9]),
        expected_net_benefit=np.array([12.0]),
        reporting={"standard": "CHEERS 2022"},
    )


def _dominance_result() -> DominanceResult:
    return DominanceResult(
        strategy_names=["A", "B"],
        costs=np.array([1.0, 2.0]),
        effects=np.array([1.0, 2.0]),
        frontier_indices=[0, 1],
        strongly_dominated_indices=[],
        extended_dominated_indices=[],
        status=["frontier", "frontier"],
        incremental_costs=np.array([1.0]),
        incremental_effects=np.array([1.0]),
        icers=np.array([1.0]),
        reporting={"standard": "CHEERS 2022"},
    )


def test_result_serializers_pass_keyword_primitives_to_native(monkeypatch) -> None:
    captured: dict[str, dict[str, object]] = {}

    def ceaf(**kwargs: object) -> dict[str, object]:
        captured["ceaf"] = kwargs
        return {"kind": "ceaf"}

    def dominance(**kwargs: object) -> dict[str, object]:
        captured["dominance"] = kwargs
        return {"kind": "dominance"}

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(
            serialize_ceaf_result=ceaf,
            serialize_dominance_result=dominance,
        ),
    )

    assert _ceaf_result().to_dict(
        analysis_id="analysis-1", decision_problem_id="decision-1"
    ) == {"kind": "ceaf"}
    assert _dominance_result().to_dict(
        analysis_id="analysis-1", decision_problem_id="decision-1"
    ) == {"kind": "dominance"}
    assert captured["ceaf"]["wtp_thresholds"] == [0.0]
    assert captured["ceaf"]["optimal_strategy_names"] == ["B"]
    assert captured["dominance"]["strategy_names"] == ["A", "B"]
    assert captured["dominance"]["costs"] == [1.0, 2.0]
    assert captured["dominance"]["icers"] == [1.0]


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("input", InputError),
        ("dimension_mismatch", DimensionMismatchError),
        ("serialization", SerializationError),
    ],
)
def test_native_errors_keep_category_and_diagnostic_code(
    monkeypatch,
    category: str,
    expected: type[Exception],
) -> None:
    def serialize(**_: object) -> dict[str, object]:
        raise NativeError(category, "invalid_result")

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(serialize_ceaf_result=serialize),
    )

    with pytest.raises(expected, match="native validation failed") as caught:
        _ceaf_result().to_dict(analysis_id="a", decision_problem_id="d")

    assert vars(caught.value)["diagnostic_code"] == "invalid_result"


def test_missing_native_extension_has_no_python_fallback(monkeypatch) -> None:
    missing = ModuleNotFoundError("No module named 'voiage._core'")

    def fail_import(_: str) -> None:
        raise missing

    monkeypatch.setattr(_runtime, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError, match="voiage._core"):
        _ceaf_result().to_dict(analysis_id="a", decision_problem_id="d")


@pytest.mark.parametrize(
    ("native_name", "invoke"),
    [
        ("compute_enbs", lambda: _runtime.compute_enbs(2.0, 1.0)),
        (
            "compute_heterogeneity",
            lambda: _runtime.compute_heterogeneity([[1.0, 2.0]], ["group"]),
        ),
        (
            "compute_structural_evpi",
            lambda: _runtime.compute_structural_evpi([[[1.0, 2.0]]], [1.0]),
        ),
        (
            "compute_structural_evppi",
            lambda: _runtime.compute_structural_evppi([[[1.0, 2.0]]], [1.0], [0]),
        ),
        (
            "compute_evsi_regression",
            lambda: _runtime.compute_evsi_regression([[1.0]], [[1.0]], [[1.0]]),
        ),
    ],
)
def test_remaining_runtime_adapters_translate_native_errors(
    monkeypatch, native_name: str, invoke: Callable[[], object]
) -> None:
    """Every stable adapter preserves the shared native error contract."""

    def fail(*_args: object, **_kwargs: object) -> object:
        raise NativeError("input", "invalid_result")

    monkeypatch.setattr(
        _runtime, "_native", lambda: SimpleNamespace(**{native_name: fail})
    )

    with pytest.raises(InputError, match="native validation failed") as caught:
        invoke()

    assert vars(caught.value)["diagnostic_code"] == "invalid_result"


def test_compute_evppi_forwards_matrix_payloads_to_native(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(net_benefit: list[list[float]], parameters: list[list[float]]) -> float:
        captured["net_benefit"] = net_benefit
        captured["parameters"] = parameters
        return 0.05

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_evppi=compute),
    )

    result = _runtime.compute_evppi([[0.0, 2.0], [1.0, 0.0]], [[0.0], [1.0]])

    assert result == 0.05
    assert captured == {
        "net_benefit": [[0.0, 2.0], [1.0, 0.0]],
        "parameters": [[0.0], [1.0]],
    }


def test_compute_evpi_forwards_matrix_payload_to_native(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(net_benefit: list[list[float]]) -> float:
        captured["net_benefit"] = net_benefit
        return 0.5

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_evpi=compute),
    )

    assert _runtime.compute_evpi([[0.0, 2.0], [1.0, 0.0]]) == 0.5
    assert captured == {"net_benefit": [[0.0, 2.0], [1.0, 0.0]]}


def test_compute_dominance_forwards_vectors_to_native(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(costs: list[float], effects: list[float]) -> dict[str, object]:
        captured.update(costs=costs, effects=effects)
        return {"frontier_indices": [0, 1]}

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_dominance=compute),
    )

    assert _runtime.compute_dominance([1.0, 2.0], [1.0, 2.0]) == {
        "frontier_indices": [0, 1]
    }
    assert captured == {"costs": [1.0, 2.0], "effects": [1.0, 2.0]}


def test_compute_ceaf_forwards_cube_arguments_to_native(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(
        net_benefit: list[list[list[float]]],
        thresholds: list[float],
        confidence: float,
    ) -> dict[str, object]:
        captured.update(
            net_benefit=net_benefit,
            thresholds=thresholds,
            confidence=confidence,
        )
        return {"optimal_strategy_indices": [0, 1]}

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_ceaf=compute),
    )
    cube = [[[1.0, 2.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, 3.0]]]

    assert _runtime.compute_ceaf(cube, [0.0, 1.0], 0.95) == {
        "optimal_strategy_indices": [0, 1]
    }
    assert captured == {
        "net_benefit": cube,
        "thresholds": [0.0, 1.0],
        "confidence": 0.95,
    }


def test_compute_evsi_forwards_seeded_kernel_arguments(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(
        net_benefit: list[list[float]],
        trial_sample_size: int,
        resample_count: int,
        seed: int,
    ) -> dict[str, object]:
        captured.update(
            net_benefit=net_benefit,
            trial_sample_size=trial_sample_size,
            resample_count=resample_count,
            seed=seed,
        )
        return {"evsi": 0.75, "draw_count": 2}

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_evsi=compute),
    )

    result = _runtime.compute_evsi([[10.0, 4.0]], 2, 4, 42)

    assert result == {"evsi": 0.75, "draw_count": 2}
    assert captured == {
        "net_benefit": [[10.0, 4.0]],
        "trial_sample_size": 2,
        "resample_count": 4,
        "seed": 42,
    }


def test_compute_normal_normal_evsi_forwards_declared_study(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(*args: object) -> float:
        captured["args"] = args
        return 124.1793655206238

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_normal_normal_two_arm_evsi=compute),
    )

    result = _runtime.compute_normal_normal_two_arm_evsi(
        0.06,
        0.03,
        1.0,
        200,
        50_000.0,
        -3_000.0,
    )

    assert result == 124.1793655206238
    assert captured["args"] == (0.06, 0.03, 1.0, 200, 50_000.0, -3_000.0)


def test_compute_evsi_efficient_linear_forwards_kernel_arguments(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(
        net_benefit: list[list[float]],
        parameter_samples: list[list[float]],
        trial_sample_size: int,
    ) -> dict[str, object]:
        captured.update(
            net_benefit=net_benefit,
            parameter_samples=parameter_samples,
            trial_sample_size=trial_sample_size,
        )
        return {"evsi": 2.0 / 3.0, "information_fraction": 1.0 / 3.0}

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_evsi_efficient_linear=compute),
    )

    result = _runtime.compute_evsi_efficient_linear(
        [[0.0, 6.0], [2.0, 4.0], [4.0, 2.0], [6.0, 0.0]],
        [[-1.0], [0.0], [1.0], [2.0]],
        2,
    )

    assert result == {"evsi": 2.0 / 3.0, "information_fraction": 1.0 / 3.0}
    assert captured == {
        "net_benefit": [[0.0, 6.0], [2.0, 4.0], [4.0, 2.0], [6.0, 0.0]],
        "parameter_samples": [[-1.0], [0.0], [1.0], [2.0]],
        "trial_sample_size": 2,
    }


def test_compute_evsi_moment_based_forwards_kernel_arguments(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def compute(
        net_benefit: list[list[float]],
        parameter_samples: list[list[float]],
        trial_sample_size: int,
    ) -> dict[str, object]:
        captured.update(
            net_benefit=net_benefit,
            parameter_samples=parameter_samples,
            trial_sample_size=trial_sample_size,
        )
        return {"evsi": 5.0 / 12.0, "estimator": "moment_based"}

    monkeypatch.setattr(
        _runtime,
        "_native",
        lambda: SimpleNamespace(compute_evsi_moment_based=compute),
    )

    result = _runtime.compute_evsi_moment_based(
        [[1.0, 2.0], [0.0, 3.0], [1.0, 2.0], [4.0, -1.0]],
        [[-1.0], [0.0], [1.0], [2.0]],
        2,
    )

    assert result == {"evsi": 5.0 / 12.0, "estimator": "moment_based"}
    assert captured == {
        "net_benefit": [[1.0, 2.0], [0.0, 3.0], [1.0, 2.0], [4.0, -1.0]],
        "parameter_samples": [[-1.0], [0.0], [1.0], [2.0]],
        "trial_sample_size": 2,
    }
