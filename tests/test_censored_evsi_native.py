"""Installed controls for the state-keyed native censored EVSI binding."""

from fractions import Fraction
from importlib import import_module
import json
from math import erf, sqrt

import pytest


def _observation(
    state_id: str, detection_probability: float, *, mean: float = 0.0
) -> dict[str, object]:
    return {
        "state_id": state_id,
        "mean": mean,
        "standard_deviation": 1.0,
        "detection_probability": detection_probability,
    }


def _action(action_id: str, observations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "action_id": action_id,
        "limit": 0.0,
        "cutpoints": [],
        "outcome_ids": ["unavailable", "below_limit", "reading"],
        "states": observations,
    }


def _problem(stages: list[list[dict[str, object]]]) -> dict[str, object]:
    return {
        "strategy_ids": ["keep", "switch"],
        "states": [
            {"state_id": "left", "prior_weight": 1.0, "net_benefits": [6.0, 0.0]},
            {"state_id": "right", "prior_weight": 1.0, "net_benefits": [0.0, 10.0]},
        ],
        "stages": stages,
    }


def _evaluate(
    problem: dict[str, object], *, max_branches: int = 128
) -> dict[str, object]:
    native = import_module("voiage._core")
    return native.compute_censored_sequential_evsi(json.dumps(problem), max_branches)


def test_censored_evsi_aligns_acquisition_states_by_id() -> None:
    # Unavailability contributes max(3, 2.5). Measured outcomes contribute 2.5.
    # Positionally swapping the detection probabilities gives EVSI 3/2 instead.
    revealing = _action(
        "inspection", [_observation("right", 0.5), _observation("left", 0.0)]
    )
    unavailable = _action(
        "offline", [_observation("left", 0.0), _observation("right", 0.0)]
    )
    result = _evaluate(_problem([[unavailable], [revealing]]))
    assert result["contract_version"] == 1
    assert result["status"] == "complete"
    assert result["state_ids"] == ["left", "right"]
    assert result["strategy_ids"] == ["keep", "switch"]
    assert result["current_value"] == ["5", "1"]
    assert result["expected_value"] == ["11", "2"]
    assert result["evsi"] == ["1", "2"]


def test_censored_evsi_matches_analytic_normal_limit() -> None:
    measured = _action(
        "inspection",
        [_observation("left", 1.0, mean=-1.0), _observation("right", 1.0, mean=1.0)],
    )
    problem = _problem([[measured]])
    problem["states"] = [
        {"state_id": "left", "prior_weight": 1.0, "net_benefits": [1.0, 0.0]},
        {"state_id": "right", "prior_weight": 1.0, "net_benefits": [0.0, 1.0]},
    ]
    result = _evaluate(problem)
    pair = result["evsi"]
    assert isinstance(pair, list)
    actual = Fraction(int(pair[0]), int(pair[1]))
    assert float(actual) == pytest.approx(erf(1.0 / sqrt(2.0)) / 2.0, rel=1e-14)


def test_censored_evsi_preserves_empty_schedule_and_budget_exhaustion() -> None:
    assert _evaluate(_problem([]))["evsi"] == ["0", "1"]
    measured = _action(
        "inspection", [_observation("left", 0.0), _observation("right", 0.5)]
    )
    result = _evaluate(_problem([[measured]]), max_branches=1)
    assert result == {
        "contract_version": 1,
        "status": "indeterminate",
        "state_ids": ["left", "right"],
        "strategy_ids": ["keep", "switch"],
        "current_value": ["5", "1"],
        "branches_examined": 1,
    }


def test_censored_evsi_rejects_duplicate_and_foreign_observation_states() -> None:
    native = import_module("voiage._core")
    duplicate = _action(
        "inspection", [_observation("left", 0.0), _observation("left", 0.5)]
    )
    with pytest.raises(native.InputError, match="unique"):
        _evaluate(_problem([[duplicate]]))

    foreign = _action(
        "inspection", [_observation("right", 0.5), _observation("foreign", 0.0)]
    )
    with pytest.raises(native.DimensionMismatchError, match="state IDs"):
        _evaluate(_problem([[foreign]]))


def test_censored_evsi_rejects_invalid_dimensions_and_cutpoints() -> None:
    native = import_module("voiage._core")
    problem = _problem([])
    problem["strategy_ids"] = ["keep"]
    with pytest.raises(native.DimensionMismatchError, match="strategy IDs"):
        _evaluate(problem)

    measured = _action(
        "inspection", [_observation("left", 0.0), _observation("right", 0.5)]
    )
    measured["cutpoints"] = [0.0]
    with pytest.raises(native.InputError, match="Cutpoints must increase"):
        _evaluate(_problem([[measured]]))
