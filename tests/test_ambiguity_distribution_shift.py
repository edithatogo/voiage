"""Runtime tests for ambiguity and distribution-shift VOI."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.ambiguity_distribution_shift import (
    value_of_ambiguity_distribution_shift,
)
from voiage.schema import ValueArray


def test_ambiguity_distribution_shift_reports_robust_and_shift_value() -> None:
    result = value_of_ambiguity_distribution_shift(
        ValueArray.from_numpy(
            np.array([[10.0, 8.0], [12.0, 7.0], [6.0, 11.0], [5.0, 13.0]]),
            ["A", "B"],
        ),
        shift_weights=np.array([[0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]),
        scenario_names=["source", "shifted"],
        scenario_probabilities=[0.5, 0.5],
        ambiguity_radius=0.1,
    )

    assert result.method_maturity == "fixture-backed"
    assert result.value > 0.0
    assert result.robust_strategy_name == "B"
    assert result.informed_optimal_strategy_names == ["A", "B"]
    assert result.shift_sensitivity.shape == (2,)
    assert result.diagnostics["drift_monitoring_status"] == "fixture-backed"


def test_ambiguity_distribution_shift_rejects_invalid_inputs() -> None:
    with pytest.raises(InputError, match="ValueArray"):
        value_of_ambiguity_distribution_shift(
            np.ones((2, 2)),  # type: ignore[arg-type]
            shift_weights=np.ones((1, 2)),
        )
    with pytest.raises(InputError, match="2D"):
        value_of_ambiguity_distribution_shift(
            ValueArray.from_numpy(np.ones(2), ["A"]),
            shift_weights=np.ones((1, 2)),
        )
    with pytest.raises(ValueError, match="shift weights"):
        value_of_ambiguity_distribution_shift(
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            shift_weights=np.ones((2, 3)),
        )
    with pytest.raises(InputError, match="match scenario count"):
        value_of_ambiguity_distribution_shift(
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            shift_weights=np.ones((2, 2)),
            scenario_probabilities=[1.0],
        )


def test_ambiguity_distribution_shift_wrapper_and_defaults() -> None:
    analysis = DecisionAnalysis(ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]))
    result = analysis.value_of_ambiguity_distribution_shift([[0.5, 0.5]])
    assert result.scenario_names == ["scenario_1"]
    assert result.scenario_probabilities.tolist() == [1.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ambiguity_radius": -1.0}, "ambiguity_radius"),
        ({"information_cost": -1.0}, "information_cost"),
        ({"shift_weights": [[0.0, 0.0]]}, "positive"),
        ({"shift_weights": [[-1.0, 2.0]]}, "non-negative"),
        ({"scenario_names": ["a", "b"]}, "scenario count"),
        ({"strategy_names": ["A"]}, "strategy count"),
        ({"scenario_probabilities": [0.0]}, "positive"),
    ],
)
def test_ambiguity_distribution_shift_rejects_additional_invalid_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "value_array": ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
        "shift_weights": [[0.5, 0.5]],
    }
    base.update(kwargs)
    with pytest.raises(InputError, match=message):
        value_of_ambiguity_distribution_shift(**base)  # type: ignore[arg-type]


def test_ambiguity_distribution_shift_rejects_nonfinite_inputs() -> None:
    with pytest.raises(InputError, match="finite"):
        value_of_ambiguity_distribution_shift(
            ValueArray.from_numpy(np.array([[np.nan, 1.0], [1.0, 1.0]]), ["A", "B"]),
            [[0.5, 0.5]],
        )
    with pytest.raises(InputError, match="finite"):
        value_of_ambiguity_distribution_shift(
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            [[0.5, 0.5]],
            scenario_probabilities=[np.nan],
        )
