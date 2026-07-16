"""Runtime tests for equity-information VOI."""

from typing import cast

import numpy as np
import pytest
import xarray as xr

from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.equity_information import value_of_equity_information
from voiage.schema import ValueArray


def test_equity_information_values_resolution_of_equity_uncertainty() -> None:
    result = value_of_equity_information(
        ValueArray.from_numpy(
            np.array([[10.0, 8.0], [12.0, 7.0], [6.0, 11.0], [5.0, 13.0]]),
            ["A", "B"],
        ),
        subgroups=["low", "low", "high", "high"],
        equity_weights=[0.5, 0.5],
        resolved_equity_weights=np.array([[0.8, 0.2], [0.2, 0.8]]),
        scenario_probabilities=[0.5, 0.5],
        policy_strata=["protected", "policy-relevant"],
    )

    assert result.method_maturity == "fixture-backed"
    assert result.value > 0.0
    assert result.baseline_optimal_strategy_name == "B"
    assert result.resolved_optimal_strategy_names == ["B", "A"]
    assert result.diagnostics["n_equity_scenarios"] == 2
    assert result.diagnostics["policy_strata"] == ["protected", "policy-relevant"]


def test_equity_information_rejects_malformed_scenarios() -> None:
    value_array = ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"])
    with pytest.raises(ValueError, match="scenario probabilities"):
        value_of_equity_information(
            value_array,
            subgroups=["low", "high"],
            equity_weights=[0.5, 0.5],
            resolved_equity_weights=np.ones((2, 2)),
            scenario_probabilities=[1.0],
        )


def test_equity_information_is_zero_when_resolution_cannot_change_decision() -> None:
    value_array = ValueArray.from_numpy(np.array([[5.0, 1.0], [4.0, 2.0]]), ["A", "B"])
    result = value_of_equity_information(
        value_array,
        subgroups=["low", "high"],
        equity_weights=[0.5, 0.5],
        resolved_equity_weights=np.array([[0.5, 0.5]]),
    )
    assert result.value == pytest.approx(0.0)


def test_decision_analysis_wraps_equity_information() -> None:
    analysis = DecisionAnalysis(ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]))
    result = analysis.value_of_equity_information(
        ["low", "high"], [0.5, 0.5], [[0.5, 0.5]]
    )
    assert result.method_maturity == "fixture-backed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"information_cost": -1.0}, "information_cost"),
        ({"equity_weights": [1.0]}, "one value per subgroup"),
        ({"equity_weights": [-1.0, 1.0]}, "non-negative"),
        ({"resolved_equity_weights": [[0.0, 0.0]]}, "must be positive"),
        (
            {"resolved_equity_weights": [[1.0, 0.0]], "policy_strata": []},
            "policy-relevant",
        ),
        ({"strategy_names": ["A"]}, "strategy count"),
    ],
)
def test_equity_information_rejects_invalid_contract_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "value_array": ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
        "subgroups": ["low", "high"],
        "equity_weights": [0.5, 0.5],
        "resolved_equity_weights": [[0.5, 0.5]],
    }
    base.update(kwargs)
    with pytest.raises(InputError, match=message):
        value_of_equity_information(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value_array", "subgroups", "resolved", "probabilities", "message"),
    [
        (
            cast("ValueArray", "invalid"),
            ["low", "high"],
            [[0.5, 0.5]],
            None,
            "ValueArray",
        ),
        (
            ValueArray.from_numpy(np.array([[np.nan, 1.0], [1.0, 1.0]]), ["A", "B"]),
            ["low", "high"],
            [[0.5, 0.5]],
            None,
            "finite",
        ),
        (
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            ["low"],
            [[0.5, 0.5]],
            None,
            "samples",
        ),
        (
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            ["low", "high"],
            [[0.5]],
            None,
            "scenario x subgroup",
        ),
        (
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            ["low", "high"],
            [[-1.0, 2.0]],
            None,
            "non-negative",
        ),
        (
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            ["low", "high"],
            [[0.5, 0.5]],
            [np.nan],
            "finite and non-negative",
        ),
        (
            ValueArray.from_numpy(np.ones((2, 2)), ["A", "B"]),
            ["low", "high"],
            [[0.5, 0.5]],
            [0.0],
            "sum to a positive",
        ),
    ],
)
def test_equity_information_rejects_additional_invalid_inputs(
    value_array: ValueArray,
    subgroups: list[str],
    resolved: list[list[float]],
    probabilities: list[float] | None,
    message: str,
) -> None:
    with pytest.raises(InputError, match=message):
        value_of_equity_information(
            value_array,
            subgroups,
            [0.5, 0.5],
            resolved,
            scenario_probabilities=probabilities,
        )


def test_equity_information_rejects_non_2d_value_array() -> None:
    invalid = object.__new__(ValueArray)
    object.__setattr__(
        invalid,
        "dataset",
        xr.Dataset(
            {"net_benefit": (("n_samples",), [1.0, 2.0])},
            coords={"strategy": ["A", "B"]},
        ),
    )
    with pytest.raises(InputError, match="2D"):
        value_of_equity_information(invalid, ["low", "high"], [0.5, 0.5], [[0.5, 0.5]])
