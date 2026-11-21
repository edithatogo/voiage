"""Installed Python controls for exact native credal decision certificates."""

from fractions import Fraction
from importlib import import_module
import json

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest


def _problem() -> dict[str, object]:
    return {
        "parameters": [{"id": "x", "lower": 0.0, "upper": 1.0}],
        "inequalities": [],
        "strategies": [
            {"strategy_id": "left", "intercept": 0.0, "coefficients": [1.0]},
            {"strategy_id": "right", "intercept": 1.0, "coefficients": [-1.0]},
            {"strategy_id": "target", "intercept": 0.0, "coefficients": [0.0]},
        ],
    }


def _fraction(pair: list[str]) -> Fraction:
    return Fraction(int(pair[0]), int(pair[1]))


def test_native_credal_witness_preserves_an_exact_nonbinary_boundary() -> None:
    native = import_module("voiage._core")
    problem = {
        "parameters": [
            {"id": "x", "lower": 0.0, "upper": 1.0},
            {"id": "other_coordinate", "lower": 0.0, "upper": 1.0},
        ],
        "inequalities": [
            {"coefficients": [3.0, 0.0], "upper": 1.0},
            {"coefficients": [-3.0, 0.0], "upper": -1.0},
        ],
        "strategies": [
            {
                "strategy_id": "proportional",
                "intercept": 0.0,
                "coefficients": [3.0, 0.0],
            },
            {"strategy_id": "constant", "intercept": 1.0, "coefficients": [0.0, 0.0]},
        ],
    }
    result = native.compute_credal_decisions(json.dumps(problem), 128)
    assert result["contract_version"] == 1
    assert result["status"] == "feasible"
    assert result["parameter_ids"] == ["x", "other_coordinate"]
    assert (
        result["gamma_maximin"]
        == result["gamma_maximax"]
        == ["constant", "proportional"]
    )
    witness = result["e_admissibility"]["proportional"]
    assert witness["status"] == "admissible"
    assert witness["point"][0] == ["1", "3"]
    assert result["ranges"]["proportional"]["minimum"]["value"] == ["1", "1"]


@settings(derandomize=True, max_examples=32, deadline=None)
@given(
    left=st.integers(min_value=1, max_value=32),
    right=st.integers(min_value=1, max_value=32),
)
def test_native_credal_mixture_matches_the_exact_crossing_line_oracle(
    left: int, right: int
) -> None:
    native = import_module("voiage._core")
    problem = _problem()
    problem["strategies"] = [
        {"strategy_id": "left", "intercept": 0.0, "coefficients": [float(left)]},
        {
            "strategy_id": "right",
            "intercept": float(right),
            "coefficients": [-float(right)],
        },
        {"strategy_id": "target", "intercept": 0.0, "coefficients": [0.0]},
    ]
    result = native.compute_credal_decisions(json.dumps(problem), 128)
    witness = result["e_admissibility"]["target"]
    assert witness["status"] == "dominated"
    assert _fraction(witness["alternative_multipliers"]["left"]) == Fraction(
        right, left + right
    )
    assert _fraction(witness["alternative_multipliers"]["right"]) == Fraction(
        left, left + right
    )
    assert _fraction(witness["margin"]) == Fraction(left * right, left + right)


def test_native_credal_binding_preserves_incomplete_and_invalid_inputs() -> None:
    native = import_module("voiage._core")
    problem = _problem()
    encoded = json.dumps(problem)
    assert native.compute_credal_decisions(encoded, 1) == {
        "contract_version": 1,
        "status": "indeterminate",
        "parameter_ids": ["x"],
        "bases_examined": 1,
    }
    with pytest.raises(native.InputError, match="basis budget"):
        native.compute_credal_decisions(encoded, 0)
    problem["inequalities"] = [{"coefficients": [1.0, 2.0], "upper": 1.0}]
    with pytest.raises(native.DimensionMismatchError):
        native.compute_credal_decisions(json.dumps(problem), 128)


@settings(derandomize=True, max_examples=32, deadline=None)
@given(intercept=st.floats(allow_nan=False, allow_infinity=False, width=64))
def test_native_credal_json_preserves_the_supplied_binary64_input(
    intercept: float,
) -> None:
    native = import_module("voiage._core")
    problem = _problem()
    problem["strategies"] = [
        {"strategy_id": "constant", "intercept": intercept, "coefficients": [0.0]},
    ]
    result = native.compute_credal_decisions(json.dumps(problem), 128)
    extremum = result["ranges"]["constant"]["minimum"]
    assert _fraction(extremum["value"]) == Fraction(intercept)
