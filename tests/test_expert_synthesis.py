"""Runtime tests for expert elicitation and evidence synthesis VOI."""

import numpy as np
import pytest

from voiage.methods.expert_synthesis import value_of_expert_synthesis


def test_expert_synthesis_returns_profile_specific_decisions() -> None:
    result = value_of_expert_synthesis(
        np.array([[[10.0, 9.8], [11.5, 11.0], [11.2, 11.6]]]),
        ["structured", "delphi"],
        ["status-quo", "elicit", "synthesize"],
        np.zeros((2, 3)),
        np.zeros((2, 3)),
    )
    assert result.method_maturity == "fixture-backed"
    assert result.expected_net_benefits.shape == (2, 3)
    assert result.optimal_strategy_by_expert_profile == {
        "structured": "elicit",
        "delphi": "synthesize",
    }


def test_expert_synthesis_rejects_negative_penalties() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        value_of_expert_synthesis(
            np.ones((1, 1, 1)),
            ["profile"],
            ["strategy"],
            np.zeros((1, 1)),
            np.ones((1, 1)) * -1,
        )


@pytest.mark.parametrize(
    "case", ["shape", "matrix", "duplicate", "finite", "reference"]
)
def test_expert_synthesis_rejects_boundary_inputs(case: str) -> None:
    values = np.ones((1, 1, 1))
    profiles = ["profile"]
    strategies = ["strategy"]
    matrix = np.zeros((1, 1))
    reference = None
    if case == "shape":
        values = np.ones((1, 2, 1))
    elif case == "matrix":
        matrix = np.zeros((2, 1))
    elif case == "duplicate":
        profiles = ["x", "x"]
        values = np.ones((1, 1, 2))
        matrix = np.zeros((2, 1))
    elif case == "finite":
        values = np.array([[[np.nan]]])
    else:
        reference = "missing"
    with pytest.raises(ValueError):
        value_of_expert_synthesis(
            values,
            profiles,
            strategies,
            matrix,
            matrix,
            reference_expert_profile=reference,
        )
