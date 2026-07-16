"""Runtime tests for data-quality and privacy VOI."""

import numpy as np
import pytest

from voiage.methods.data_quality import value_of_data_quality


def test_data_quality_returns_profile_specific_decisions() -> None:
    result = value_of_data_quality(
        np.array([[[10.0, 9.0], [12.0, 11.4], [11.5, 12.1]]]),
        ["clean", "noisy"],
        ["status-quo", "collect", "link"],
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.ones((2, 3)),
    )
    assert result.method_maturity == "fixture-backed"
    assert result.expected_net_benefits.shape == (2, 3)
    assert result.optimal_strategy_by_data_quality_profile == {
        "clean": "collect",
        "noisy": "link",
    }
    assert result.robust_strategy in result.strategy_names


def test_data_quality_rejects_invalid_linkage_weights() -> None:
    with pytest.raises(ValueError, match="Linkage weights"):
        value_of_data_quality(
            np.ones((1, 1, 1)),
            ["profile"],
            ["strategy"],
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.ones((1, 1)) * 2,
        )


@pytest.mark.parametrize(
    "case", ["shape", "matrix", "duplicate", "finite", "error", "reference"]
)
def test_data_quality_rejects_boundary_inputs(case: str) -> None:
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
    elif case == "error":
        matrix = np.full((1, 1), 2.0)
    else:
        reference = "missing"
    with pytest.raises(ValueError):
        value_of_data_quality(
            values,
            profiles,
            strategies,
            matrix,
            matrix,
            matrix,
            matrix,
            reference_data_quality_profile=reference,
        )
