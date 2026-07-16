"""Runtime tests for monitoring and surveillance VOI."""

import numpy as np
import pytest

from voiage.methods.monitoring_surveillance import value_of_monitoring_surveillance


def test_monitoring_surveillance_reports_revision_and_stopping() -> None:
    result = value_of_monitoring_surveillance(
        np.array([[[10.0, 10.4], [11.0, 11.5]]]),
        ["status-quo", "monitor"],
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.array([[0.2, 0.4], [0.6, 0.8]]),
    )
    assert result.method_maturity == "fixture-backed"
    assert result.stopping_period == 1
    assert result.optimal_strategy_by_period["1"] == "monitor"


def test_monitoring_surveillance_rejects_invalid_revision_probability() -> None:
    with pytest.raises(ValueError, match="revision probabilities"):
        value_of_monitoring_surveillance(
            np.ones((1, 1, 1)),
            ["strategy"],
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.ones((1, 1)) * 2,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("shape", "period x strategy"),
        ("duplicate", "unique"),
        ("finite", "finite"),
        ("frequency", "positive"),
        ("threshold", "[0, 1]"),
    ],
)
def test_monitoring_surveillance_rejects_boundary_inputs(
    change: str, message: str
) -> None:
    values = np.ones((1, 1, 1))
    strategies = ["strategy"]
    matrix = np.zeros((1, 1))
    kwargs: dict[str, object] = {}
    if change == "shape":
        kwargs["monitoring_costs"] = np.zeros((2, 1))
    elif change == "duplicate":
        strategies = ["x", "x"]
        values = np.ones((1, 2, 1))
        matrix = np.zeros((1, 2))
    elif change == "finite":
        values = np.array([[[np.nan]]])
    elif change == "frequency":
        kwargs["surveillance_frequency"] = 0.0
    else:
        kwargs["stopping_threshold"] = 2.0
    with pytest.raises(ValueError, match=message):
        value_of_monitoring_surveillance(
            values,
            strategies,
            kwargs.pop("monitoring_costs", matrix),
            matrix,
            matrix,
            matrix,
            **kwargs,
        )
