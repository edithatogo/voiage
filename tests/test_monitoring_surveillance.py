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
