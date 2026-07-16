"""Tests for federated and privacy-preserving VOI."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.federated_privacy_preserving import (
    value_of_federated_privacy_preserving,
)


def _inputs() -> dict[str, object]:
    return {
        "site_summaries": [[8.0, 7.0], [6.0, 9.0], [7.0, 8.0]],
        "site_weights": [0.2, 0.5, 0.3],
        "privacy_budgets": [1.0, 0.8, 1.2],
        "prior_strategy_values": [6.5, 7.0],
        "strategy_names": ["status_quo", "privacy_preserving"],
        "noise_scale": 0.0,
        "individual_data_access": "blocked",
        "seed": 0,
    }


def test_federated_result_reports_aggregate_and_site_contributions() -> None:
    result = value_of_federated_privacy_preserving(**_inputs())
    assert result.method_maturity == "fixture-backed"
    assert result.selected_strategy == "privacy_preserving"
    assert result.aggregated_net_benefits.tolist() == pytest.approx([6.7, 8.3])
    assert result.site_contribution_values.shape == (3,)
    assert result.privacy_loss == pytest.approx(sum(1 / x for x in [1.0, 0.8, 1.2]))
    assert result.diagnostics["individual_data_access"] == "blocked"
    assert result.diagnostics["parity_status"] == "deferred"


def test_synthetic_fallback_is_used_when_individual_access_is_blocked() -> None:
    payload = _inputs()
    payload["site_summaries"] = [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]
    payload["synthetic_site_summaries"] = [[5.0, 6.0], [7.0, 6.0], [6.0, 8.0]]
    result = value_of_federated_privacy_preserving(**payload)
    assert result.diagnostics["synthetic_fallback_used"] is True
    assert result.selected_strategy == "privacy_preserving"


def test_privacy_noise_is_reproducible_and_analysis_wrapper_is_available() -> None:
    payload = _inputs()
    payload["noise_scale"] = 0.2
    first = value_of_federated_privacy_preserving(**payload)
    second = value_of_federated_privacy_preserving(**payload)
    np.testing.assert_array_equal(
        first.aggregated_net_benefits, second.aggregated_net_benefits
    )
    wrapped = DecisionAnalysis(np.ones((2, 2))).value_of_federated_privacy_preserving(
        **_inputs()
    )
    assert wrapped.selected_strategy == "privacy_preserving"


@pytest.mark.parametrize(
    "overrides",
    [
        {"site_summaries": [[1.0, 2.0]]},
        {"site_summaries": [[1.0, float("nan")]]},
        {"site_weights": [1.0]},
        {"privacy_budgets": [0.0, 1.0, 1.0]},
        {"individual_data_access": "available"},
        {"noise_scale": -1.0},
    ],
)
def test_federated_inputs_are_validated(overrides: dict[str, object]) -> None:
    payload = _inputs()
    payload.update(overrides)
    with pytest.raises(Exception):
        value_of_federated_privacy_preserving(**payload)
