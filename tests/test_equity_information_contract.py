"""Contract tests for equity-information VOI fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from voiage.methods.equity_information import value_of_equity_information
from voiage.schema import ValueArray


def test_equity_information_fixture_is_deterministic() -> None:
    root = Path("specs/frontier/equity-information/v1/fixtures")
    source = json.loads((root / "normative/equity-information-input.json").read_text())
    expected = json.loads(
        (root / "normative/value-of-equity-information.json").read_text()
    )
    result = value_of_equity_information(
        ValueArray.from_numpy(
            np.asarray(source["net_benefit"]), source["strategy_names"]
        ),
        source["subgroups"],
        source["equity_weights"],
        source["resolved_equity_weights"],
        source["scenario_probabilities"],
        source["information_cost"],
        source["strategy_names"],
        source["policy_strata"],
    )
    assert result.value == pytest.approx(expected["value"])
    assert (
        result.baseline_optimal_strategy_name
        == expected["baseline_optimal_strategy_name"]
    )
    assert (
        result.resolved_optimal_strategy_names
        == expected["resolved_optimal_strategy_names"]
    )
    np.testing.assert_allclose(
        result.resolved_social_welfare, expected["resolved_social_welfare"]
    )
    assert result.method_maturity == "fixture-backed"


def test_equity_information_fixture_documents_external_gates() -> None:
    evidence = json.loads(
        Path("specs/frontier/equity-information/v1/fixtures/evidence.json").read_text()
    )
    assert evidence["open_data_gate"]["status"] == "blocked"
    assert evidence["parity_gate"]["status"] == "deferred"
