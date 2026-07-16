"""Contract tests for ambiguity and distribution-shift VOI."""

import json
from pathlib import Path

import numpy as np
import pytest

from voiage.methods.ambiguity_distribution_shift import (
    value_of_ambiguity_distribution_shift,
)
from voiage.schema import ValueArray


def test_ambiguity_distribution_shift_fixture_is_deterministic() -> None:
    root = Path("specs/frontier/ambiguity-distribution-shift/v1/fixtures")
    source = json.loads(
        (root / "normative/ambiguity-distribution-shift-input.json").read_text()
    )
    expected = json.loads(
        (root / "normative/value-of-ambiguity-distribution-shift.json").read_text()
    )
    result = value_of_ambiguity_distribution_shift(
        ValueArray.from_numpy(
            np.asarray(source["net_benefit"]), source["strategy_names"]
        ),
        source["shift_weights"],
        source["strategy_names"],
        source["scenario_names"],
        source["scenario_probabilities"],
        source["ambiguity_radius"],
        source["information_cost"],
    )
    assert result.value == pytest.approx(expected["value"])
    assert result.robust_strategy_name == expected["robust_strategy_name"]
    assert (
        result.informed_optimal_strategy_names
        == expected["informed_optimal_strategy_names"]
    )
    np.testing.assert_allclose(result.scenario_regret, expected["scenario_regret"])


def test_ambiguity_distribution_shift_fixture_documents_external_gates() -> None:
    evidence = json.loads(
        Path(
            "specs/frontier/ambiguity-distribution-shift/v1/fixtures/evidence.json"
        ).read_text()
    )
    assert evidence["open_data_gate"]["status"] == "blocked"
    assert evidence["parity_gate"]["status"] == "deferred"
