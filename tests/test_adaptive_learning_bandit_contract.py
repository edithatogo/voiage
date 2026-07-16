"""Frontier contract tests for adaptive learning and bandit VOI."""

import json
from pathlib import Path

import pytest

from voiage.methods.adaptive_learning_bandit import (
    value_of_adaptive_learning_bandit,
)

ROOT = Path("specs/frontier/adaptive-learning-bandit/v1/fixtures")


def test_bandit_normative_fixture_is_deterministic() -> None:
    payload = json.loads(
        (ROOT / "normative/adaptive-learning-bandit-input.json").read_text()
    )
    expected = json.loads(
        (ROOT / "normative/value-of-adaptive-learning-bandit.json").read_text()
    )
    result = value_of_adaptive_learning_bandit(**payload)
    assert result.selected_arms.tolist() == expected["selected_arms"]
    assert result.total_reward == pytest.approx(expected["total_reward"])
    assert result.regret == pytest.approx(expected["regret"])
    assert result.method_maturity == expected["method_maturity"]


def test_bandit_evidence_keeps_promotion_blocked() -> None:
    evidence = json.loads((ROOT / "evidence.json").read_text())
    assert evidence["stable_promotion"] is False
    assert "licensed" in evidence["open_data_status"]
    assert evidence["parity_status"].startswith("deferred")
