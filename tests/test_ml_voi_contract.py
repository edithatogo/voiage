"""Contract tests distinguishing information gain from decision VOI."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "specs/frontier/ai-assisted-evidence-triage/v1/fixtures/normative"
    / "eig-versus-voi.json"
)


def test_eig_fixture_keeps_information_and_decision_values_distinct() -> None:
    """Entropy reduction must not be mislabeled as economic VOI."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["expected_information_gain_nats"] > 0
    assert payload["expected_decision_voi"] == (
        payload["posterior_expected_utility"]
        - payload["current_expected_utility"]
        - payload["information_cost"]
    )
    assert "utility" in payload["interpretation"]
    assert "cost" in payload["interpretation"]
    assert payload["expected_decision_voi"] != payload["expected_information_gain_nats"]
