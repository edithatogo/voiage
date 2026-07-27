"""Regression contract for deterministic adversarial ML/agent VOI fixtures."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "specs/frontier/ai-assisted-evidence-triage/v1/fixtures/normative"
    / "adversarial-scenarios.json"
)


def test_adversarial_fixture_covers_high_risk_information_failures() -> None:
    """Offline scenarios must model information actions and decision losses."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]
    ids = {scenario["id"] for scenario in scenarios}

    assert payload["schema_version"] == "1.0.0"
    assert {
        "prompt-injection",
        "retrieval-poisoning",
        "correlated-judge-failure",
        "provider-drift",
        "human-escalation",
    } <= ids
    assert all(scenario["information_action"] for scenario in scenarios)
    assert all(scenario["decision_loss_if_missed"] > 0 for scenario in scenarios)
    assert all(scenario["review_cost"] >= 0 for scenario in scenarios)
