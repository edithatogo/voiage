"""Executable contract for the single-maintainer panel-gate workflow."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "conductor" / "panel-gate-policy.json"
ASSESSMENT = ROOT / "conductor" / "panel-gate-assessment-20260820.md"


def test_policy_covers_all_remaining_gate_classes() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert policy["accountability_model"] == "single-maintainer"
    assessment = policy["assessment"]
    assert assessment["mode"] == "role-separated-panel-of-agents"
    assert len(assessment["roles"]) == 4
    required = set(assessment["required_report_fields"])
    assert {
        "options",
        "tradeoffs",
        "contingencies",
        "fallback",
        "recommendation",
        "dissent",
    } <= required

    expected = {
        "scientific-validity",
        "stable-promotion",
        "cross-language-parity",
        "hosted-exact-head",
        "release",
        "publication-and-registry",
        "github-issue-and-project-closure",
    }
    gates = {gate["id"]: gate for gate in policy["gates"]}
    assert set(gates) == expected
    for gate in gates.values():
        assert gate["assessment_delegate"] == "role-separated-panel-of-agents"
        assert gate["panel_can_authorize"] is False
        assert gate["accountable_decision"]
        assert gate["current_status"]
        assert gate["preconditions"]
        assert gate["contingency"]


def test_policy_and_assessment_preserve_experimental_boundary() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert "experimental" in ASSESSMENT.read_text(encoding="utf-8")
    assert "never an approval" in policy["assessment"]["authorization_boundary"]
    assert policy["gates"][0]["current_status"] == "pending"
    assert all(gate["panel_can_authorize"] is False for gate in policy["gates"])
