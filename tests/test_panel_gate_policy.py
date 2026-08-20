"""Executable contract for the single-maintainer panel-gate workflow."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "conductor" / "panel-gate-policy.json"
ASSESSMENT = ROOT / "conductor" / "panel-gate-assessment-20260820.md"
OWNER_DECISION = ROOT / "conductor" / "owner-option-a-decision-20260821.json"


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
    assert policy["gates"][0]["current_status"] == "satisfied-experimental-only"
    assert all(gate["panel_can_authorize"] is False for gate in policy["gates"])


def test_owner_option_a_decision_is_candidate_bound_and_fail_closed() -> None:
    receipt = json.loads(OWNER_DECISION.read_text(encoding="utf-8"))

    assert receipt["accountability_model"] == "single-maintainer-scientist"
    assert receipt["candidate_commit"] == "5dcfd9765aed5caca19346b6698c126f63e2eca9"
    assert receipt["candidate_tree"] == "62c938982231051ff6f2277a2a4187ada5d68fa8"
    assert receipt["scientific_decision"] == "scientifically_acceptable_experimental"
    assert receipt["maturity_decision"] == "retain_experimental"
    assert receipt["release_authorization"] == (
        "stable_core_with_experimental_frontier_apis"
    )
    assert receipt["external_registry_and_publication"] == "pending_external"
    assert receipt["issue_closure"] == "staged_only"
    assert receipt["panel_dissent_preserved"] is True
    assert receipt["source_statement_sha256"] == (
        "3cc1cf5178a75d29cd73ed41c7cca364b1b27ec9ee0c796ac8431c23ff1547bd"
    )
