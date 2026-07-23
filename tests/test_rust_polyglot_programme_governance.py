"""Governance contract for the comprehensive Rust-first programme."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_rust_polyglot_programme import (
    PARENT_ISSUE,
    PARENT_TRACK,
    TRACK_ISSUES,
    validate_local,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_programme_has_parent_and_ten_child_tracks() -> None:
    assert PARENT_TRACK in TRACK_ISSUES
    assert TRACK_ISSUES[PARENT_TRACK] == PARENT_ISSUE
    assert len(TRACK_ISSUES) == 11
    assert set(TRACK_ISSUES.values()) == set(range(313, 324))


def test_programme_local_governance_is_consistent() -> None:
    assert validate_local(REPO_ROOT) == []


def test_programme_covers_every_approved_workstream() -> None:
    required = {
        "voi_method_census_contract_reconciliation_20260723",
        "external_voi_library_feature_parity_20260723",
        "stable_voi_rust_core_completion_20260723",
        "value_of_perspective_completion_20260723",
        "supported_frontier_method_completion_20260723",
        "ml_llm_agent_voi_20260723",
        "polyglot_abi_binding_parity_20260723",
        "datasets_worked_examples_20260723",
        "quality_release_automation_20260723",
        "research_contribution_ai_transparency_20260723",
    }
    assert set(TRACK_ISSUES) - {PARENT_TRACK} == required
