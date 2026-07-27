"""Governance contract for the comprehensive Rust-first programme."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_rust_polyglot_programme import (
    FRONTIER_PARENT_ISSUE,
    FRONTIER_SUBISSUES,
    FRONTIER_TRACK,
    PARENT_ISSUE,
    PARENT_TRACK,
    TRACK_ISSUES,
    missing_frontier_subissues,
    missing_required_subissues,
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


def test_programme_allows_additional_governed_historical_subissues() -> None:
    observed = set(TRACK_ISSUES.values()) - {PARENT_ISSUE}
    observed.add(416)

    assert missing_required_subissues(observed) == set()
    assert missing_required_subissues(observed - {314}) == {314}


def test_frontier_track_has_five_native_method_gap_subissues() -> None:
    assert TRACK_ISSUES[FRONTIER_TRACK] == FRONTIER_PARENT_ISSUE == 318
    assert set(FRONTIER_SUBISSUES) == set(range(556, 561))
    assert {
        fields["record id"] for fields in FRONTIER_SUBISSUES.values()
    } == {
        "deterministic-sensitivity-analysis",
        "value-of-distributional-information",
        "qualitative-voi",
        "value-of-flexibility",
        "mcda-voi",
    }
    assert missing_frontier_subissues(set(FRONTIER_SUBISSUES)) == set()
    assert missing_frontier_subissues(set(FRONTIER_SUBISSUES) - {557}) == {557}


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
