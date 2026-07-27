"""Governance contract for the comprehensive Rust-first programme."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_rust_polyglot_programme import (
    EXPECTED_PROJECT_VIEWS,
    FRONTIER_PARENT_ISSUE,
    FRONTIER_SUBISSUES,
    FRONTIER_TRACK,
    INDUSTRY_SUBISSUES,
    PARENT_ISSUE,
    PARENT_TRACK,
    SUBISSUE_GROUPS,
    TRACK_ISSUES,
    missing_frontier_subissues,
    missing_required_subissues,
    missing_track_subissues,
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


def test_frontier_track_has_governed_native_method_gap_subissues() -> None:
    assert TRACK_ISSUES[FRONTIER_TRACK] == FRONTIER_PARENT_ISSUE == 318
    assert set(range(556, 561)) < set(FRONTIER_SUBISSUES)
    assert {570, 571, 572, 582} <= set(FRONTIER_SUBISSUES)
    assert {fields["record id"] for fields in FRONTIER_SUBISSUES.values()} >= {
        "deterministic-sensitivity-analysis",
        "value-of-distributional-information",
        "qualitative-voi",
        "value-of-flexibility",
        "mcda-voi",
        "risk-sensitive-constrained-voi",
        "information-source-portfolio-voi",
        "experiment-portfolio-voi",
        "forecast-signal-information-voi",
    }
    assert missing_frontier_subissues(set(FRONTIER_SUBISSUES)) == set()
    assert missing_frontier_subissues(set(FRONTIER_SUBISSUES) - {557}) == {557}


def test_industry_and_adoption_subissues_cover_every_approved_initiative() -> None:
    assert set(INDUSTRY_SUBISSUES) == set(range(565, 585))
    assert {fields["record id"] for fields in INDUSTRY_SUBISSUES.values()} == {
        "landscape-open-source-inventory",
        "industry-decision-problem-contract",
        "landscape-gap-review-roadmap-proposal",
        "landscape-commercial-hosted-inventory",
        "landscape-schema-review-protocol",
        "risk-sensitive-constrained-voi",
        "experiment-portfolio-voi",
        "forecast-signal-information-voi",
        "landscape-capability-adoption-map",
        "churn-retention-policy-voi-example",
        "industry-domain-example-packs",
        "decision-focused-model-value",
        "domain-template-adapter-registry",
        "policy-uplift-voi",
        "industry-decision-contract-binding-parity",
        "decision-registry-cards",
        "local-decision-studio-reporting",
        "information-source-portfolio-voi",
        "enterprise-integration-adapters",
        "decision-correctness-industry-assurance",
    }


def test_industry_subissues_have_one_existing_track_parent() -> None:
    expected = {
        "voi_method_census_contract_reconciliation_20260723": {566},
        "external_voi_library_feature_parity_20260723": {
            565,
            567,
            568,
            569,
            573,
        },
        "supported_frontier_method_completion_20260723": {
            556,
            557,
            558,
            559,
            560,
            570,
            571,
            572,
            582,
        },
        "ml_llm_agent_voi_20260723": {576, 578},
        "polyglot_abi_binding_parity_20260723": {579},
        "datasets_worked_examples_20260723": {574, 575, 577},
        "quality_release_automation_20260723": {580, 581, 583, 584},
    }
    assert {
        track_id: set(group["issues"]) for track_id, group in SUBISSUE_GROUPS.items()
    } == expected
    for track_id, issue_numbers in expected.items():
        assert missing_track_subissues(track_id, issue_numbers) == set()
        removed = min(issue_numbers)
        assert missing_track_subissues(track_id, issue_numbers - {removed}) == {removed}


def test_moscow_and_mermaid_contracts_cover_industry_decision_value() -> None:
    requirements = (REPO_ROOT / "conductor" / "requirements.md").read_text(
        encoding="utf-8"
    )
    design = (REPO_ROOT / "conductor" / "design.md").read_text(encoding="utf-8")
    for token in (
        "risk-sensitive and constrained VOI",
        "policy and uplift VOI",
        "information-source portfolio",
        "experiment-portfolio VOI",
        "forecast and signal information",
        "Decision Registry",
        "customer churn",
        "commercial and open-source software",
    ):
        assert token.casefold() in requirements.casefold()
    for token in (
        "Decision-value middleware",
        "Customer churn and retention",
        "Software landscape review",
        "Decision Registry and Decision Studio",
    ):
        assert token.casefold() in design.casefold()


def test_project_views_cover_delivery_priority_risk_and_review_workflows() -> None:
    assert EXPECTED_PROJECT_VIEWS == {
        "Current Delivery": {
            "layout": "TABLE_LAYOUT",
            "filter": 'status:"In Progress"',
        },
        "Next: Software Landscape": {
            "layout": "TABLE_LAYOUT",
            "filter": 'track-id:"external_voi_library_feature_parity_20260723"',
        },
        "MoSCoW & Priority": {
            "layout": "BOARD_LAYOUT",
            "filter": "",
        },
        "Industry & Adoption": {
            "layout": "TABLE_LAYOUT",
            "filter": 'record-type:"Development ledger"',
        },
        "Gates & High Risk": {
            "layout": "TABLE_LAYOUT",
            "filter": "risk-level:High",
        },
        "Evidence & Review Due": {
            "layout": "TABLE_LAYOUT",
            "filter": "evidence-state:Unverified",
        },
    }


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
