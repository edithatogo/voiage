"""Tests for Decision Cards, Decision Registry, and Signed Result Bundles (#580)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiage.decision_card import (
    DecisionBundle,
    DecisionCard,
    DecisionProblemSnapshot,
    Governance,
    HumanApproval,
    InformationValuation,
    Lineage,
    ResidualUncertainty,
    SelectedPolicy,
    create_decision_card,
    validate_decision_card,
)
from voiage.exceptions import InputError


def _sample_card() -> DecisionCard:
    dp = DecisionProblemSnapshot(
        problem_id="churn_campaign_2026",
        title="Customer Churn Retention Policy Selection",
        alternatives=["Status Quo", "Automated Discount", "Concierge Outreach"],
        criterion="Maximize Expected Net Benefit",
    )
    sp = SelectedPolicy(
        name="Automated Discount",
        rationale="Maximizes expected net benefit given estimated customer lifetime value.",
        expected_net_benefit=150000.0,
        baseline_comparison="+$50,000 vs Status Quo",
    )
    iv = InformationValuation(
        evpi=25000.0,
        evppi={"clv": 15000.0, "treatment_effect": 12000.0},
        evsi={"pilot_n500": 18000.0},
        enbs={"pilot_n500": 8000.0},
        recommended_information_action="Pilot trial with n=500 before full rollout",
    )
    ru = ResidualUncertainty(
        top_drivers=["treatment_effect", "clv"],
        risk_quantiles={"p05": 110000.0, "p50": 150000.0, "p95": 190000.0},
        sensitivity_summary="Policy remains optimal if treatment effect >= 0.08.",
    )
    ha = HumanApproval(
        approver="Chief Commercial Officer",
        approved_at="2026-08-22T12:00:00Z",
        rationale="Approved pilot trial based on positive ENBS.",
    )
    gov = Governance(
        owner="Marketing Analytics Team",
        reviewers=["Decision Science Lead", "Finance Director"],
        human_approval=ha,
        expiry_date="2027-08-22",
        refresh_cadence="quarterly",
    )
    lin = Lineage(
        model_version="2.1.0",
        input_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        dataset_version="customer_churn_v1",
        code_version="v2.1.0",
    )
    return DecisionCard(
        decision_id="dec_churn_2026_q3",
        version="1.0.0",
        title="Q3 Retention Campaign Selection & Trial VOI",
        status="approved",
        created_at="2026-08-22T10:00:00Z",
        decision_problem=dp,
        selected_policy=sp,
        information_valuation=iv,
        residual_uncertainty=ru,
        governance=gov,
        lineage=lin,
    )


def test_decision_card_json_schema_validation() -> None:
    card = _sample_card()
    card_dict = card.to_dict()
    assert validate_decision_card(card_dict) is True


def test_decision_card_serialization_round_trip() -> None:
    card = _sample_card()
    json_str = card.to_json()
    reconstructed = DecisionCard.from_json(json_str)
    assert reconstructed == card
    assert reconstructed.compute_hash() == card.compute_hash()


def test_create_decision_card_helper() -> None:
    dp = DecisionProblemSnapshot(
        problem_id="prob_1",
        title="Problem 1",
        alternatives=["A", "B"],
        criterion="Max Benefit",
    )
    sp = SelectedPolicy(name="A", rationale="Optimal", expected_net_benefit=100.0)
    iv = InformationValuation(evpi=10.0)

    card = create_decision_card(
        decision_id="dec_test_01",
        title="Test Decision",
        decision_problem=dp,
        selected_policy=sp,
        information_valuation=iv,
        owner="Alice",
        reviewers=["Bob"],
    )
    assert card.decision_id == "dec_test_01"
    assert card.status == "draft"
    assert card.governance.owner == "Alice"
    assert card.governance.human_approval is None
    assert card.to_dict()["status"] == "draft"


def test_decision_card_markdown_and_html_rendering() -> None:
    card = _sample_card()
    md = card.to_markdown()
    assert "# Decision Card:" in md
    assert "Chief Commercial Officer" in md
    assert "$150,000.00" in md

    html = card.to_html()
    assert "<!DOCTYPE html>" in html
    assert "APPROVED" in html
    assert "Expected Net Benefit" in html


def test_decision_bundle_signing_and_verification() -> None:
    card = _sample_card()
    inputs = {"raw_sample_count": 10000, "parameter_distributions": ["normal", "beta"]}

    bundle = DecisionBundle(card=card, input_payload=inputs)
    assert bundle.bundle_hash != ""
    assert bundle.verify_integrity() is True

    # Tamper test: modify input payload
    tampered_inputs = {"raw_sample_count": 9999}
    tampered_bundle = DecisionBundle(
        card=card, input_payload=tampered_inputs, bundle_hash=bundle.bundle_hash
    )
    assert tampered_bundle.verify_integrity() is False


def test_decision_card_error_handling() -> None:
    with pytest.raises(InputError, match="must be a dictionary"):
        DecisionCard.from_dict("invalid")  # type: ignore[arg-type]

    non_existent = Path("specs/decision-cards/schemas/v1/missing.schema.json")
    with pytest.raises(InputError, match="not found"):
        validate_decision_card({}, schema_path=non_existent)
