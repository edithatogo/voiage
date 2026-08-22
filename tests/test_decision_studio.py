"""Tests for Local Decision Studio and Business Reporting (#581)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiage.decision_studio import (
    DecisionStudioSession,
    compute_expected_loss,
    compute_mcda_scores,
    create_decision_studio_session,
    load_decision_studio_fixture,
    validate_decision_studio_session,
)
from voiage.exceptions import InputError


def test_load_and_validate_decision_studio_fixture() -> None:
    session = load_decision_studio_fixture()
    assert session.session_id == "studio_session_pricing_q3"
    assert len(session.scenarios) == 3
    assert session.expected_losses["Usage-Based Model"] == 0.0
    assert session.voi_summary["evpi"] == 45000.0
    assert session.mcda_evaluation is not None
    assert validate_decision_studio_session(session.to_dict()) is True


def test_create_and_validate_decision_studio_session() -> None:
    base_payoffs = {
        "Status Quo": 100000.0,
        "Option A": 140000.0,
        "Option B": 130000.0,
    }
    scenarios_adjustments = {
        "High Inflation": {"Option A": 0.8, "Option B": 1.1, "Status Quo": 0.9},
        "Tech Boom": {"global_multiplier": 1.25},
    }
    mcda_criteria_matrix = {
        "Status Quo": {"cost": 90.0, "reach": 50.0},
        "Option A": {"cost": 60.0, "reach": 85.0},
        "Option B": {"cost": 75.0, "reach": 80.0},
    }
    mcda_weights = {"cost": 0.4, "reach": 0.6}

    session = create_decision_studio_session(
        session_id="session_test_01",
        title="Marketing Strategy Decision Studio",
        decision_problem_id="dp_marketing_01",
        decision_card_id="card_marketing_01",
        base_payoffs=base_payoffs,
        scenarios_adjustments=scenarios_adjustments,
        evpi=18000.0,
        status_quo_choice="Status Quo",
        evppi={"reach_uncertainty": 12000.0},
        mcda_criteria_matrix=mcda_criteria_matrix,
        mcda_weights=mcda_weights,
    )

    assert session.session_id == "session_test_01"
    assert len(session.scenarios) == 3  # Base Case + 2 scenarios
    assert session.expected_losses["Option A"] == 0.0
    assert session.expected_losses["Status Quo"] == 40000.0

    # Validate schema
    assert validate_decision_studio_session(session.to_dict()) is True


def test_render_reports_and_dashboards() -> None:
    session = load_decision_studio_fixture()
    md_report = session.render_markdown_report()
    assert "# Decision Studio Executive Report" in md_report
    assert "Expected Opportunity Loss (Regret)" in md_report
    assert "Multi-Criteria Decision Analysis (MCDA) Scoring" in md_report

    html_dashboard = session.render_html_dashboard()
    assert "<!DOCTYPE html>" in html_dashboard
    assert "Scenario Robustness Analysis" in html_dashboard
    assert "Expected Opportunity Loss" in html_dashboard


def test_render_markdown_without_mcda() -> None:
    session = create_decision_studio_session(
        session_id="session_no_mcda",
        title="Simple Decision Session",
        decision_problem_id="dp_simple",
        decision_card_id="card_simple",
        base_payoffs={"A": 10.0, "B": 20.0},
        scenarios_adjustments={},
        evpi=5.0,
        status_quo_choice="A",
    )
    assert "mcda_evaluation" not in session.to_dict()
    md_report = session.render_markdown_report()
    assert "MCDA" not in md_report


def test_edge_cases_and_error_handling() -> None:
    # Empty payoffs in compute_expected_loss
    assert compute_expected_loss({}) == {}

    # Empty inputs in compute_mcda_scores
    assert compute_mcda_scores({}, {"w": 1.0}) == {}
    assert compute_mcda_scores({"A": {"c": 1.0}}, {}) == {}

    # Non-dictionary input to from_dict
    with pytest.raises(InputError, match="must be a dictionary"):
        DecisionStudioSession.from_dict("invalid")  # type: ignore[arg-type]

    # Empty base_payoffs in create_decision_studio_session
    with pytest.raises(InputError, match="cannot be empty"):
        create_decision_studio_session(
            session_id="s1",
            title="T",
            decision_problem_id="dp",
            decision_card_id="card",
            base_payoffs={},
            scenarios_adjustments={},
            evpi=0.0,
            status_quo_choice="A",
        )

    # Missing schema path
    non_existent_schema = Path("specs/decision-studio/missing.schema.json")
    with pytest.raises(InputError, match="not found"):
        validate_decision_studio_session({}, schema_path=non_existent_schema)

    # Missing fixture path
    non_existent_fixture = Path("specs/decision-studio/missing_fixture.json")
    with pytest.raises(InputError, match="not found"):
        load_decision_studio_fixture(fixture_path=non_existent_fixture)
