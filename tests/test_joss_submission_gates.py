"""Submission diagnostics recognize completed checks without bypassing gates."""

import json
from pathlib import Path

import pytest

from scripts.validate_joss import _validate_submission_gates

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    """Construct hypothetical test evidence, never a repository attestation."""
    readiness = json.loads((ROOT / "paper/joss-readiness-manifest.json").read_text())
    assurance = json.loads((ROOT / "paper/joss-editorial-assurance.json").read_text())
    readiness["manuscript_gates"].update(
        citation_and_source_audit="pass", official_pdf_build_and_visual_review="pass"
    )
    for gate in readiness["repository_gates"]:
        readiness["repository_gates"][gate] = "ready"
    readiness["submission_route"]["pyopensci_acceptance"] = "accepted"
    for gate in (
        "permanent_arxiv_identifier_and_announcement_before_joss_submission",
        "community_engagement_before_joss_submission",
    ):
        readiness["author_project_sequence"][gate] = "ready"
    for gate in (
        "authorship_funding_conflicts",
        "citation_source_check",
        "all_retained_ai_outputs_reviewed_modified_and_validated",
    ):
        assurance["human_review"][gate] = "confirmed"
    (tmp_path / "paper").mkdir()
    return readiness, assurance


def _validate(tmp_path: Path, readiness: dict, assurance: dict) -> list[str]:
    for name, value in (
        ("joss-readiness-manifest", readiness),
        ("joss-editorial-assurance", assurance),
    ):
        (tmp_path / f"paper/{name}.json").write_text(json.dumps(value))
    return _validate_submission_gates(tmp_path, "A factual manuscript.")


def test_completed_machine_checks_are_not_false_blockers(tmp_path: Path) -> None:
    assert _validate(tmp_path, *_fixture(tmp_path)) == []


@pytest.mark.parametrize("layer", ["manuscript_gates", "repository_gates"])
def test_missing_gate_layers_cannot_vacuously_pass(tmp_path: Path, layer: str) -> None:
    readiness, assurance = _fixture(tmp_path)
    readiness[layer] = {}
    assert _validate(tmp_path, readiness, assurance)


@pytest.mark.parametrize("status", [None, "pending", "not_started", True])
def test_selected_partner_route_requires_acceptance(
    tmp_path: Path, status: object
) -> None:
    readiness, assurance = _fixture(tmp_path)
    readiness["submission_route"]["pyopensci_acceptance"] = status
    assert any("partner" in item for item in _validate(tmp_path, readiness, assurance))


def test_missing_route_cannot_bypass_partner_eligibility(tmp_path: Path) -> None:
    readiness, assurance = _fixture(tmp_path)
    del readiness["submission_route"]
    assert any("route" in item for item in _validate(tmp_path, readiness, assurance))


def test_historical_human_assurance_is_not_current_confirmation(tmp_path: Path) -> None:
    readiness, assurance = _fixture(tmp_path)
    assurance["human_review"]["authorship_funding_conflicts"] = "pending"
    assert any(
        "authorship_funding_conflicts" in item
        for item in _validate(tmp_path, readiness, assurance)
    )
