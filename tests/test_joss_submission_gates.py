"""Submission diagnostics recognize completed checks without bypassing gates."""

from hashlib import sha256
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


def _authorize_journal_first(tmp_path: Path, readiness: dict) -> Path:
    """Bind synthetic authorization without claiming a real venue outcome."""
    relative = (
        "conductor/tracks/v2_2_release_and_venue_submissions_20260830/"
        "maintainer-venue-decision-20260831.json"
    )
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "voiage.maintainer-venue-decision.v1",
                "source": "current_user_message",
                "candidate_version": "2.2.0",
                "journal_first_authorized": True,
                "user_statement": "Synthetic fixture authorizes journal first.",
            }
        )
    )
    sequence = readiness["author_project_sequence"]
    sequence["permanent_arxiv_identifier_and_announcement_before_joss_submission"] = (
        "not_required_by_maintainer"
    )
    sequence["maintainer_decision"] = {
        "path": relative,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }
    return path


def test_bound_journal_first_decision_retires_only_arxiv_gate(tmp_path: Path) -> None:
    readiness, assurance = _fixture(tmp_path)
    _authorize_journal_first(tmp_path, readiness)
    assert _validate(tmp_path, readiness, assurance) == []
    readiness["author_project_sequence"][
        "community_engagement_before_joss_submission"
    ] = "not_required_by_maintainer"
    assert any(
        "community_engagement" in finding
        for finding in _validate(tmp_path, readiness, assurance)
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "hash",
        "false",
        "version",
        "schema",
        "source",
        "statement",
        "malformed",
        "path",
    ],
)
def test_journal_first_requires_valid_bound_authority(
    tmp_path: Path, failure: str
) -> None:
    readiness, assurance = _fixture(tmp_path)
    path = _authorize_journal_first(tmp_path, readiness)
    binding = readiness["author_project_sequence"]["maintainer_decision"]
    receipt = json.loads(path.read_text())
    if failure == "missing":
        path.unlink()
    elif failure == "hash":
        binding["sha256"] = "0" * 64
    elif failure == "path":
        binding["path"] = "../../outside.json"
    else:
        key, value = {
            "false": ("journal_first_authorized", "true"),
            "version": ("candidate_version", "2.0.0"),
            "schema": ("schema_version", "unknown"),
            "source": ("source", "agent_inference"),
            "statement": ("user_statement", ""),
            "malformed": ("unused", None),
        }[failure]
        receipt[key] = value
        path.write_text("{" if failure == "malformed" else json.dumps(receipt))
        binding["sha256"] = sha256(path.read_bytes()).hexdigest()
    assert any(
        "permanent_arxiv" in finding
        for finding in _validate(tmp_path, readiness, assurance)
    )


@pytest.mark.parametrize("status", [None, "pending", "waived", True, {}])
def test_unknown_arxiv_status_fails_closed(tmp_path: Path, status: object) -> None:
    readiness, assurance = _fixture(tmp_path)
    _authorize_journal_first(tmp_path, readiness)
    readiness["author_project_sequence"][
        "permanent_arxiv_identifier_and_announcement_before_joss_submission"
    ] = status
    assert any(
        "permanent_arxiv" in finding
        for finding in _validate(tmp_path, readiness, assurance)
    )


def test_journal_first_preserves_partner_and_human_gates(tmp_path: Path) -> None:
    readiness, assurance = _fixture(tmp_path)
    _authorize_journal_first(tmp_path, readiness)
    readiness["submission_route"]["pyopensci_acceptance"] = "pending"
    assurance["human_review"]["citation_source_check"] = "pending"
    findings = _validate(tmp_path, readiness, assurance)
    assert any("partner" in finding for finding in findings)
    assert any("citation_source_check" in finding for finding in findings)
