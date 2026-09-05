"""Tests for the cross-venue submission-readiness contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_submission_readiness import (
    validate_contract,
    validate_pyopensci_evidence,
    validate_r_distribution_evidence,
    validate_ropensci_evidence,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs" / "submission-readiness" / "targets.json"


def test_submission_contract_covers_current_and_future_targets() -> None:
    summary = validate_contract(CONTRACT, ROOT)

    assert summary["target_count"] >= 20
    assert {
        "arxiv",
        "joss",
        "pyopensci",
        "ropensci",
        "r-journal",
        "journal-of-statistical-software",
        "numfocus",
        "pypi",
        "cran",
        "julia-general",
        "conda-forge",
        "software-heritage",
        "scicrunch-rrid",
    } <= set(summary["targets"])


def test_submission_contract_preserves_external_authority() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    for target in contract["targets"]:
        assert target["authority"]["prepare"] == "repository"
        assert target["authority"]["submit"] in {"human", "external-system"}
        assert target["authority"]["accept"] == "external"
        assert target["acceptance_evidence"]


def test_submission_contract_routes_every_target_to_a_github_execution_lane() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    planned_targets = {
        target for lane in contract["execution_lanes"] for target in lane["targets"]
    }

    assert set(contract["required_target_ids"]) <= planned_targets
    assert all(
        lane["issue_url"].startswith("https://github.com/edithatogo/voiage/issues/")
        for lane in contract["execution_lanes"]
    )


def test_submission_contract_requires_current_criteria_refresh_evidence() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    refresh = contract["criteria_refresh"]

    assert set(contract["required_target_ids"]) <= set(refresh["target_ids"])
    assert (ROOT / refresh["evidence"]).is_file()


def test_submission_contract_separates_criteria_and_status_refreshes() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    refresh = contract["evidence_refresh"]
    record = json.loads((ROOT / refresh["evidence"]).read_text(encoding="utf-8"))

    assert contract["criteria_refresh"]["reviewed_at"] == "2026-08-29"
    assert refresh["reviewed_at"] == "2026-09-03"
    assert set(contract["required_target_ids"]) <= set(refresh["target_ids"])
    assert record["state"] == refresh["state"]
    assert record["final_root_pr"] == refresh["final_root_pr"]


def test_submission_contract_routes_current_execution_issues() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    lanes = {lane["id"]: lane for lane in contract["execution_lanes"]}

    assert lanes["python-community-review"]["issue_url"].endswith("/1037")
    assert lanes["hpc-distribution-readiness"]["issue_url"].endswith("/1025")
    assert lanes["paper-and-author-boundaries"]["issue_url"].endswith("/296")
    assert lanes["distinct-publication-and-sustainability-assessment"][
        "issue_url"
    ].endswith("/1026")


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("current_issue_lanes", "paper_and_author_boundaries", 299, "current issue"),
        (
            "historical_completed_issue_lanes",
            "distinct_publication_and_sustainability_assessment",
            614,
            "historical issue",
        ),
        ("release", "github_and_pypi_published", False, "release evidence"),
        ("external_outcomes", "pyopensci_acceptance", "accepted", "remain pending"),
        (
            "repository_state",
            "easybuild_final_root_graph_merged",
            False,
            "repository state",
        ),
    ],
)
def test_submission_contract_rejects_rebound_evidence_refresh_fields(
    tmp_path: Path, section: str, field: str, value: object, message: str
) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = ROOT / contract["evidence_refresh"]["evidence"]
    record = json.loads(source.read_text(encoding="utf-8"))
    record[section][field] = value
    evidence = tmp_path / "refresh.json"
    evidence.write_text(json.dumps(record), encoding="utf-8")
    contract["evidence_refresh"]["evidence"] = "refresh.json"
    contract["evidence_refresh"]["evidence_sha256"] = hashlib.sha256(
        evidence.read_bytes()
    ).hexdigest()
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_contract(invalid, tmp_path)


def test_submission_contract_rejects_closed_execution_issue_route(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    lane = next(
        item
        for item in contract["execution_lanes"]
        if item["id"] == "paper-and-author-boundaries"
    )
    lane["issue_url"] = "https://github.com/edithatogo/voiage/issues/299"
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="current open issue"):
        validate_contract(invalid, ROOT)


def test_complete_refresh_requires_root_merge_in_repository_state(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = ROOT / contract["evidence_refresh"]["evidence"]
    record = json.loads(source.read_text(encoding="utf-8"))
    root_pr = {
        "number": 1087,
        "head_sha": "1" * 40,
        "merge_sha": "2" * 40,
        "reviewed_tree": "3" * 40,
        "merged_tree": "3" * 40,
        "tree_equal": True,
        "terminal_checks": 42,
    }
    record["state"] = "complete"
    record["final_root_pr"] = root_pr
    record["repository_state"]["easybuild_final_root_graph_merged"] = False
    evidence = tmp_path / "refresh.json"
    evidence.write_text(json.dumps(record), encoding="utf-8")
    contract["evidence_refresh"].update(
        {
            "state": "complete",
            "final_root_pr": root_pr,
            "evidence": "refresh.json",
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    )
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="repository state"):
        validate_contract(invalid, tmp_path)


def test_complete_refresh_rejects_rebound_root_merge_identity(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = ROOT / contract["evidence_refresh"]["evidence"]
    record = json.loads(source.read_text(encoding="utf-8"))
    record["final_root_pr"]["head_sha"] = "1" * 40
    contract["evidence_refresh"]["final_root_pr"]["head_sha"] = "1" * 40
    evidence = tmp_path / "refresh.json"
    evidence.write_text(json.dumps(record), encoding="utf-8")
    contract["evidence_refresh"].update(
        {
            "evidence": "refresh.json",
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    )
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="bound to PR #1087"):
        validate_contract(invalid, tmp_path)


def test_submission_contract_rejects_non_commit_refresh_base(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = ROOT / contract["evidence_refresh"]["evidence"]
    record = json.loads(source.read_text(encoding="utf-8"))
    record["base_main"] = "main"
    evidence = tmp_path / "refresh.json"
    evidence.write_text(json.dumps(record), encoding="utf-8")
    contract["evidence_refresh"].update(
        {
            "evidence": "refresh.json",
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    )
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="exact commit"):
        validate_contract(invalid, tmp_path)


def test_submission_contract_preserves_current_human_and_hpc_gates() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    targets = {target["id"]: target for target in contract["targets"]}
    requirements = {
        target_id: {item["id"]: item["status"] for item in target["requirements"]}
        for target_id, target in targets.items()
    }

    assert requirements["pyopensci"]["ai-affirmation"] == "satisfied"
    assert requirements["pyopensci"]["survey-and-human-written-submission"] == "pending"
    assert (
        requirements["spack"]["current-security-native-build-and-module-smoke"]
        == "pending"
    )
    assert requirements["spack"]["upstream-review-and-indexing"] == "external"
    assert requirements["easybuild"]["final-root-graph-validation"] == "satisfied"
    assert requirements["easybuild"]["native-foss-build-and-module-smoke"] == "pending"
    assert requirements["easybuild"]["upstream-review-and-indexing"] == "external"
    assert (
        requirements["yggdrasil"]["v2-2-candidate-update-and-submission"] == "pending"
    )


def test_submission_contract_rejects_false_complete_evidence_refresh(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["evidence_refresh"]["state"] = "awaiting_final_root_pr_merge"
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="record binding"):
        validate_contract(invalid, ROOT)


def test_pyopensci_matrix_records_commitment_but_defers_external_inquiry() -> None:
    summary = validate_pyopensci_evidence(
        ROOT / "specs" / "submission-readiness" / "pyopensci-evidence.json", ROOT
    )

    assert summary["criterion_count"] >= 10
    assert summary["deferred"] == ["external-inquiry"]


def test_ropensci_matrix_records_resolved_self_contained_installation() -> None:
    summary = validate_ropensci_evidence(
        ROOT / "specs" / "submission-readiness" / "ropensci-evidence.json", ROOT
    )

    assert summary["criterion_count"] >= 10
    assert summary["statuses"]["self-contained-installation"] == "satisfied"
    assert summary["statuses"]["pkgcheck"] == "satisfied"
    assert (
        summary["statuses"]["current-source-distribution-evidence"] == "hosted_pending"
    )
    assert summary["distribution"] == {"job_count": 16, "input_count": 4}


def test_ropensci_inquiry_is_staged_without_claiming_submission() -> None:
    draft = (
        ROOT / "docs" / "release" / "ropensci-presubmission-inquiry-draft.md"
    ).read_text(encoding="utf-8")

    assert "prepared locally; not posted or submitted" in draft
    assert "89.47% line coverage" in draft
    assert "Questions for an editor" in draft
    assert "has not been posted, submitted, or sent" in draft
    assert "full, unsuppressed `R CMD check --as-cran`" in draft
    assert "literal zero-NOTE" in draft


@pytest.mark.parametrize(
    "mutation",
    [
        "run",
        "job",
        "input",
        "current_input",
        "archive",
        "manual_digest",
        "notes",
        "packet",
        "external_action",
    ],
)
def test_r_distribution_evidence_rejects_rebound_claims(
    tmp_path: Path, mutation: str
) -> None:
    receipt_path = (
        ROOT / "specs/submission-readiness/r-distribution-evidence-20260902.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "run":
        receipt["hosted_run"]["run_id"] += 1
    elif mutation == "job":
        receipt["hosted_run"]["jobs"][0]["conclusion"] = "failure"
    elif mutation == "input":
        receipt["tested_input_equality"]["paths"][0]["tested_head_object_id"] = "0" * 40
    elif mutation == "current_input":
        receipt["tested_input_equality"]["paths"][0]["current_revision_object_id"] = (
            "0" * 40
        )
    elif mutation == "archive":
        receipt["source_archive"]["sha256"] = "0" * 64
    elif mutation == "manual_digest":
        receipt["source_archive"]["manual_check_receipt"]["sha256"] = "0" * 64
    elif mutation == "notes":
        receipt["check_outcome"]["notes"] = []
        receipt["check_outcome"]["strict_zero_note_criterion_met"] = True
    elif mutation == "packet":
        receipt["review_packet"]["state"] = "submitted"
    else:
        receipt["external_actions"]["ropensci_inquiry"] = True
    mutated = tmp_path / "r-distribution-evidence.json"
    mutated.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_r_distribution_evidence(mutated, ROOT, require_current_workflow=False)


def test_submission_contract_rejects_ready_target_with_unmet_gate(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["targets"][0]["readiness"] = "ready"
    contract["targets"][0]["requirements"][0]["status"] = "pending"
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be ready"):
        validate_contract(invalid, ROOT)


def test_submission_contract_rejects_missing_evidence_path(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["targets"][0]["requirements"][0]["evidence"] = ["docs/does-not-exist.md"]
    invalid = tmp_path / "targets.json"
    invalid.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence path"):
        validate_contract(invalid, ROOT)


def test_updated_r_workflow_cannot_inherit_historical_qualification() -> None:
    with pytest.raises(ValueError, match="tested inputs"):
        validate_r_distribution_evidence(
            ROOT / "specs/submission-readiness/r-distribution-evidence-20260902.json",
            ROOT,
        )


def test_pending_r_qualification_cannot_claim_current_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        content = original(path, *args, **kwargs)
        if path.name == "r-workflow-requalification-20260905.json":
            payload = json.loads(content)
            payload["current_workflow_qualified"] = True
            return json.dumps(payload)
        return content

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="requalification boundary"):
        validate_ropensci_evidence(
            ROOT / "specs/submission-readiness/ropensci-evidence.json", ROOT
        )
