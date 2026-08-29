"""Tests for the cross-venue submission-readiness contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_submission_readiness import (
    validate_contract,
    validate_pyopensci_evidence,
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
    assert summary["statuses"]["pkgcheck"] == "hosted_pending"


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
