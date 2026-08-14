"""Fail-closed tests for H8-D-B remediation and review intake."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
import pytest

import voiage.sampling_harm_review_preparation as review_preparation

SamplingHarmReviewPreparationError = (
    review_preparation.SamplingHarmReviewPreparationError
)
_load_and_validate_sampling_harm_remediation_intake = (
    review_preparation.load_and_validate_sampling_harm_remediation_intake
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/sampling-acquisition-harm/v1"
SCHEMAS = CONTRACT / "schemas"
HISTORICAL_VALIDATION_TIME = datetime(2026, 8, 3, tzinfo=UTC)


def load_and_validate_sampling_harm_remediation_intake(
    *, repository_root: Path
) -> dict[str, Any]:
    return _load_and_validate_sampling_harm_remediation_intake(
        repository_root=repository_root,
        now=HISTORICAL_VALIDATION_TIME,
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_contract(tmp_path: Path) -> Path:
    target = tmp_path / CONTRACT.relative_to(ROOT)
    shutil.copytree(CONTRACT, target)
    scientific_schemas = ROOT / "specs/frontier/governance/scientific-review/v1/schemas"
    scientific_target = (
        tmp_path / "specs/frontier/governance/scientific-review/v1/schemas"
    )
    shutil.copytree(scientific_schemas, scientific_target)
    return target


def test_h8db_artifacts_validate_against_strict_schemas() -> None:
    for name in (
        "adjacent-method-non-alias-delta",
        "remediation-register",
        "reviewer-intake-readiness",
        "source-review-intake-readiness",
        "governance-administrative-delta",
    ):
        schema = _json(SCHEMAS / f"{name}.schema.json")
        artifact_name = (
            "governance-administrative-delta-20260803.json"
            if name == "governance-administrative-delta"
            else f"{name}.json"
        )
        artifact = _json(CONTRACT / artifact_name)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(artifact)


def test_adjacent_method_delta_is_complete_and_non_executable() -> None:
    delta = _json(CONTRACT / "adjacent-method-non-alias-delta.json")
    issues = {record["issue"]: record for record in delta["issues"]}
    assert set(issues) == {570, 571, 595, 598}
    assert delta["historical_disposition_preserved"] is True
    assert delta["runtime_authority"] is False
    assert all(
        record["relationship"] == "not_sampling_acquisition_harm"
        for record in issues.values()
    )
    assert all(record["execution_reuse_allowed"] is False for record in issues.values())
    receipt = load_and_validate_sampling_harm_remediation_intake(repository_root=ROOT)
    assert receipt["effective_adjacent_issues"] == [570, 571, 595, 598]


def test_effective_disposition_rejects_duplicate_issue_mutation(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    mutated = _json(contract / "adjacent-method-non-alias-delta.json")
    mutated["issues"][1] = mutated["issues"][0]
    (contract / "adjacent-method-non-alias-delta.json").write_text(
        json.dumps(mutated), encoding="utf-8"
    )
    with pytest.raises(SamplingHarmReviewPreparationError):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_remediation_loader_wraps_invalid_json(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    (contract / "reviewer-intake-readiness.json").write_text(
        "not-json", encoding="utf-8"
    )
    with pytest.raises(
        SamplingHarmReviewPreparationError,
        match="cannot load remediation artifact reviewer-intake-readiness.json",
    ):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_remediation_loader_rejects_non_object_artifact(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    (contract / "reviewer-intake-readiness.json").write_text("[]", encoding="utf-8")
    with pytest.raises(
        SamplingHarmReviewPreparationError,
        match="remediation artifact and schema must be objects",
    ):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_effective_disposition_rejects_base_byte_drift(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    base = contract / "research-disposition.json"
    base.write_text(base.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        SamplingHarmReviewPreparationError,
        match="effective disposition base binding mismatch",
    ):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("issue", "effective adjacent-method projection is incomplete"),
        ("reuse", "effective adjacent-method projection permits execution reuse"),
    ],
)
def test_effective_disposition_semantics_defend_beyond_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    contract = _copy_contract(tmp_path)
    delta_path = contract / "adjacent-method-non-alias-delta.json"
    delta = _json(delta_path)
    if mutation == "issue":
        delta["issues"][1]["issue"] = 999
    else:
        delta["issues"][1]["execution_reuse_allowed"] = True
    delta_path.write_text(json.dumps(delta), encoding="utf-8")
    monkeypatch.setattr(review_preparation, "_validate_json", lambda *_a, **_k: None)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_remediation_register_rejects_authenticated_synthesis_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_contract(tmp_path)
    monkeypatch.setattr(
        review_preparation,
        "validate_sampling_harm_automated_challenge",
        lambda *_a, **_k: {"synthesis_sha256": "0" * 64},
    )
    with pytest.raises(
        SamplingHarmReviewPreparationError,
        match="remediation register synthesis digest mismatch",
    ):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("summary", "remediation register summary drift"),
        ("bindings", "remediation register evidence binding drift"),
    ],
)
def test_remediation_register_semantics_defend_beyond_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    contract = _copy_contract(tmp_path)
    register_path = contract / "remediation-register.json"
    register = _json(register_path)
    if mutation == "summary":
        register["summary"]["pending"] = 18
    else:
        register["bindings"]["candidate_commit"] = "0" * 40
    register_path.write_text(json.dumps(register), encoding="utf-8")
    monkeypatch.setattr(review_preparation, "_validate_json", lambda *_a, **_k: None)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_semantic_validator_rejects_register_severity_mutation(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    register = _json(contract / "remediation-register.json")
    for finding in register["findings"]:
        finding["severity"] = "Medium"
    (contract / "remediation-register.json").write_text(
        json.dumps(register), encoding="utf-8"
    )
    with pytest.raises(SamplingHarmReviewPreparationError):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_semantic_validator_rejects_tandem_synthesis_mutation(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    synthesis_path = (
        contract / "reviews/h8d-automated-challenge-synthesis-20260803.json"
    )
    synthesis = _json(synthesis_path)
    register_path = contract / "remediation-register.json"
    register = _json(register_path)
    synthesis["findings"][0]["normalized_severity"] = "Medium"
    synthesis["findings"][1]["normalized_severity"] = "High"
    register["findings"][0]["severity"] = "Medium"
    register["findings"][1]["severity"] = "High"
    synthesis_path.write_text(json.dumps(synthesis), encoding="utf-8")
    register_path.write_text(json.dumps(register), encoding="utf-8")
    with pytest.raises(SamplingHarmReviewPreparationError):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_semantic_validator_rejects_snapshot_mutation(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    (contract / "governance-snapshot.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SamplingHarmReviewPreparationError):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_governance_delta_rejects_future_observation(tmp_path: Path) -> None:
    contract = _copy_contract(tmp_path)
    governance_path = contract / "governance-administrative-delta-20260803.json"
    governance = _json(governance_path)
    governance["observed_at"] = "2999-01-01T00:00:00Z"
    governance["expires_at"] = "2999-01-02T00:00:00Z"
    governance_path.write_text(json.dumps(governance), encoding="utf-8")
    with pytest.raises(
        SamplingHarmReviewPreparationError,
        match="governance delta observation time is in the future",
    ):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


@pytest.mark.parametrize(
    ("observed_at", "expires_at"),
    [
        ("2026-08-02T21:19:47Z", "2026-08-02T21:19:47Z"),
        ("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
    ],
)
def test_governance_delta_rejects_invalid_or_expired_window(
    tmp_path: Path, observed_at: str, expires_at: str
) -> None:
    contract = _copy_contract(tmp_path)
    governance_path = contract / "governance-administrative-delta-20260803.json"
    governance = _json(governance_path)
    governance["observed_at"] = observed_at
    governance["expires_at"] = expires_at
    governance_path.write_text(json.dumps(governance), encoding="utf-8")
    with pytest.raises(
        SamplingHarmReviewPreparationError,
        match="governance delta is expired or has an invalid expiry",
    ):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("project_order", "governance delta Project readback is incomplete"),
        ("project_digest", "Project field digest mismatch for issue 850"),
    ],
)
def test_governance_project_semantics_defend_beyond_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    contract = _copy_contract(tmp_path)
    governance_path = contract / "governance-administrative-delta-20260803.json"
    governance = _json(governance_path)
    if mutation == "project_order":
        governance["project_28"]["items"][:2] = reversed(
            governance["project_28"]["items"][:2]
        )
    else:
        governance["project_28"]["items"][0]["owner_role"] = "mutated owner"
    governance_path.write_text(json.dumps(governance), encoding="utf-8")
    monkeypatch.setattr(review_preparation, "_validate_json", lambda *_a, **_k: None)
    with pytest.raises(SamplingHarmReviewPreparationError, match=message):
        load_and_validate_sampling_harm_remediation_intake(repository_root=tmp_path)


def test_intake_schemas_reject_duplicate_roles_fields_and_authority() -> None:
    reviewer = _json(CONTRACT / "reviewer-intake-readiness.json")
    reviewer["required_scientific_roles"][1] = reviewer["required_scientific_roles"][0]
    reviewer_validator = Draft202012Validator(
        _json(SCHEMAS / "reviewer-intake-readiness.schema.json")
    )
    with pytest.raises(ValidationError):
        reviewer_validator.validate(reviewer)

    source = _json(CONTRACT / "source-review-intake-readiness.json")
    source["required_record_fields"][1] = source["required_record_fields"][0]
    source_validator = Draft202012Validator(
        _json(SCHEMAS / "source-review-intake-readiness.schema.json")
    )
    with pytest.raises(ValidationError):
        source_validator.validate(source)

    governance = _json(CONTRACT / "governance-administrative-delta-20260803.json")
    governance["authority"] = {f"arbitrary_{index}": False for index in range(14)}
    governance_validator = Draft202012Validator(
        _json(SCHEMAS / "governance-administrative-delta.schema.json"),
        format_checker=FormatChecker(),
    )
    with pytest.raises(ValidationError):
        governance_validator.validate(governance)


def test_remediation_register_preserves_all_findings_as_pending() -> None:
    register = _json(CONTRACT / "remediation-register.json")
    findings = register["findings"]
    expected = {
        "H8D-ED-01",
        "H8D-ED-02",
        "H8D-ED-03",
        "H8D-ED-04",
        "H8D-EST-001",
        "H8D-EST-002",
        "H8D-EST-003",
        "H8D-EST-004",
        "H8D-API-GOV-01",
        "H8D-API-GOV-02",
        "H8D-API-GOV-03",
        "H8D-API-GOV-04",
        "H8D-GP-01",
        "H8D-GP-02",
        "H8D-DS-01",
        "H8D-DS-02",
        "H8D-DS-03",
        "H8D-DS-04",
        "H8D-DS-05",
    }
    assert {finding["finding_id"] for finding in findings} == expected
    assert register["summary"] == {
        "total": 19,
        "critical": 1,
        "high": 15,
        "medium": 3,
        "pending": 19,
        "resolved": 0,
    }
    assert register["disposition_paths"]["applies_to_every_finding"] is True
    assert register["disposition_paths"]["selection_authorized"] is False
    assert register["disposition_paths"]["ds03_blocking_in_both"] is True
    assert all(
        path["authorized"] is False
        for name, path in register["disposition_paths"].items()
        if name in {"reviewed_generic_exclusion", "future_narrow_candidate"}
    )
    assert all(finding["disposition_status"] == "pending" for finding in findings)
    assert all(
        finding["invalidation"] == "substantive_new_packet_required"
        for finding in findings
    )
    dissent = next(item for item in findings if item["finding_id"] == "H8D-EST-003")
    assert dissent["preserves_dissent"] is True
    assert register["authority"] == {
        "h8_d_satisfied": False,
        "h8_e_satisfied": False,
        "h8_f_satisfied": False,
        "scientific_acceptance": False,
        "runtime": False,
        "real_study": False,
        "publication": False,
        "release": False,
    }


def test_reviewer_intake_contains_no_identity_or_receipt() -> None:
    intake = _json(CONTRACT / "reviewer-intake-readiness.json")
    assert len(intake["required_scientific_roles"]) == 5
    assert all(
        role["assignment_status"] == "unassigned"
        for role in intake["required_scientific_roles"]
    )
    assert all(
        role["eligible"] is False for role in intake["required_scientific_roles"]
    )
    assert len(intake["human_confirmation_roles"]) == 2
    assert all(
        role["assignment_status"] == "unassigned"
        for role in intake["human_confirmation_roles"]
    )
    assert intake["precommission_prerequisites"] == {
        "source_review_ready": False,
        "replacement_packet_frozen": False,
    }
    assert intake["downstream_gates"] == {
        "all_findings_dispositioned": False,
        "signed_verdict_received": False,
    }
    assert intake["receipt_received"] is False
    assert intake["authority"] == {
        "independent_review": False,
        "human_confirmation": False,
        "scientific_disposition": False,
        "maintainer_decision": False,
    }
    serialized = json.dumps(intake).lower()
    assert "placeholder" not in serialized
    assert "example reviewer" not in serialized


def test_source_intake_is_blocked_until_bytes_rights_and_drift_exist() -> None:
    intake = _json(CONTRACT / "source-review-intake-readiness.json")
    assert set(intake["finding_ids"]) == {
        "H8D-ED-04",
        "H8D-EST-002",
        "H8D-API-GOV-03",
        "H8D-GP-02",
        "H8D-DS-05",
    }
    assert intake["status"] == "blocked_external_evidence"
    assert intake["retained_source_bytes"] is False
    assert intake["retention_required"] is False
    assert intake["rights_review_complete"] is False
    assert intake["applicability_review_complete"] is False
    assert intake["drift_review_complete"] is False
    assert intake["independent_retrieval_receipt"] is False
    assert intake["source_authority"] is False
    assert intake["required_record_fields"] == [
        "stable_source_id",
        "official_locator",
        "retrieved_at",
        "retrieval_status",
        "retention_rights_status",
        "redistribution_rights_status",
        "retained_bytes_sha256_if_permitted",
        "observation_sha256",
        "independent_retrieval_receipt",
        "drift_assessment",
        "jurisdiction_applicability",
    ]


def test_governance_delta_preserves_history_without_closing_h8() -> None:
    delta = _json(CONTRACT / "governance-administrative-delta-20260803.json")
    assert delta["historical_snapshot"]["preserved"] is True
    assert delta["historical_snapshot"]["current_status"] == "historical_not_current"
    assert delta["expires_at"] == "2026-08-09T21:19:47Z"
    assert delta["supersedes"] == []
    assert delta["hosted_change_classification"] == "bounded_metadata_only"
    assert delta["candidate_semantic_delta_requires_new_packet"] is True
    assert (
        delta["issues"]["850"]["body_sha256"]
        == "2a739678116a1c984998703f45ec508893166d200035d2c16601cc1b959582d8"
    )
    assert (
        delta["issues"]["853"]["body_sha256"]
        == "3f8293ab4eb46633031965ab5296ad5481a14913b63be5e5b821824c1fcb6756"
    )
    assert delta["issues"]["864"] == {
        "url": "https://github.com/edithatogo/voiage/issues/864",
        "state": "OPEN",
        "updated_at": "2026-08-02T21:03:55Z",
        "body_sha256": "38464e37df37b8ece642a0e2ddd533f47cee08d04d69b099d4a83892f94ae001",
        "node_id": "I_kwDOPF8PXM8AAAABLKxOOA",
        "native_parent": 853,
    }
    assert [item["issue"] for item in delta["project_28"]["items"]] == [
        850,
        853,
        864,
    ]
    assert (
        delta["pull_request_863"]["head_sha"]
        == "13456c7af74a7136b8aa71a74a01c47e64019480"
    )
    assert (
        delta["pull_request_863"]["merge_sha"]
        == "0c3f4314368693a07d0f9f6996e123a1d20ef208"
    )
    assert delta["pull_request_863"]["checks"] == {
        "success": 38,
        "governed_skip": 3,
        "neutral": 1,
        "failed": 0,
        "pending": 0,
        "unresolved_threads": 0,
    }
    assert all(value is False for value in delta["authority"].values())
