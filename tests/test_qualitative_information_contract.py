"""Portable contract tests for issue #558 qualitative VOI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from voiage.contracts.qualitative_information import (
    QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
    QUALITATIVE_INFORMATION_AUDIT_EVENT_SCHEMA_V1,
    QUALITATIVE_INFORMATION_RENDERING_SCHEMA_V1,
    QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1,
    qualitative_assessment_content_digest,
    qualitative_audit_event_digest,
    validate_qualitative_information_result_semantics,
    validate_qualitative_information_semantics,
)
from voiage.methods.qualitative_information import (
    qualitative_information_from_specification,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/qualitative-information/v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_normative_fixture_and_portable_schemas_validate() -> None:
    names = (
        "qualitative-information-assessment.schema.json",
        "qualitative-information-result.schema.json",
        "qualitative-information-audit-event.schema.json",
        "qualitative-information-rendering.schema.json",
    )
    schemas = [_json(CONTRACT / "schemas" / name) for name in names]
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    rendering = _json(CONTRACT / "fixtures/normative/rendering.json")
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(schemas[0]).validate(payload)
    Draft202012Validator(schemas[1]).validate(expected)
    Draft202012Validator(schemas[3]).validate(rendering)
    for event in payload["audit_history"]:
        Draft202012Validator(schemas[2]).validate(event)
    validate_qualitative_information_semantics(payload)
    assert schemas == [
        QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
        QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1,
        QUALITATIVE_INFORMATION_AUDIT_EVENT_SCHEMA_V1,
        QUALITATIVE_INFORMATION_RENDERING_SCHEMA_V1,
    ]
    assert (
        hashlib.sha256(rendering["content"].encode()).hexdigest()
        == rendering["content_sha256"]
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("decision", "alternatives"), ["same", "same"], "unique"),
        (("assessment_version",), 0, "constraint: minimum"),
        (("audit_history", 1, "previous_event_id"), "wrong", "chain"),
        (("questions", 0, "judgements", 0, "priority_class"), 3, "constraint: enum"),
    ],
)
def test_contract_and_semantics_fail_closed(
    path: tuple[str | int, ...], value: Any, message: str
) -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    target: Any = payload
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(Exception, match=message):
        validate_qualitative_information_semantics(payload)


def test_capabilities_do_not_overclaim_bindings_or_maturity() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "experimental"
    assert capabilities["numerical_estimand"] is False
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "executable"
    assert surfaces["rust"]["status"] == "unsupported"
    assert surfaces["r"]["status"] == "unsupported"
    assert surfaces["julia"]["status"] == "unsupported"
    assert surfaces["mojo"]["status"] == "external"


def test_contract_evidence_is_sha256_pinned() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    assert evidence["stable_claim_allowed"] is False
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_runtime_matches_normative_fixture() -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    assert (
        qualitative_information_from_specification(payload).to_contract_dict()
        == expected
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result.update(workflow_status="unverified"), "workflow status"),
        (
            lambda result: result["diagnostics"].update(score=99),
            "additionalProperties",
        ),
        (
            lambda result: result["priority_groups"][0].update(
                question_ids=["q-02-register"]
            ),
            "priority group",
        ),
        (
            lambda result: result.update(unresolved_question_ids=["q-01-trial"]),
            "unresolved question IDs",
        ),
        (
            lambda result: result["question_results"][0].update(
                resolved_priority_class="routine"
            ),
            "resolved priority",
        ),
        (
            lambda result: result["question_results"][0].update(
                verified_human_reviewers=[]
            ),
            "verified human",
        ),
        (
            lambda result: result["question_results"][1].update(
                information_question="private", rationale_by_reviewer={"r": "private"}
            ),
            "redacted result",
        ),
        (
            lambda result: result["question_results"][0].update(
                unverified_ai_contributors=["ai"]
            ),
            "unverified AI",
        ),
    ],
)
def test_result_contract_rejects_contradictions_and_cardinal_fields(
    mutation, message: str
) -> None:
    result = _json(CONTRACT / "fixtures/normative/expected.json")
    mutation(result)
    with pytest.raises(ValueError, match=message):
        validate_qualitative_information_result_semantics(result)


def test_result_rejects_approval_for_an_incomplete_workflow() -> None:
    result = _json(CONTRACT / "fixtures/normative/expected.json")
    question = result["question_results"][0]
    question["consensus_status"] = "incomplete"
    question["resolved_priority_class"] = None
    question["resolved_recommendation_class"] = None
    result["priority_groups"][0]["question_ids"] = []
    result["unresolved_question_ids"] = [question["question_id"]]
    result["workflow_status"] = "incomplete"
    with pytest.raises(ValueError, match="approval"):
        validate_qualitative_information_result_semantics(result)


def _apply_semantic_pathology(payload: dict[str, Any], case: str) -> None:
    reviewers = payload["reviewers"]
    sources = payload["sources"]
    questions = payload["questions"]
    audit = payload["audit_history"]
    if case == "duplicate-reviewer":
        reviewers[1]["reviewer_id"] = reviewers[0]["reviewer_id"]
    elif case == "duplicate-source":
        sources[1]["source_id"] = sources[0]["source_id"]
    elif case == "duplicate-question":
        questions[1]["question_id"] = questions[0]["question_id"]
    elif case == "duplicate-event":
        audit[1]["event_id"] = audit[0]["event_id"]
    elif case == "unknown-accountable":
        payload["decision"]["accountable_reviewer_ids"] = ["reviewer-unknown"]
    elif case == "redacted-citation":
        sources[1]["citation"] = "private text"
    elif case == "unavailable-citation":
        sources[0]["access_status"] = "unavailable"
    elif case == "duplicate-judgement":
        questions[0]["judgements"][1]["reviewer_id"] = "reviewer-a"
    elif case == "unknown-source":
        questions[0]["judgements"][0]["source_ids"] = ["source-unknown"]
    elif case == "unknown-human":
        questions[0]["judgements"][0]["reviewer_id"] = "reviewer-unknown"
    elif case == "unverified-human":
        questions[0]["judgements"][0]["verification_state"] = "unverified"
    elif case == "human-ai-provenance":
        questions[0]["judgements"][0]["ai_provenance"] = {
            "provider": "none",
            "model_version": "none",
            "input_reference": "none",
        }
    elif case in {"ai-no-provenance", "ai-self-verified", "no-human"}:
        targets = (
            questions[0]["judgements"]
            if case == "no-human"
            else [questions[0]["judgements"][0]]
        )
        for index, judgement in enumerate(targets):
            judgement["reviewer_id"] = f"ai-{index}"
            judgement["actor_type"] = "ai"
            judgement["verification_state"] = (
                "verified" if case == "ai-self-verified" else "unverified"
            )
            if case != "ai-no-provenance":
                judgement["ai_provenance"] = {
                    "provider": "synthetic",
                    "model_version": "1",
                    "input_reference": "fixture",
                }
    elif case == "backwards-time":
        audit[1]["timestamp"] = "2025-01-01T00:00:00Z"
    elif case == "unknown-human-actor":
        audit[1]["actor"]["actor_id"] = "reviewer-unknown"
    elif case == "ai-event-no-provenance":
        audit[1]["actor"] = {"actor_id": "ai", "actor_type": "ai"}
    elif case == "non-accountable-approval":
        audit[2]["actor"]["actor_id"] = "reviewer-b"
    elif case == "version-mismatch":
        audit[-1]["assessment_version"] = 2
    else:  # pragma: no cover - guarded by the parametrized case list
        raise AssertionError(case)

    if case in {
        "unknown-human-actor",
        "ai-event-no-provenance",
        "non-accountable-approval",
        "version-mismatch",
    }:
        assessment_digest = qualitative_assessment_content_digest(payload)
        previous_digest = None
        for event in audit:
            event["assessment_content_digest"] = assessment_digest
            event["previous_content_digest"] = previous_digest
            event["content_digest"] = qualitative_audit_event_digest(event)
            previous_digest = event["content_digest"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate-reviewer", "reviewer IDs must be unique"),
        ("duplicate-source", "source IDs must be unique"),
        ("duplicate-question", "question IDs must be unique"),
        ("duplicate-event", "audit event IDs must be unique"),
        ("unknown-accountable", "accountable reviewer IDs"),
        ("redacted-citation", "redacted source citation"),
        ("unavailable-citation", "unavailable source citation"),
        ("duplicate-judgement", "judgement reviewer IDs"),
        ("unknown-source", "source IDs must identify"),
        ("unknown-human", "declared reviewer"),
        ("unverified-human", "verification state"),
        ("human-ai-provenance", "must not declare AI"),
        ("ai-no-provenance", "requires provider"),
        ("ai-self-verified", "cannot self-declare"),
        ("no-human", "requires a human judgement"),
        ("backwards-time", "non-decreasing"),
        ("unknown-human-actor", "human audit actor"),
        ("ai-event-no-provenance", "AI audit event requires"),
        ("non-accountable-approval", "accountable human reviewer"),
        ("version-mismatch", "must match assessment_version"),
    ],
)
def test_cross_field_semantic_pathologies_fail_closed(case: str, message: str) -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    _apply_semantic_pathology(payload, case)
    with pytest.raises(ValueError, match=message):
        validate_qualitative_information_semantics(payload)
