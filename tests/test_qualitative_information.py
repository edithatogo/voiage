"""Rule, dissent, audit and pathology tests for issue #558."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from voiage.contracts.qualitative_information import (
    qualitative_assessment_content_digest,
    qualitative_audit_event_digest,
)
from voiage.exceptions import InputError
from voiage.methods.qualitative_information import (
    qualitative_information_from_specification,
    render_qualitative_information_text,
)

ROOT = Path(__file__).parents[1]
INPUT = ROOT / "specs/frontier/qualitative-information/v1/fixtures/normative/input.json"
CASES = ROOT / "specs/frontier/qualitative-information/v1/fixtures/cases"


def _payload() -> dict[str, Any]:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def _rebind(payload: dict[str, Any]) -> None:
    assessment_digest = qualitative_assessment_content_digest(payload)
    previous_digest = None
    for event in payload["audit_history"]:
        event["assessment_content_digest"] = assessment_digest
        event["previous_content_digest"] = previous_digest
        event["content_digest"] = qualitative_audit_event_digest(event)
        previous_digest = event["content_digest"]


def _apply_recipe(payload: dict[str, Any], recipe: dict[str, Any]) -> None:
    for operation in recipe["operations"]:
        parts = operation["path"].strip("/").split("/")
        target: Any = payload
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        final = parts[-1]
        if operation["op"] == "add" and final == "-":
            target.append(operation["value"])
        elif isinstance(target, list):
            target[int(final)] = operation["value"]
        else:
            target[final] = operation["value"]


def test_no_cardinal_pseudo_score_is_emitted() -> None:
    result = qualitative_information_from_specification(_payload())
    serialized = json.dumps(result.to_contract_dict(), sort_keys=True)
    assert "score" not in serialized.lower()
    assert "currency" not in serialized.lower()
    assert result.numerical_estimand is False
    assert result.method_maturity == "experimental"


def test_question_order_does_not_change_groups_or_result_order() -> None:
    payload = _payload()
    reversed_payload = deepcopy(payload)
    reversed_payload["questions"] = list(reversed(payload["questions"]))
    _rebind(reversed_payload)
    original = qualitative_information_from_specification(payload)
    permuted = qualitative_information_from_specification(reversed_payload)
    assert original.question_results == permuted.question_results
    assert original.priority_groups == permuted.priority_groups


def test_dissent_is_preserved_and_never_silently_resolved() -> None:
    payload = _payload()
    judgements = payload["questions"][0]["judgements"]
    judgements[1]["priority_class"] = "routine"
    judgements[1]["recommendation_class"] = "do_not_pursue"
    payload["audit_history"][-1]["action"] = "review"
    _rebind(payload)
    result = qualitative_information_from_specification(payload)
    question = result.question_results[0]
    assert question.consensus_status == "dissent"
    assert question.resolved_priority_class is None
    assert question.priority_classes == ["urgent", "routine"]
    assert question.question_id in result.unresolved_question_ids


def test_missing_evidence_and_unverified_ai_keep_result_incomplete() -> None:
    payload = _payload()
    payload["questions"][0]["missing_fields"] = ["equity_ethics"]
    payload["questions"][0]["judgements"].append(
        {
            "reviewer_id": "ai-helper",
            "actor_type": "ai",
            "priority_class": "urgent",
            "recommendation_class": "pursue_now",
            "confidence": "high",
            "potential_impact": "unknown",
            "feasibility": "uncertain",
            "timeliness": "uncertain",
            "equity_ethics": "uncertain",
            "cost_burden": "unknown",
            "rationale": "Unverified machine suggestion.",
            "source_ids": ["source-redacted"],
            "verification_state": "unverified",
            "ai_provenance": {
                "provider": "synthetic-provider",
                "model_version": "synthetic-model-1",
                "input_reference": "fixture:redacted-input",
            },
        }
    )
    payload["audit_history"][-1]["action"] = "review"
    _rebind(payload)
    result = qualitative_information_from_specification(payload)
    assert result.workflow_status == "unverified"
    assert result.human_approval_status == "pending"
    assert result.question_results[0].consensus_status == "unverified"


def test_redacted_source_content_is_not_rendered() -> None:
    result = qualitative_information_from_specification(_payload())
    text = render_qualitative_information_text(result)
    assert "private source text" not in text
    assert "[REDACTED]" in text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["audit_history"][1].update(sequence=7),
        lambda payload: payload["audit_history"][0].update(
            actor={"actor_id": "ai", "actor_type": "ai"}, action="approve"
        ),
        lambda payload: payload["reviewers"].clear(),
        lambda payload: payload["questions"][0].update(judgements=[]),
    ],
)
def test_audit_and_human_boundary_pathologies_fail_closed(mutation) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(InputError):
        qualitative_information_from_specification(payload)


def test_text_rendering_is_deterministic_and_accessible() -> None:
    result = qualitative_information_from_specification(_payload())
    first = render_qualitative_information_text(result)
    second = render_qualitative_information_text(result)
    assert first == second
    assert "Qualitative information assessment" in first
    assert "Workflow status:" in first
    assert "Unresolved questions:" in first


def test_current_approval_is_required_and_stale_approval_is_unverified() -> None:
    payload = _payload()
    payload["assessment_version"] = 2
    payload["audit_history"][-1]["assessment_version"] = 2
    payload["audit_history"][-1]["action"] = "update"
    _rebind(payload)
    result = qualitative_information_from_specification(payload)
    assert result.workflow_status == "unverified"
    assert result.human_approval_status == "pending"


def test_approval_cannot_coexist_with_missing_or_dissenting_assessment() -> None:
    payload = _payload()
    payload["questions"][0]["missing_fields"] = ["equity_ethics"]
    _rebind(payload)
    with pytest.raises(InputError, match="complete verified consensus"):
        qualitative_information_from_specification(payload)


def test_system_approval_and_forged_human_override_fail_closed() -> None:
    system_payload = _payload()
    system_payload["audit_history"][-1]["actor"] = {
        "actor_id": "reviewer-a",
        "actor_type": "system",
    }
    _rebind(system_payload)
    with pytest.raises(InputError, match="accountable human"):
        qualitative_information_from_specification(system_payload)

    override_payload = _payload()
    judgement = deepcopy(override_payload["questions"][0]["judgements"][0])
    judgement.update(
        reviewer_id="ai-reviewed",
        actor_type="ai",
        verification_state="human_verified",
        ai_provenance={
            "provider": "synthetic",
            "model_version": "1",
            "input_reference": "fixture",
        },
        human_override={
            "reviewer_id": "reviewer-a",
            "audit_event_id": "forged-event",
        },
    )
    override_payload["questions"][0]["judgements"].append(judgement)
    _rebind(override_payload)
    with pytest.raises(InputError, match="accountable review"):
        qualitative_information_from_specification(override_payload)


def test_human_override_must_bind_current_version_and_snapshot() -> None:
    payload = _payload()
    payload["decision"]["accountable_reviewer_ids"].append("reviewer-b")
    judgement = deepcopy(payload["questions"][0]["judgements"][0])
    judgement.update(
        reviewer_id="ai-reviewed",
        actor_type="ai",
        verification_state="human_verified",
        ai_provenance={
            "provider": "synthetic",
            "model_version": "1",
            "input_reference": "fixture",
        },
        human_override={
            "reviewer_id": "reviewer-b",
            "audit_event_id": "event-review",
        },
    )
    payload["questions"][0]["judgements"].append(judgement)
    _rebind(payload)
    payload["audit_history"][1]["assessment_content_digest"] = "0" * 64
    payload["audit_history"][1]["content_digest"] = qualitative_audit_event_digest(
        payload["audit_history"][1]
    )
    for index in range(2, len(payload["audit_history"])):
        payload["audit_history"][index]["previous_content_digest"] = payload[
            "audit_history"
        ][index - 1]["content_digest"]
        payload["audit_history"][index]["content_digest"] = (
            qualitative_audit_event_digest(payload["audit_history"][index])
        )
    with pytest.raises(InputError, match="current assessment"):
        qualitative_information_from_specification(payload)


def test_redaction_applies_to_result_rendering_and_validation_errors() -> None:
    private_marker = "private-source-content-558"
    payload = _payload()
    question = payload["questions"][1]
    question["information_question"] = private_marker
    for judgement in question["judgements"]:
        judgement["rationale"] = private_marker
    _rebind(payload)
    result = qualitative_information_from_specification(payload)
    assert private_marker not in json.dumps(result.to_contract_dict())
    assert private_marker not in render_qualitative_information_text(result)

    invalid = _payload()
    invalid["questions"][0]["judgements"][0]["priority_class"] = private_marker
    with pytest.raises(InputError) as captured:
        qualitative_information_from_specification(invalid)
    assert private_marker not in str(captured.value)


def test_audit_digest_timestamp_and_version_tampering_fail_closed() -> None:
    digest_payload = _payload()
    digest_payload["audit_history"][1]["content_digest"] = "0" * 64
    with pytest.raises(InputError, match="content_digest"):
        qualitative_information_from_specification(digest_payload)

    timestamp_payload = _payload()
    timestamp_payload["audit_history"][1]["timestamp"] = "not-a-time"
    with pytest.raises(InputError, match="constraint: format"):
        qualitative_information_from_specification(timestamp_payload)

    version_payload = _payload()
    version_payload["audit_history"][0]["assessment_version"] = 2
    _rebind(version_payload)
    with pytest.raises(InputError, match="versions must be non-decreasing"):
        qualitative_information_from_specification(version_payload)


def test_equal_priorities_form_a_complete_deterministic_tie_group() -> None:
    payload = _payload()
    tied = deepcopy(payload["questions"][0])
    tied["question_id"] = "q-00-tied"
    payload["questions"].append(tied)
    _rebind(payload)
    result = qualitative_information_from_specification(payload)
    assert result.priority_groups[0] == {
        "priority_class": "urgent",
        "question_ids": ["q-00-tied", "q-01-trial"],
    }


@pytest.mark.parametrize("name", ["disagreement", "incomplete-ai"])
def test_committed_case_recipes_execute_to_their_expected_states(name: str) -> None:
    recipe = json.loads((CASES / f"{name}.json").read_text(encoding="utf-8"))
    payload = _payload()
    _apply_recipe(payload, recipe)
    _rebind(payload)
    result = qualitative_information_from_specification(payload).to_contract_dict()
    expected = recipe["expected"]
    assert result["workflow_status"] == expected["workflow_status"]
    question = next(
        item
        for item in result["question_results"]
        if item["question_id"] == expected["question_id"]
    )
    assert question["consensus_status"] == expected["consensus_status"]


def test_committed_adversarial_recipe_fails_closed() -> None:
    recipe = json.loads((CASES / "adversarial-audit.json").read_text(encoding="utf-8"))
    payload = _payload()
    _apply_recipe(payload, recipe)
    with pytest.raises(InputError, match=recipe["expected_error"]):
        qualitative_information_from_specification(payload)
