"""Rule, dissent, audit and pathology tests for issue #558."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from voiage.exceptions import InputError
from voiage.methods.qualitative_information import (
    qualitative_information_from_specification,
    render_qualitative_information_text,
)

ROOT = Path(__file__).parents[1]
INPUT = ROOT / "specs/frontier/qualitative-information/v1/fixtures/normative/input.json"


def _payload() -> dict[str, object]:
    return json.loads(INPUT.read_text(encoding="utf-8"))


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
    original = qualitative_information_from_specification(payload)
    permuted = qualitative_information_from_specification(reversed_payload)
    assert original.question_results == permuted.question_results
    assert original.priority_groups == permuted.priority_groups


def test_dissent_is_preserved_and_never_silently_resolved() -> None:
    payload = _payload()
    judgements = payload["questions"][0]["judgements"]
    judgements[1]["priority_class"] = "routine"
    judgements[1]["recommendation_class"] = "do_not_pursue"
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
    result = qualitative_information_from_specification(payload)
    assert result.workflow_status == "incomplete"
    assert result.human_approval_status == "pending"
    assert result.question_results[0].consensus_status == "incomplete"


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
