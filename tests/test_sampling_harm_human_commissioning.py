"""Fail-closed tests for the H8-D human commissioning preflight."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from voiage.sampling_harm_human_commissioning import (
    SamplingHarmHumanCommissioningError,
    _load_object,
    load_and_validate_sampling_harm_candidate_decision,
    load_and_validate_sampling_harm_human_commissioning,
    validate_sampling_harm_candidate_decision,
    validate_sampling_harm_human_commissioning,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/sampling-acquisition-harm/v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _inputs() -> tuple[dict[str, Any], ...]:
    return (
        _json(CONTRACT / "human-commissioning-preflight-20260803.json"),
        _json(CONTRACT / "schemas/human-commissioning-preflight.schema.json"),
        _json(CONTRACT / "remediation-readiness-delta-20260803.json"),
        _json(CONTRACT / "reviewer-intake-readiness.json"),
        _json(CONTRACT / "source-review-intake-readiness.json"),
    )


def _decision_inputs() -> tuple[dict[str, Any], ...]:
    preflight, _preflight_schema, delta, reviewers, sources = _inputs()
    return (
        _json(CONTRACT / "candidate-context-decision-20260803.json"),
        _json(CONTRACT / "schemas/candidate-context-decision.schema.json"),
        preflight,
        delta,
        reviewers,
        sources,
    )


def test_canonical_preflight_is_blocked_without_claiming_authority() -> None:
    result = load_and_validate_sampling_harm_human_commissioning(ROOT)
    assert result["commissioning_status"] == "blocked_prerequisites"
    assert result["candidate_decision"] == "decision_required"
    assert result["pending_findings"] == 19
    assert result["source_prerequisites"] == 5
    assert result["eligible_reviewers"] == 0
    assert result["ready"] is False


def test_authenticated_option_one_decision_advances_only_candidate_gate() -> None:
    result = load_and_validate_sampling_harm_candidate_decision(ROOT)
    assert result == {
        "commissioning_status": "blocked_prerequisites",
        "candidate_decision": "selected",
        "selected_option": "reviewed_exclusion_generic_kernel",
        "remaining_blockers": 6,
        "pending_findings": 19,
        "source_review_ready": False,
        "eligible_reviewers": 0,
        "ready": False,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda decision: decision.__setitem__(
                "selected_option_id", "bounded_non_authorizing_candidate"
            ),
            "decision receipt invalid",
        ),
        (
            lambda decision: decision["authenticated_receipt"].__setitem__(
                "url", "https://github.com/edithatogo/voiage/issues/873"
            ),
            "decision receipt invalid",
        ),
        (
            lambda decision: decision["remaining_blocker_ids"].pop(),
            "decision receipt invalid",
        ),
        (
            lambda decision: decision["authority_boundary"].__setitem__(
                "reviewer_eligibility", True
            ),
            "decision receipt invalid",
        ),
    ],
)
def test_candidate_decision_mutations_fail_closed(mutation: Any, message: str) -> None:
    decision, schema, preflight, delta, reviewers, sources = map(
        deepcopy, _decision_inputs()
    )
    mutation(decision)
    with pytest.raises(SamplingHarmHumanCommissioningError, match=message):
        validate_sampling_harm_candidate_decision(
            decision,
            preflight,
            delta,
            reviewers,
            sources,
            schema=schema,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda decision: decision["remaining_blocker_ids"].pop(),
            "remaining blocker inventory mismatch",
        ),
        (
            lambda decision: decision["authority_boundary"].__setitem__(
                "reviewer_eligibility", True
            ),
            "decision receipt claims unavailable authority",
        ),
    ],
)
def test_candidate_decision_semantic_guards_reject_schema_bypass(
    mutation: Any, message: str
) -> None:
    decision, _schema, preflight, delta, reviewers, sources = map(
        deepcopy, _decision_inputs()
    )
    mutation(decision)
    with pytest.raises(SamplingHarmHumanCommissioningError, match=message):
        validate_sampling_harm_candidate_decision(
            decision,
            preflight,
            delta,
            reviewers,
            sources,
            schema={"type": "object"},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda _decision, preflight, *_: preflight.__setitem__(
                "candidate_context", None
            ),
            "candidate context is absent",
        ),
        (
            lambda _decision, preflight, *_: preflight["candidate_context"].__setitem__(
                "recommended_option_id", "bounded_non_authorizing_candidate"
            ),
            "does not supersede the exact preflight option",
        ),
        (
            lambda _decision, _preflight, delta, *_: delta["summary"].__setitem__(
                "pending", 18
            ),
            "nineteen pending findings are required",
        ),
        (
            lambda _decision, _preflight, _delta, reviewers, _sources: reviewers[
                "required_scientific_roles"
            ][0].__setitem__("eligible", True),
            "unexpectedly advances reviewer eligibility",
        ),
        (
            lambda _decision, _preflight, _delta, _reviewers, sources: (
                sources.__setitem__("source_authority", True)
            ),
            "unexpectedly advances source authority",
        ),
    ],
)
def test_candidate_decision_prerequisites_fail_closed(
    mutation: Any, message: str
) -> None:
    decision, _schema, preflight, delta, reviewers, sources = map(
        deepcopy, _decision_inputs()
    )
    mutation(decision, preflight, delta, reviewers, sources)
    with pytest.raises(SamplingHarmHumanCommissioningError, match=message):
        validate_sampling_harm_candidate_decision(
            decision,
            preflight,
            delta,
            reviewers,
            sources,
            schema={"type": "object"},
        )


def test_loader_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SamplingHarmHumanCommissioningError, match="contain an object"):
        _load_object(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda preflight, *_: preflight.__setitem__(
                "commissioning_status", "ready_to_commission"
            ),
            "preflight invalid",
        ),
        (
            lambda preflight, *_: preflight["candidate_context"].__setitem__(
                "decision_status", "selected"
            ),
            "preflight invalid",
        ),
        (
            lambda preflight, *_: preflight["authority_boundary"].__setitem__(
                "h8_d_satisfied", True
            ),
            "preflight invalid",
        ),
        (
            lambda preflight, *_: preflight["source_prerequisite_finding_ids"].pop(),
            "source prerequisite binding mismatch",
        ),
        (
            lambda preflight, *_: preflight["required_human_roles"].pop(),
            "reviewer role binding mismatch",
        ),
        (
            lambda preflight, *_: preflight["required_human_roles"][0].__setitem__(
                "stage", "h8_g"
            ),
            "reviewer stage binding mismatch",
        ),
        (
            lambda preflight, *_: preflight["input_bindings"][0].__setitem__(
                "sha256", "f" * 64
            ),
            "input digest mismatch",
        ),
        (
            lambda preflight, *_: preflight["input_bindings"].pop(),
            "input binding inventory mismatch",
        ),
        (
            lambda preflight, *_: preflight["input_bindings"][0].__setitem__(
                "path", "/invalid/remediation-readiness-delta-20260803.json"
            ),
            "input binding path is unsafe",
        ),
        (
            lambda preflight, *_: preflight["candidate_context"]["options"][
                0
            ].__setitem__(
                "option_id",
                preflight["candidate_context"]["options"][1]["option_id"],
            ),
            "candidate option inventory mismatch",
        ),
        (
            lambda preflight, *_: preflight["candidate_context"]["options"][
                0
            ].__setitem__("recommended", False),
            "exactly one candidate option must be recommended",
        ),
        (
            lambda _preflight, _schema, delta, *_: delta["summary"].__setitem__(
                "pending", 18
            ),
            "nineteen pending findings are required",
        ),
        (
            lambda _preflight, _schema, delta, *_: delta.__setitem__("groups", None),
            "finding readiness groups are absent",
        ),
        (
            lambda _preflight, _schema, delta, *_: delta["groups"][
                "source_review_prerequisite"
            ].pop(),
            "source prerequisite binding mismatch",
        ),
        (
            lambda _preflight, _schema, _delta, reviewers, *_: reviewers.__setitem__(
                "required_scientific_roles", None
            ),
            "reviewer intake roles are absent",
        ),
        (
            lambda _preflight, _schema, _delta, reviewers, *_: reviewers[
                "required_scientific_roles"
            ][0].__setitem__("eligible", True),
            "canonical intake unexpectedly claims an eligible reviewer",
        ),
        (
            lambda _preflight, _schema, _delta, reviewers, *_: reviewers.__setitem__(
                "receipt_received", True
            ),
            "canonical reviewer intake unexpectedly claims authority",
        ),
        (
            lambda _preflight, _schema, _delta, _reviewers, sources: (
                sources.__setitem__("source_authority", True)
            ),
            "canonical source intake unexpectedly claims authority",
        ),
        (
            lambda _preflight, _schema, _delta, _reviewers, sources: sources[
                "finding_ids"
            ].pop(),
            "source intake finding mismatch",
        ),
        (
            lambda preflight, *_: preflight["blocker_ids"].pop(),
            "commissioning blocker mismatch",
        ),
    ],
)
def test_preflight_mutations_fail_closed(mutation: Any, message: str) -> None:
    preflight, schema, delta, reviewers, sources = map(deepcopy, _inputs())
    mutation(preflight, schema, delta, reviewers, sources)
    with pytest.raises(SamplingHarmHumanCommissioningError, match=message):
        validate_sampling_harm_human_commissioning(
            preflight,
            delta,
            reviewers,
            sources,
            schema=schema,
            contract_root=CONTRACT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda preflight: preflight["preconditions"].__setitem__(
                "candidate_context_selected", True
            ),
            "claims unavailable authority",
        ),
        (
            lambda preflight: preflight["authority_boundary"].__setitem__(
                "reviewer_eligibility", True
            ),
            "claims unavailable authority",
        ),
        (
            lambda preflight: preflight["privacy_and_security"].__setitem__(
                "credentials_in_repository", True
            ),
            "permits sensitive repository data",
        ),
    ],
)
def test_semantic_guards_reject_schema_bypass(mutation: Any, message: str) -> None:
    preflight, _schema, delta, reviewers, sources = map(deepcopy, _inputs())
    mutation(preflight)
    with pytest.raises(SamplingHarmHumanCommissioningError, match=message):
        validate_sampling_harm_human_commissioning(
            preflight,
            delta,
            reviewers,
            sources,
            schema={"type": "object"},
            contract_root=CONTRACT,
        )


def test_preflight_has_one_recommended_option_and_privacy_boundary() -> None:
    preflight, schema, delta, reviewers, sources = _inputs()
    result = validate_sampling_harm_human_commissioning(
        preflight,
        delta,
        reviewers,
        sources,
        schema=schema,
        contract_root=CONTRACT,
    )
    assert result["recommended_option"] == "reviewed_exclusion_generic_kernel"
    assert preflight["privacy_and_security"]["credentials_in_repository"] is False
    assert (
        preflight["privacy_and_security"]["personal_contact_details_in_repository"]
        is False
    )
    assert preflight["privacy_and_security"]["raw_signatures_in_repository"] is False
