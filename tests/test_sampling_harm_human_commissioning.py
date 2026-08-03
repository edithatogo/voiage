"""Fail-closed tests for the H8-D human commissioning preflight."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from voiage.sampling_harm_human_commissioning import (
    SamplingHarmHumanCommissioningError,
    _load_object,
    load_and_validate_sampling_harm_human_commissioning,
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


def test_canonical_preflight_is_blocked_without_claiming_authority() -> None:
    result = load_and_validate_sampling_harm_human_commissioning(ROOT)
    assert result["commissioning_status"] == "blocked_prerequisites"
    assert result["candidate_decision"] == "decision_required"
    assert result["pending_findings"] == 19
    assert result["source_prerequisites"] == 5
    assert result["eligible_reviewers"] == 0
    assert result["ready"] is False


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
