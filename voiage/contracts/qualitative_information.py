"""Installed portable contract for experimental qualitative information assessment."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import TYPE_CHECKING, Any, Final, cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping


PRIORITY_CLASSES: Final = ("urgent", "high", "routine", "defer")
RECOMMENDATION_CLASSES: Final = (
    "pursue_now",
    "pursue_if_feasible",
    "monitor",
    "do_not_pursue",
)

_JUDGEMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": [
        "reviewer_id",
        "actor_type",
        "potential_impact",
        "feasibility",
        "timeliness",
        "equity_ethics",
        "cost_burden",
        "priority_class",
        "recommendation_class",
        "confidence",
        "rationale",
        "source_ids",
        "verification_state",
    ],
    "properties": {
        "reviewer_id": {"type": "string", "minLength": 1},
        "actor_type": {"enum": ["human", "ai"]},
        "potential_impact": {"enum": ["major", "moderate", "limited", "unknown"]},
        "feasibility": {"enum": ["feasible", "uncertain", "not_feasible"]},
        "timeliness": {"enum": ["timely", "uncertain", "too_late"]},
        "equity_ethics": {"enum": ["acceptable", "uncertain", "concern"]},
        "cost_burden": {"enum": ["low", "moderate", "high", "unknown"]},
        "priority_class": {"enum": list(PRIORITY_CLASSES)},
        "recommendation_class": {"enum": list(RECOMMENDATION_CLASSES)},
        "confidence": {"enum": ["high", "moderate", "low", "unknown"]},
        "rationale": {"type": "string", "minLength": 1},
        "source_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "verification_state": {"enum": ["verified", "unverified", "human_verified"]},
        "ai_provenance": {
            "type": "object",
            "required": ["provider", "model_version", "input_reference"],
            "properties": {
                "provider": {"type": "string", "minLength": 1},
                "model_version": {"type": "string", "minLength": 1},
                "input_reference": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "human_override": {
            "type": "object",
            "required": ["reviewer_id", "audit_event_id"],
            "properties": {
                "reviewer_id": {"type": "string", "minLength": 1},
                "audit_event_id": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
    },
    "allOf": [
        {
            "if": {"properties": {"verification_state": {"const": "human_verified"}}},
            "then": {"required": ["human_override"]},
            "else": {"not": {"required": ["human_override"]}},
        }
    ],
    "additionalProperties": False,
}

_AUDIT_EVENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": [
        "event_id",
        "sequence",
        "previous_event_id",
        "timestamp",
        "assessment_version",
        "actor",
        "action",
        "content_digest",
        "assessment_content_digest",
        "previous_content_digest",
        "redacted",
    ],
    "properties": {
        "event_id": {"type": "string", "minLength": 1},
        "sequence": {"type": "integer", "minimum": 1},
        "previous_event_id": {
            "oneOf": [{"type": "string", "minLength": 1}, {"type": "null"}]
        },
        "timestamp": {"type": "string", "format": "date-time"},
        "assessment_version": {"type": "integer", "minimum": 1},
        "actor": {
            "type": "object",
            "required": ["actor_id", "actor_type"],
            "properties": {
                "actor_id": {"type": "string", "minLength": 1},
                "actor_type": {"enum": ["human", "ai", "system"]},
            },
            "additionalProperties": False,
        },
        "action": {
            "enum": ["create", "update", "review", "approve", "redact", "ai_assist"]
        },
        "content_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "assessment_content_digest": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "previous_content_digest": {
            "oneOf": [
                {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                {"type": "null"},
            ]
        },
        "redacted": {"type": "boolean"},
        "ai_provenance": _JUDGEMENT_SCHEMA["properties"]["ai_provenance"],  # type: ignore[index]
        "override_event_id": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}

QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1: Final[dict[str, object]] = cast(
    "dict[str, object]",
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://voiage.dev/schemas/frontier/qualitative-information-assessment.v1.json",
        "title": "QualitativeInformationAssessmentV1Experimental",
        "type": "object",
        "required": [
            "schema_version",
            "assessment_id",
            "assessment_version",
            "method_maturity",
            "numerical_estimand",
            "decision",
            "reviewers",
            "sources",
            "questions",
            "audit_history",
            "policy",
            "provenance",
        ],
        "properties": {
            "schema_version": {"const": "1.0.0"},
            "assessment_id": {"type": "string", "minLength": 1},
            "assessment_version": {"type": "integer", "minimum": 1},
            "method_maturity": {"const": "experimental"},
            "numerical_estimand": {"const": False},
            "decision": {
                "type": "object",
                "required": [
                    "decision_id",
                    "title",
                    "context",
                    "alternatives",
                    "accountable_reviewer_ids",
                ],
                "properties": {
                    "decision_id": {"type": "string", "minLength": 1},
                    "title": {"type": "string", "minLength": 1},
                    "context": {"type": "string", "minLength": 1},
                    "alternatives": {
                        "type": "array",
                        "minItems": 2,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "accountable_reviewer_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "additionalProperties": False,
            },
            "reviewers": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["reviewer_id", "name", "role"],
                    "properties": {
                        "reviewer_id": {"type": "string", "minLength": 1},
                        "name": {"type": "string", "minLength": 1},
                        "role": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
            },
            "sources": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "source_id",
                        "citation",
                        "access_status",
                        "provenance",
                    ],
                    "properties": {
                        "source_id": {"type": "string", "minLength": 1},
                        "citation": {"type": "string", "minLength": 1},
                        "access_status": {
                            "enum": ["accessible", "redacted", "unavailable"]
                        },
                        "provenance": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
            },
            "questions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "question_id",
                        "information_question",
                        "uncertainty_or_evidence_gap",
                        "information_action",
                        "missing_fields",
                        "redaction_status",
                        "judgements",
                    ],
                    "properties": {
                        "question_id": {"type": "string", "minLength": 1},
                        "information_question": {"type": "string", "minLength": 1},
                        "uncertainty_or_evidence_gap": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "information_action": {"type": "string", "minLength": 1},
                        "missing_fields": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "redaction_status": {"enum": ["none", "partial", "full"]},
                        "judgements": {
                            "type": "array",
                            "minItems": 1,
                            "items": _JUDGEMENT_SCHEMA,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "audit_history": {
                "type": "array",
                "minItems": 1,
                "items": _AUDIT_EVENT_SCHEMA,
            },
            "policy": {
                "type": "object",
                "required": [
                    "priority_order",
                    "recommendation_order",
                    "conflict_policy",
                    "missingness_policy",
                    "ai_policy",
                    "tie_policy",
                ],
                "properties": {
                    "priority_order": {
                        "const": list(PRIORITY_CLASSES),
                    },
                    "recommendation_order": {
                        "const": list(RECOMMENDATION_CLASSES),
                    },
                    "conflict_policy": {"const": "preserve_dissent_no_resolution"},
                    "missingness_policy": {"const": "mark_incomplete"},
                    "ai_policy": {"const": "human_verification_required"},
                    "tie_policy": {"const": "complete_sets_declared_order"},
                },
                "additionalProperties": False,
            },
            "provenance": {
                "type": "object",
                "required": [
                    "fixture_id",
                    "contract_reference",
                    "source_snapshot",
                    "redaction_policy_reference",
                ],
                "properties": {
                    "fixture_id": {"type": "string", "minLength": 1},
                    "contract_reference": {"type": "string", "minLength": 1},
                    "source_snapshot": {"type": "string", "minLength": 1},
                    "redaction_policy_reference": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    },
)

QUALITATIVE_INFORMATION_AUDIT_EVENT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/qualitative-information-audit-event.v1.json",
    "title": "QualitativeInformationAuditEventV1Experimental",
    **_AUDIT_EVENT_SCHEMA,
}

QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/qualitative-information-result.v1.json",
    "title": "QualitativeInformationResultV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "assessment_id",
        "assessment_version",
        "analysis_type",
        "method_maturity",
        "numerical_estimand",
        "workflow_status",
        "human_approval_status",
        "priority_groups",
        "unresolved_question_ids",
        "question_results",
        "audit_history_digest",
        "audit_event_count",
        "source_summary",
        "provenance",
        "diagnostics",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "assessment_id": {"type": "string", "minLength": 1},
        "assessment_version": {"type": "integer", "minimum": 1},
        "analysis_type": {"const": "qualitative_information_assessment"},
        "method_maturity": {"const": "experimental"},
        "numerical_estimand": {"const": False},
        "workflow_status": {"enum": ["complete", "incomplete", "unverified"]},
        "human_approval_status": {"enum": ["approved", "pending"]},
        "priority_groups": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": ["priority_class", "question_ids"],
                "properties": {
                    "priority_class": {"enum": list(PRIORITY_CLASSES)},
                    "question_ids": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "additionalProperties": False,
            },
        },
        "unresolved_question_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "question_results": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "question_id",
                    "information_question",
                    "priority_classes",
                    "recommendation_classes",
                    "resolved_priority_class",
                    "resolved_recommendation_class",
                    "consensus_status",
                    "verified_human_reviewers",
                    "unverified_ai_contributors",
                    "missing_fields",
                    "redaction_status",
                    "rationale_by_reviewer",
                ],
                "properties": {
                    "question_id": {"type": "string", "minLength": 1},
                    "information_question": {"type": "string", "minLength": 1},
                    "priority_classes": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"enum": list(PRIORITY_CLASSES)},
                    },
                    "recommendation_classes": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"enum": list(RECOMMENDATION_CLASSES)},
                    },
                    "resolved_priority_class": {
                        "oneOf": [{"enum": list(PRIORITY_CLASSES)}, {"type": "null"}]
                    },
                    "resolved_recommendation_class": {
                        "oneOf": [
                            {"enum": list(RECOMMENDATION_CLASSES)},
                            {"type": "null"},
                        ]
                    },
                    "consensus_status": {
                        "enum": ["consensus", "dissent", "incomplete", "unverified"]
                    },
                    "verified_human_reviewers": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "unverified_ai_contributors": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "missing_fields": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "redaction_status": {"enum": ["none", "partial", "full"]},
                    "rationale_by_reviewer": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "minLength": 1},
                    },
                },
                "additionalProperties": False,
            },
        },
        "audit_history_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "audit_event_count": {"type": "integer", "minimum": 1},
        "source_summary": {
            "type": "object",
            "required": ["accessible", "redacted", "unavailable"],
            "properties": {
                "accessible": {"type": "integer", "minimum": 0},
                "redacted": {"type": "integer", "minimum": 0},
                "unavailable": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": ["fixture_id", "contract_reference"],
            "properties": {
                "fixture_id": {"type": "string", "minLength": 1},
                "contract_reference": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "diagnostics": {
            "type": "object",
            "required": [
                "aggregation_policy",
                "classification_policy",
                "conflict_policy",
                "missingness_policy",
                "ai_policy",
                "tie_policy",
                "privacy_boundary",
                "quantitative_voi_claim",
            ],
            "properties": {
                "aggregation_policy": {"const": "none_non_cardinal"},
                "classification_policy": {
                    "const": "accountable_human_declared_no_derivation"
                },
                "conflict_policy": {"const": "preserve_dissent_no_resolution"},
                "missingness_policy": {"const": "mark_incomplete"},
                "ai_policy": {"const": "human_verification_required"},
                "tie_policy": {"const": "complete_sets_declared_order"},
                "privacy_boundary": {"const": "redacted_content_not_returned"},
                "quantitative_voi_claim": {"const": False},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

QUALITATIVE_INFORMATION_RENDERING_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/qualitative-information-rendering.v1.json",
    "title": "QualitativeInformationRenderingV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "assessment_id",
        "media_type",
        "accessibility",
        "content_sha256",
        "content",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "assessment_id": {"type": "string", "minLength": 1},
        "media_type": {"const": "text/plain"},
        "accessibility": {
            "type": "object",
            "required": [
                "wcag_reference",
                "headings_present",
                "no_colour_only_semantics",
                "redactions_preserved",
            ],
            "properties": {
                "wcag_reference": {"const": "WCAG-2.2"},
                "headings_present": {"const": True},
                "no_colour_only_semantics": {"const": True},
                "redactions_preserved": {"const": True},
            },
            "additionalProperties": False,
        },
        "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "content": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}


def _ids(records: list[Mapping[str, Any]], key: str, label: str) -> list[str]:
    values = [str(record[key]) for record in records]
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qualitative_assessment_content_digest(specification: Mapping[str, object]) -> str:
    """Digest the current assessment content, excluding its append-only history."""
    content = dict(specification)
    content.pop("audit_history", None)
    return _canonical_digest(content)


def qualitative_audit_event_digest(event: Mapping[str, object]) -> str:
    """Digest one audit event, excluding the digest field itself."""
    content = dict(event)
    content.pop("content_digest", None)
    return _canonical_digest(content)


def validate_qualitative_information_semantics(
    specification: Mapping[str, object],
) -> None:
    """Validate cross-field, audit-chain, redaction and human/AI invariants."""
    try:
        Draft202012Validator(
            QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
            format_checker=FormatChecker(),
        ).validate(specification)
    except ValidationError as error:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise ValueError(
            f"invalid specification at {path} (constraint: {error.validator})"
        ) from error

    payload = cast("Mapping[str, Any]", specification)
    reviewers = cast("list[Mapping[str, Any]]", payload["reviewers"])
    sources = cast("list[Mapping[str, Any]]", payload["sources"])
    questions = cast("list[Mapping[str, Any]]", payload["questions"])
    audit = cast("list[Mapping[str, Any]]", payload["audit_history"])
    reviewer_ids = set(_ids(reviewers, "reviewer_id", "reviewer IDs"))
    source_ids = set(_ids(sources, "source_id", "source IDs"))
    _ids(questions, "question_id", "question IDs")
    event_ids = _ids(audit, "event_id", "audit event IDs")

    accountable = set(payload["decision"]["accountable_reviewer_ids"])
    if not accountable <= reviewer_ids:
        raise ValueError("accountable reviewer IDs must identify declared reviewers")

    for source in sources:
        if source["access_status"] == "redacted" and source["citation"] != "[REDACTED]":
            raise ValueError("redacted source citation must be [REDACTED]")
        if (
            source["access_status"] == "unavailable"
            and source["citation"] != "[UNAVAILABLE]"
        ):
            raise ValueError("unavailable source citation must be [UNAVAILABLE]")

    for question in questions:
        human_count = 0
        judgement_ids: set[str] = set()
        for judgement in question["judgements"]:
            actor_id = judgement["reviewer_id"]
            if actor_id in judgement_ids:
                raise ValueError("question judgement reviewer IDs must be unique")
            judgement_ids.add(actor_id)
            if not set(judgement["source_ids"]) <= source_ids:
                raise ValueError("judgement source IDs must identify declared sources")
            if judgement["actor_type"] == "human":
                human_count += 1
                if actor_id not in reviewer_ids:
                    raise ValueError(
                        "human judgement must identify a declared reviewer"
                    )
                if judgement["verification_state"] != "verified":
                    raise ValueError(
                        "human judgement verification state must be verified"
                    )
                if "ai_provenance" in judgement:
                    raise ValueError("human judgement must not declare AI provenance")
            else:
                if "ai_provenance" not in judgement:
                    raise ValueError(
                        "AI judgement requires provider and model provenance"
                    )
                if judgement["verification_state"] == "verified":
                    raise ValueError("AI judgement cannot self-declare verified")
        if human_count == 0:
            raise ValueError("each question requires a human judgement")

    previous: str | None = None
    previous_digest: str | None = None
    previous_timestamp: datetime | None = None
    previous_version = 0
    current_content_digest = qualitative_assessment_content_digest(specification)
    for index, event in enumerate(audit, 1):
        if event["sequence"] != index:
            raise ValueError("audit sequence must be contiguous and positive")
        if event["previous_event_id"] != previous:
            raise ValueError("audit previous_event_id chain is broken")
        if event["previous_content_digest"] != previous_digest:
            raise ValueError("audit previous_content_digest chain is broken")
        timestamp = datetime.fromisoformat(event["timestamp"])
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError("audit timestamps must be non-decreasing")
        if event["assessment_version"] < previous_version:
            raise ValueError("audit assessment versions must be non-decreasing")
        if event["content_digest"] != qualitative_audit_event_digest(event):
            raise ValueError("audit content_digest does not match its event")
        actor = event["actor"]
        if actor["actor_type"] == "human" and actor["actor_id"] not in reviewer_ids:
            raise ValueError("human audit actor must identify a declared reviewer")
        if actor["actor_type"] == "ai":
            if event["action"] == "approve":
                raise ValueError("AI audit actor cannot approve an assessment")
            if "ai_provenance" not in event:
                raise ValueError(
                    "AI audit event requires provider and model provenance"
                )
        if event["action"] == "approve" and (
            actor["actor_type"] != "human" or actor["actor_id"] not in accountable
        ):
            raise ValueError("approval must be made by an accountable human reviewer")
        previous = event_ids[index - 1]
        previous_digest = event["content_digest"]
        previous_timestamp = timestamp
        previous_version = event["assessment_version"]
    if audit[-1]["assessment_version"] != payload["assessment_version"]:
        raise ValueError("final audit event must match assessment_version")
    if audit[-1]["assessment_content_digest"] != current_content_digest:
        raise ValueError("final audit event must bind the current assessment content")

    event_by_id = {event["event_id"]: event for event in audit}
    for question in questions:
        for judgement in question["judgements"]:
            if judgement["verification_state"] != "human_verified":
                continue
            override = judgement["human_override"]
            reviewer_id = override["reviewer_id"]
            event = event_by_id.get(override["audit_event_id"])
            if reviewer_id not in accountable or event is None:
                raise ValueError("AI human override must identify accountable review")
            if event["actor"] != {"actor_id": reviewer_id, "actor_type": "human"}:
                raise ValueError(
                    "AI human override actor does not match its review event"
                )
            if event["action"] not in {"review", "approve"}:
                raise ValueError("AI human override must reference review or approval")
            if (
                event["assessment_version"] != payload["assessment_version"]
                or event["assessment_content_digest"] != current_content_digest
            ):
                raise ValueError(
                    "AI human override must reference the current assessment"
                )

    approval_blocked = False
    for question in questions:
        human = [
            judgement
            for judgement in question["judgements"]
            if judgement["actor_type"] == "human"
        ]
        priorities = {judgement["priority_class"] for judgement in human}
        recommendations = {judgement["recommendation_class"] for judgement in human}
        unverified_ai = any(
            judgement["actor_type"] == "ai"
            and judgement["verification_state"] != "human_verified"
            for judgement in question["judgements"]
        )
        approval_blocked = approval_blocked or bool(
            question["missing_fields"]
            or unverified_ai
            or len(priorities) != 1
            or len(recommendations) != 1
        )
    if audit[-1]["action"] == "approve" and approval_blocked:
        raise ValueError("approval requires complete verified consensus")


def validate_qualitative_information_result_semantics(
    result: Mapping[str, object],
) -> None:
    """Validate result-state consistency and the non-cardinal output boundary."""
    try:
        Draft202012Validator(QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1).validate(result)
    except ValidationError as error:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise ValueError(
            f"invalid result at {path} (constraint: {error.validator})"
        ) from error

    payload = cast("Mapping[str, Any]", result)
    questions = cast("list[Mapping[str, Any]]", payload["question_results"])
    groups = cast("list[Mapping[str, Any]]", payload["priority_groups"])
    question_ids = _ids(questions, "question_id", "result question IDs")
    by_id = {question["question_id"]: question for question in questions}
    if [group["priority_class"] for group in groups] != list(PRIORITY_CLASSES):
        raise ValueError("priority groups must follow the declared complete order")
    grouped = [question_id for group in groups for question_id in group["question_ids"]]
    if len(grouped) != len(set(grouped)) or not set(grouped) <= set(question_ids):
        raise ValueError("priority groups must contain unique declared question IDs")
    for group in groups:
        for question_id in group["question_ids"]:
            if by_id[question_id]["resolved_priority_class"] != group["priority_class"]:
                raise ValueError("priority group conflicts with resolved priority")
    expected_grouped = {
        question["question_id"]
        for question in questions
        if question["resolved_priority_class"] is not None
    }
    if set(grouped) != expected_grouped:
        raise ValueError("priority groups must contain every resolved question")
    for question in questions:
        is_consensus = question["consensus_status"] == "consensus"
        resolved = (
            question["resolved_priority_class"] is not None
            and question["resolved_recommendation_class"] is not None
        )
        if is_consensus != resolved:
            raise ValueError("question consensus and resolved classes are inconsistent")
        if (
            question["resolved_priority_class"] is not None
            and question["resolved_priority_class"] not in question["priority_classes"]
        ):
            raise ValueError("resolved priority must be a reported priority class")
        if (
            question["resolved_recommendation_class"] is not None
            and question["resolved_recommendation_class"]
            not in question["recommendation_classes"]
        ):
            raise ValueError("resolved recommendation must be a reported class")
        if is_consensus and not question["verified_human_reviewers"]:
            raise ValueError("consensus requires a verified human reviewer")
        if (
            question["unverified_ai_contributors"]
            and question["consensus_status"] != "unverified"
        ):
            raise ValueError("unverified AI contributors require unverified status")
        if question["redaction_status"] != "none" and (
            question["information_question"] != "[REDACTED]"
            or any(
                rationale != "[REDACTED]"
                for rationale in question["rationale_by_reviewer"].values()
            )
        ):
            raise ValueError("redacted result content must use stable markers")
    expected_unresolved = [
        question["question_id"]
        for question in questions
        if question["consensus_status"] != "consensus"
    ]
    if payload["unresolved_question_ids"] != expected_unresolved:
        raise ValueError("unresolved question IDs do not match question states")
    statuses = {question["consensus_status"] for question in questions}
    if statuses & {"incomplete", "dissent"}:
        expected_workflow = "incomplete"
    elif "unverified" in statuses or payload["human_approval_status"] != "approved":
        expected_workflow = "unverified"
    else:
        expected_workflow = "complete"
    if payload["workflow_status"] != expected_workflow:
        raise ValueError("workflow status does not match question and approval states")
    if (
        payload["human_approval_status"] == "approved"
        and payload["workflow_status"] != "complete"
    ):
        raise ValueError("accountable approval requires a complete workflow")
    if (
        payload["workflow_status"] == "complete"
        and payload["human_approval_status"] != "approved"
    ):
        raise ValueError("complete workflow requires accountable human approval")


__all__ = [
    "PRIORITY_CLASSES",
    "QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1",
    "QUALITATIVE_INFORMATION_AUDIT_EVENT_SCHEMA_V1",
    "QUALITATIVE_INFORMATION_RENDERING_SCHEMA_V1",
    "QUALITATIVE_INFORMATION_RESULT_SCHEMA_V1",
    "RECOMMENDATION_CLASSES",
    "qualitative_assessment_content_digest",
    "qualitative_audit_event_digest",
    "validate_qualitative_information_result_semantics",
    "validate_qualitative_information_semantics",
]
