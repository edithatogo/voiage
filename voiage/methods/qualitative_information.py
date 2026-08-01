"""Experimental non-cardinal qualitative information assessment workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, TypedDict

from voiage.contracts.qualitative_information import (
    validate_qualitative_information_result_semantics,
    validate_qualitative_information_semantics,
)
from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping


class _PriorityGroup(TypedDict):
    priority_class: str
    question_ids: list[str]


@dataclass(frozen=True)
class QualitativeQuestionResult:
    """Transparent non-cardinal disposition for one information question."""

    question_id: str
    information_question: str
    priority_classes: list[str]
    recommendation_classes: list[str]
    resolved_priority_class: str | None
    resolved_recommendation_class: str | None
    consensus_status: str
    verified_human_reviewers: list[str]
    unverified_ai_contributors: list[str]
    missing_fields: list[str]
    redaction_status: str
    rationale_by_reviewer: dict[str, str]


@dataclass(frozen=True)
class QualitativeInformationResult:
    """Versioned result for a qualitative information assessment."""

    schema_version: str
    assessment_id: str
    assessment_version: int
    analysis_type: str
    method_maturity: str
    numerical_estimand: bool
    workflow_status: str
    human_approval_status: str
    priority_groups: list[_PriorityGroup]
    unresolved_question_ids: list[str]
    question_results: list[QualitativeQuestionResult]
    audit_history_digest: str
    audit_event_count: int
    source_summary: dict[str, int]
    provenance: dict[str, str]
    diagnostics: dict[str, object]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-compatible result contract."""
        return asdict(self)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_unique(values: list[str], order: list[str]) -> list[str]:
    present = set(values)
    return [item for item in order if item in present]


def qualitative_information_from_specification(
    specification: Mapping[str, object],
) -> QualitativeInformationResult:
    """Validate and evaluate a portable qualitative information assessment.

    The workflow preserves declared ordinal classes. It performs no averaging,
    weighting, interpolation or conversion to probability, utility or money.
    """
    try:
        payload = json.loads(json.dumps(specification, ensure_ascii=False))
        validate_qualitative_information_semantics(payload)
    except (TypeError, ValueError, OverflowError) as error:
        raise_input_error(str(error))

    policy = payload["policy"]
    priority_order = policy["priority_order"]
    recommendation_order = policy["recommendation_order"]
    question_results: list[QualitativeQuestionResult] = []

    for question in sorted(payload["questions"], key=lambda item: item["question_id"]):
        human = [
            item
            for item in question["judgements"]
            if item["actor_type"] == "human"
            and item["verification_state"] == "verified"
        ]
        unverified_ai = sorted(
            item["reviewer_id"]
            for item in question["judgements"]
            if item["actor_type"] == "ai"
            and item["verification_state"] != "human_verified"
        )
        priorities = _ordered_unique(
            [item["priority_class"] for item in human], priority_order
        )
        recommendations = _ordered_unique(
            [item["recommendation_class"] for item in human],
            recommendation_order,
        )
        missing = sorted(question["missing_fields"])
        if unverified_ai:
            consensus = "unverified"
        elif missing or not human:
            consensus = "incomplete"
        elif len(priorities) > 1 or len(recommendations) > 1:
            consensus = "dissent"
        else:
            consensus = "consensus"
        resolved_priority = priorities[0] if consensus == "consensus" else None
        resolved_recommendation = (
            recommendations[0] if consensus == "consensus" else None
        )
        question_results.append(
            QualitativeQuestionResult(
                question_id=question["question_id"],
                information_question=(
                    "[REDACTED]"
                    if question["redaction_status"] != "none"
                    else question["information_question"]
                ),
                priority_classes=priorities,
                recommendation_classes=recommendations,
                resolved_priority_class=resolved_priority,
                resolved_recommendation_class=resolved_recommendation,
                consensus_status=consensus,
                verified_human_reviewers=sorted(item["reviewer_id"] for item in human),
                unverified_ai_contributors=unverified_ai,
                missing_fields=missing,
                redaction_status=question["redaction_status"],
                rationale_by_reviewer={
                    item["reviewer_id"]: (
                        "[REDACTED]"
                        if question["redaction_status"] != "none"
                        or any(
                            source["access_status"] != "accessible"
                            for source in payload["sources"]
                            if source["source_id"] in item["source_ids"]
                        )
                        else item["rationale"]
                    )
                    for item in sorted(
                        question["judgements"],
                        key=lambda judgement: judgement["reviewer_id"],
                    )
                },
            )
        )

    unresolved = [
        item.question_id
        for item in question_results
        if item.consensus_status != "consensus"
    ]
    groups: list[_PriorityGroup] = [
        {
            "priority_class": priority,
            "question_ids": [
                item.question_id
                for item in question_results
                if item.resolved_priority_class == priority
            ],
        }
        for priority in priority_order
    ]
    accountable = set(payload["decision"]["accountable_reviewer_ids"])
    final_event = payload["audit_history"][-1]
    approved = (
        final_event["action"] == "approve"
        and final_event["actor"]["actor_type"] == "human"
        and final_event["actor"]["actor_id"] in accountable
        and final_event["assessment_version"] == payload["assessment_version"]
    )
    statuses = {item.consensus_status for item in question_results}
    if statuses & {"incomplete", "dissent"}:
        workflow_status = "incomplete"
    elif "unverified" in statuses or not approved:
        workflow_status = "unverified"
    else:
        workflow_status = "complete"
    human_approval_status = "approved" if workflow_status == "complete" else "pending"
    source_summary = {
        status: sum(source["access_status"] == status for source in payload["sources"])
        for status in ("accessible", "redacted", "unavailable")
    }

    result = QualitativeInformationResult(
        schema_version="1.0.0",
        assessment_id=payload["assessment_id"],
        assessment_version=payload["assessment_version"],
        analysis_type="qualitative_information_assessment",
        method_maturity="experimental",
        numerical_estimand=False,
        workflow_status=workflow_status,
        human_approval_status=human_approval_status,
        priority_groups=groups,
        unresolved_question_ids=unresolved,
        question_results=question_results,
        audit_history_digest=_canonical_digest(payload["audit_history"]),
        audit_event_count=len(payload["audit_history"]),
        source_summary=source_summary,
        provenance={
            key: payload["provenance"][key]
            for key in ("fixture_id", "contract_reference")
        },
        diagnostics={
            "aggregation_policy": "none_non_cardinal",
            "classification_policy": "accountable_human_declared_no_derivation",
            "conflict_policy": policy["conflict_policy"],
            "missingness_policy": policy["missingness_policy"],
            "ai_policy": policy["ai_policy"],
            "tie_policy": policy["tie_policy"],
            "privacy_boundary": "redacted_content_not_returned",
            "quantitative_voi_claim": False,
        },
    )
    validate_qualitative_information_result_semantics(result.to_contract_dict())
    return result


def render_qualitative_information_text(
    result: QualitativeInformationResult,
) -> str:
    """Render a deterministic text alternative without colour-only semantics."""
    lines = [
        "Qualitative information assessment",
        f"Assessment: {result.assessment_id} (version {result.assessment_version})",
        f"Workflow status: {result.workflow_status}",
        f"Human approval: {result.human_approval_status}",
        "Priority groups:",
    ]
    for group in result.priority_groups:
        questions = ", ".join(group["question_ids"]) or "none"
        lines.append(f"- {group['priority_class']}: {questions}")
    unresolved = ", ".join(result.unresolved_question_ids) or "none"
    lines.append(f"Unresolved questions: {unresolved}")
    lines.append("Question dispositions:")
    for question in result.question_results:
        priority = question.resolved_priority_class or "/".join(
            question.priority_classes
        )
        recommendation = question.resolved_recommendation_class or "/".join(
            question.recommendation_classes
        )
        redaction = " [REDACTED]" if question.redaction_status != "none" else ""
        lines.append(
            f"- {question.question_id}: {question.consensus_status}; "
            f"priority={priority or 'unresolved'}; "
            f"recommendation={recommendation or 'unresolved'}{redaction}"
        )
    lines.append(
        "Boundary: ordinal workflow only; no probability, utility, currency or numerical VOI."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "QualitativeInformationResult",
    "QualitativeQuestionResult",
    "qualitative_information_from_specification",
    "render_qualitative_information_text",
]
