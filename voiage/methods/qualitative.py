"""Auditable qualitative Value of Information assessments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class QualitativeVOIResult:
    """Structured analyst assessment; deliberately contains no numeric VOI."""

    information_source: str
    decision: str
    value_rating: str
    rationale: str
    evidence_ids: tuple[str, ...]
    analyst: str


def assess_qualitative_voi(
    information_source: str,
    decision: str,
    value_rating: str,
    rationale: str,
    evidence_ids: Sequence[str],
    analyst: str,
) -> QualitativeVOIResult:
    """Create a fail-closed, auditable qualitative VOI record."""
    allowed = {"none", "low", "moderate", "high", "critical"}
    if value_rating not in allowed:
        raise ValueError(f"value_rating must be one of {sorted(allowed)}")
    if not all(str(item).strip() for item in (information_source, decision, rationale, analyst)):
        raise ValueError("source, decision, rationale, and analyst are required")
    ids = tuple(str(item) for item in evidence_ids if str(item).strip())
    if not ids:
        raise ValueError("at least one evidence identifier is required")
    return QualitativeVOIResult(information_source, decision, value_rating, rationale, ids, analyst)
