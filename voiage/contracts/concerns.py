"""Typed concern and evidence records for auditable analysis governance."""

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic runtime annotation
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from voiage.contracts.analysis import ContractModel, Identifier


class Relation(ContractModel):
    """Typed relationship to another canonical governance record."""

    relation: Literal[
        "informs",
        "depends_on",
        "blocks",
        "mitigates",
        "mitigated_by",
        "resolved_by",
        "supersedes",
        "implements",
        "supports",
        "challenges",
    ]
    target_id: Identifier


class GovernanceRecord(ContractModel):
    """Fields shared by the pinned VOP governance record family."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    record_version: int = Field(default=1, ge=1)
    id: Identifier
    title: Identifier
    summary: Identifier
    repository: Literal["vop_poc_nz", "voiage", "shared"] = "shared"
    track_ids: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    evidence_reference_ids: tuple[str, ...] = ()
    issue_link_ids: tuple[str, ...] = ()
    relations: tuple[Relation, ...] = ()
    owner_role: str = "maintainer"
    priority: Literal["P0", "P1", "P2", "P3"] = "P2"
    moscow: Literal["must", "should", "could", "wont_now"] = "should"
    gate: Literal[
        "none", "local", "external", "human", "credential", "hardware", "publication"
    ] = "local"
    visibility: Literal["public", "repository", "local_private"] = "public"
    tags: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator(
        "track_ids",
        "requirement_ids",
        "evidence_reference_ids",
        "issue_link_ids",
        "relations",
        "tags",
        mode="before",
    )
    @classmethod
    def normalize_arrays(cls, value: object) -> object:
        """Accept canonical JSON arrays at the strict Python boundary."""
        return tuple(value) if isinstance(value, list) else value


class EvidenceReference(GovernanceRecord):
    """Pinned VOP evidence reference with privacy-safe local locators."""

    record_type: Literal["evidence_reference"] = "evidence_reference"
    status: Literal["unverified", "verified", "failed", "blocked", "superseded"]
    evidence_kind: Literal[
        "source",
        "derivation",
        "test",
        "benchmark",
        "run",
        "artifact",
        "review",
        "external_verification",
    ]
    locator_kind: Literal[
        "local_path",
        "url",
        "doi",
        "github_run",
        "commit",
        "pull_request",
        "issue",
        "release",
    ]
    locator: Identifier
    observed_at: datetime
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    claim_ids: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()
    challenges: tuple[str, ...] = ()

    @field_validator("claim_ids", "supports", "challenges", mode="before")
    @classmethod
    def normalize_evidence_arrays(cls, value: object) -> object:
        """Accept canonical JSON arrays at the strict Python boundary."""
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def protect_local_paths(self) -> Self:
        """Ensure local paths cannot enter public or repository projections."""
        if self.locator_kind == "local_path" and self.visibility != "local_private":
            raise ValueError("local_path evidence must use local_private visibility")
        return self


class ConcernSpec(GovernanceRecord):
    """Canonical concern compatible with the pinned VOP JSON Schema."""

    record_type: Literal["concern"] = "concern"
    status: Literal["open", "investigating", "monitoring", "resolved", "accepted"]
    question: Identifier
    impact_if_unresolved: Identifier
    resolution_criteria: tuple[Identifier, ...]
    raised_by_role: str = "reviewer"

    @field_validator("resolution_criteria", mode="before")
    @classmethod
    def normalize_resolution_criteria(cls, value: object) -> object:
        """Accept canonical JSON arrays at the strict Python boundary."""
        return tuple(value) if isinstance(value, list) else value
