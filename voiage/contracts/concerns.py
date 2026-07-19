"""Typed concern and evidence records for auditable analysis governance."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from voiage.contracts.analysis import ContractModel, Identifier


class EvidenceReference(ContractModel):
    """Reference evidence without embedding private or binary content."""

    artifact_id: Identifier
    kind: Literal["test", "fixture", "benchmark", "document", "issue"]
    location: Identifier
    visibility: Literal["public", "repository", "local-private"] = "repository"
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    description: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class ConcernSpec(ContractModel):
    """Machine-readable concern linked to provenance-bearing evidence."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    concern_id: Identifier
    title: Identifier
    category: Literal[
        "architecture",
        "numerical",
        "compatibility",
        "security",
        "performance",
        "governance",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    status: Literal["proposed", "accepted", "mitigated", "blocked"]
    statement: Identifier
    evidence: tuple[EvidenceReference, ...] = ()
    owner: Identifier | None = None
    extensions: dict[str, JsonValue] = Field(default_factory=dict)
