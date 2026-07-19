"""Strict, immutable analysis and result contracts."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic resolves runtime annotations
import hashlib
import json
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_serializer,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
ParameterDType = Literal["float32", "float64", "int64", "bool", "string"]


class ContractModel(BaseModel):
    """Common fail-closed configuration for public contract models."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
        defer_build=True,
    )


class ParameterSpec(ContractModel):
    """Machine-readable semantics for one analysis parameter."""

    parameter_id: Identifier
    label: Identifier | None = None
    role: Literal["uncertain", "decision", "design", "outcome", "hyperparameter"]
    dtype: ParameterDType
    dimensions: tuple[Identifier, ...] = ()
    unit: Identifier | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    distribution: dict[str, JsonValue] | None = None
    required: bool = True
    description: str | None = None
    extensions: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterSpec:
        """Reject inverted closed bounds."""
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")
        return self


class NumericalPolicy(ContractModel):
    """Explicit numerical equivalence and execution policy."""

    dtype: Literal["float32", "float64"] = "float64"
    absolute_tolerance: float = Field(default=1e-10, ge=0.0)
    relative_tolerance: float = Field(default=1e-8, ge=0.0)
    execution_mode: Literal["deterministic", "stochastic"] = "deterministic"
    seed: int | None = Field(default=None, ge=0)
    deterministic_fixture_mode: bool = False
    nan_policy: Literal["raise", "propagate", "omit"] = "raise"
    overflow_policy: Literal["raise", "warn"] = "raise"
    backend_preference: tuple[Identifier, ...] = ("numpy",)
    required_capabilities: frozenset[str] = frozenset()
    allow_fallback: bool = False
    use_jit: bool = False
    device: Identifier | None = None
    chunk_size: int | None = Field(default=None, ge=1)
    parallelism: int | None = Field(default=None, ge=1)

    @field_serializer("required_capabilities", when_used="json")
    def serialize_required_capabilities(self, value: frozenset[str]) -> tuple[str, ...]:
        """Serialize capability sets in canonical order."""
        return tuple(sorted(value))


class AnalysisSpec(ContractModel):
    """Declarative identity and scientific intent for one analysis."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    analysis_id: Identifier
    decision_problem_id: Identifier
    method_family: Identifier
    method_contract_version: Identifier
    strategy_names: tuple[Identifier, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    method_options: dict[str, JsonValue] = Field(default_factory=dict)
    input_artifact_ids: tuple[Identifier, ...] = ()
    numerical_policy: NumericalPolicy = Field(default_factory=NumericalPolicy)
    tags: frozenset[Identifier] = frozenset()
    extensions: dict[str, JsonValue] = Field(default_factory=dict)

    @field_serializer("tags", when_used="json")
    def serialize_tags(self, value: frozenset[str]) -> tuple[str, ...]:
        """Serialize tag sets in canonical order."""
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_unique_identifiers(self) -> AnalysisSpec:
        """Require stable identifiers to be unique within the specification."""
        if len(set(self.strategy_names)) != len(self.strategy_names):
            raise ValueError("strategy_names must be unique")
        parameter_ids = tuple(item.parameter_id for item in self.parameters)
        if len(set(parameter_ids)) != len(parameter_ids):
            raise ValueError("parameter identifiers must be unique")
        return self

    def canonical_json(self) -> str:
        """Return deterministic JSON used for provenance fingerprints."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def contract_digest(self) -> str:
        """Return the SHA-256 digest of the canonical specification JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class RunContext(ContractModel):
    """Auditable runtime identity captured at dispatch time."""

    run_id: Identifier
    spec_digest: Identifier
    input_digest: Identifier | None = None
    requested_backend: Identifier | None = None
    selected_backend: Identifier
    backend_version: str | None = None
    device: Identifier
    capabilities: frozenset[str] = frozenset()
    package_version: str
    python_version: str
    platform: str
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_serializer("capabilities", when_used="json")
    def serialize_capabilities(self, value: frozenset[str]) -> tuple[str, ...]:
        """Serialize runtime capabilities in canonical order."""
        return tuple(sorted(value))


class DiagnosticRecord(ContractModel):
    """One stable warning or execution caveat."""

    severity: Literal["info", "warning", "critical"]
    code: Identifier
    message: Identifier
    capability: Identifier | None = None
    backend: Identifier | None = None


class DiagnosticEnvelope(ContractModel):
    """Trust and degradation metadata kept separate from provenance."""

    analysis_id: Identifier
    status: Literal["ok", "degraded", "unsupported", "approximate"] = "ok"
    backend: Identifier | None = None
    warnings: tuple[DiagnosticRecord, ...] = ()
    unsupported_capabilities: tuple[Identifier, ...] = ()
    degraded_paths: tuple[Identifier, ...] = ()
    approximation_caveats: tuple[str, ...] = ()


class Provenance(ContractModel):
    """Replay metadata describing how a result was produced."""

    backend: Identifier
    method_family: Identifier
    package_version: str | None = None
    seed: int | None = Field(default=None, ge=0)
    fixture_id: Identifier | None = None
    input_artifact_ids: tuple[Identifier, ...] = ()
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ScalarPayload(ContractModel):
    """Typed payload for scalar calculation pilots."""

    value: float


class AnalysisResult[PayloadT](ContractModel):
    """Generic, JSON-safe result envelope for calculation kernels."""

    schema_version: Literal["2.0.0"] = "2.0.0"
    analysis_id: Identifier
    decision_problem_id: Identifier
    method_family: Identifier
    method_contract_version: Identifier
    method_maturity: Literal[
        "stable",
        "fixture-backed",
        "approximate",
        "experimental",
        "backend-dependent",
    ]
    numerical_policy: NumericalPolicy
    payload: PayloadT
    run_context: RunContext
    diagnostics: DiagnosticEnvelope
    provenance: Provenance
