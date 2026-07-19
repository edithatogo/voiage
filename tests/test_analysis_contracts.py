"""Canonical Pydantic v2 analysis-contract tests for C13."""

from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from voiage.contracts.analysis import (
    AnalysisResult,
    AnalysisSpec,
    DiagnosticEnvelope,
    NumericalPolicy,
    ParameterSpec,
    Provenance,
    RunContext,
    ScalarPayload,
)
from voiage.contracts.concerns import ConcernSpec, EvidenceReference


def _spec() -> AnalysisSpec:
    return AnalysisSpec(
        analysis_id="evpi-screening-001",
        decision_problem_id="screening-program-001",
        method_family="evpi",
        method_contract_version="1.0.0",
        strategy_names=("usual-care", "screening"),
        parameters=(
            ParameterSpec(
                parameter_id="prevalence",
                role="uncertain",
                dtype="float64",
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        ),
    )


def test_contract_models_are_strict_frozen_and_extra_forbidding() -> None:
    with pytest.raises(ValidationError):
        ParameterSpec.model_validate(
            {
                "parameter_id": "prevalence",
                "role": "uncertain",
                "dtype": "float64",
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        NumericalPolicy(chunk_size="128")
    with pytest.raises(ValidationError):
        ParameterSpec(
            parameter_id="prevalence",
            role="uncertain",
            dtype="float64",
            lower_bound=1.0,
            upper_bound=0.0,
        )

    spec = _spec()
    with pytest.raises(ValidationError):
        spec.analysis_id = "changed"  # type: ignore[misc]


def test_concern_contract_carries_typed_privacy_aware_evidence() -> None:
    concern = ConcernSpec(
        concern_id="C13-capability-dispatch",
        title="Backend selection must be fail closed",
        category="architecture",
        severity="high",
        status="accepted",
        statement="A kernel must not execute on an incapable backend.",
        evidence=(
            EvidenceReference(
                artifact_id="test-capability-selection",
                kind="test",
                location="tests/test_calculation_kernel.py",
                visibility="repository",
            ),
        ),
    )
    assert concern.evidence[0].visibility == "repository"
    with pytest.raises(ValidationError):
        EvidenceReference(
            artifact_id="private",
            kind="test",
            location="C:/private/result.json",
            visibility="public",
            unexpected_field=True,  # type: ignore[call-arg]
        )


def test_analysis_spec_enforces_unique_identifiers_and_stable_digest() -> None:
    spec = _spec()
    assert (
        spec.contract_digest()
        == spec.model_validate_json(spec.model_dump_json()).contract_digest()
    )
    with pytest.raises(ValidationError, match="strategy_names must be unique"):
        spec.model_copy(update={"strategy_names": ("same", "same")}).model_validate(
            {
                **spec.model_dump(),
                "strategy_names": ("same", "same"),
            }
        )


def test_contract_json_schema_is_deterministic_and_closed() -> None:
    first = json.dumps(AnalysisSpec.model_json_schema(), sort_keys=True)
    second = json.dumps(AnalysisSpec.model_json_schema(), sort_keys=True)
    assert first == second
    schema = json.loads(first)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == "2.0.0"


def test_typed_result_envelope_round_trips_without_numpy_objects() -> None:
    spec = _spec()
    policy = NumericalPolicy()
    context = RunContext(
        run_id="run-001",
        spec_digest=spec.contract_digest(),
        selected_backend="numpy",
        device="cpu",
        capabilities=frozenset({"dense-array", "deterministic"}),
        package_version="0.0.0",
        python_version="3.14",
        platform="test",
    )
    result = AnalysisResult[ScalarPayload](
        analysis_id=spec.analysis_id,
        decision_problem_id=spec.decision_problem_id,
        method_family="evpi",
        method_contract_version="1.0.0",
        method_maturity="stable",
        numerical_policy=policy,
        payload=ScalarPayload(value=1.25),
        run_context=context,
        diagnostics=DiagnosticEnvelope(analysis_id=spec.analysis_id),
        provenance=Provenance(backend="numpy", method_family="evpi"),
    )
    restored = AnalysisResult[ScalarPayload].model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
    assert restored.payload.value == 1.25
