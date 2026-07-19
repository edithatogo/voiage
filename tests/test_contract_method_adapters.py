"""C13 additive input and perspective result adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.contracts.adapters import (
    adapt_backend,
    adapt_parameter_set,
    adapt_value_array,
    analysis_spec_from_inputs,
)
from voiage.contracts.analysis import NumericalPolicy
from voiage.contracts.capabilities import BackendCapabilities
from voiage.contracts.perspective import (
    adapt_perspective_result,
    run_perspective,
)
from voiage.main_backends import get_backend
from voiage.methods.perspective import value_of_perspective
from voiage.schema import ParameterSet, ValueArray


def _values() -> np.ndarray:
    return np.array(
        [
            [[10.0, 0.0], [0.0, 20.0]],
            [[10.0, 0.0], [0.0, 20.0]],
        ]
    )


def test_legacy_backend_exposes_non_abstract_capability_descriptor() -> None:
    backend = get_backend("numpy")
    descriptor = backend.capability_descriptor
    assert isinstance(descriptor, BackendCapabilities)
    assert descriptor == adapt_backend(backend).capabilities
    assert "value_of_perspective" in descriptor.method_families


def test_value_array_and_parameter_set_adapters_preserve_existing_instances() -> None:
    values = ValueArray.from_numpy_perspectives(
        _values(),
        strategy_names=["A", "B"],
        perspective_names=["payer", "societal"],
    )
    parameters = ParameterSet.from_numpy_or_dict({"prevalence": np.array([0.1, 0.2])})
    assert adapt_value_array(values) is values
    assert adapt_parameter_set(parameters) is parameters

    converted_values = adapt_value_array(
        _values(),
        strategy_names=("A", "B"),
        perspective_names=("payer", "societal"),
    )
    converted_parameters = adapt_parameter_set({"prevalence": np.array([0.1, 0.2])})
    assert converted_values.strategy_names == ["A", "B"]
    assert converted_values.perspective_names == ["payer", "societal"]
    assert converted_parameters is not None
    assert converted_parameters.parameter_names == ["prevalence"]

    spec = analysis_spec_from_inputs(
        analysis_id="perspective-001",
        decision_problem_id="decision-001",
        method_family="value_of_perspective",
        method_contract_version="1.1.0",
        values=converted_values,
        parameters=converted_parameters,
    )
    assert spec.strategy_names == ("A", "B")
    assert spec.parameters[0].parameter_id == "prevalence"
    assert spec.parameters[0].dtype == "float64"


def test_perspective_envelope_preserves_existing_dataclass_result() -> None:
    value_array = ValueArray.from_numpy_perspectives(
        _values(),
        strategy_names=["A", "B"],
        perspective_names=["payer", "societal"],
    )
    legacy = value_of_perspective(value_array)
    payload = adapt_perspective_result(legacy)
    envelope = run_perspective(
        value_array,
        analysis_id="perspective-001",
        decision_problem_id="decision-001",
        policy=NumericalPolicy(backend_preference=("numpy",)),
    )

    assert payload.value == pytest.approx(legacy.value)
    assert payload.regret_matrix == tuple(
        tuple(float(item) for item in row) for row in legacy.regret_matrix
    )
    assert envelope.payload == payload
    assert envelope.run_context.selected_backend == "numpy"
    assert envelope.method_contract_version == "1.1.0"
    assert envelope.model_dump(mode="json")["payload"]["strategy_names"] == [
        "A",
        "B",
    ]


def test_perspective_fallback_is_honest_and_capabilities_fail_closed() -> None:
    values = _values()
    fallback = run_perspective(
        values,
        analysis_id="perspective-fallback",
        decision_problem_id="decision-001",
        strategy_names=("A", "B"),
        perspective_names=("payer", "societal"),
        policy=NumericalPolicy(
            backend_preference=("jax", "numpy"),
            allow_fallback=True,
        ),
    )
    assert fallback.run_context.selected_backend == "numpy"
    assert fallback.diagnostics.status == "degraded"
    assert fallback.diagnostics.warnings[0].code == "backend_fallback"

    with pytest.raises(ValueError, match="kernel requirements"):
        run_perspective(
            values,
            analysis_id="perspective-jit",
            decision_problem_id="decision-001",
            strategy_names=("A", "B"),
            perspective_names=("payer", "societal"),
            policy=NumericalPolicy(use_jit=True),
        )


def test_value_array_label_overrides_cannot_diverge_from_spec() -> None:
    value_array = ValueArray.from_numpy_perspectives(
        _values(),
        strategy_names=["A", "B"],
        perspective_names=["payer", "societal"],
    )
    with pytest.raises(ValueError, match="strategy_names"):
        run_perspective(
            value_array,
            analysis_id="perspective-labels",
            decision_problem_id="decision-001",
            strategy_names=("X", "Y"),
        )
