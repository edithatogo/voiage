"""Capability-aware calculation-kernel and legacy-parity tests."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.contracts.adapters import adapt_backend
from voiage.contracts.analysis import AnalysisSpec, NumericalPolicy
from voiage.contracts.capabilities import (
    BackendCapabilities,
    Capability,
    CapabilityBackend,
    UnsupportedCapabilityError,
    select_backend,
)
from voiage.contracts.kernel import EvpiKernel, dispatch_calculation, run_evpi
from voiage.main_backends import get_backend


class _UnsupportedBackend:
    capabilities = BackendCapabilities(
        backend_name="limited",
        backend_version="1",
        method_families=frozenset({"evpi"}),
        dtypes=frozenset({"float32"}),
        devices=frozenset({"cpu"}),
        features=frozenset({Capability.DENSE_ARRAY}),
    )

    def calculate_evpi(self, net_benefits: np.ndarray) -> float:
        return float(net_benefits.mean())


class _JitBackend:
    capabilities = BackendCapabilities(
        backend_name="jit-capable",
        backend_version="1",
        method_families=frozenset({"evpi"}),
        dtypes=frozenset({"float64"}),
        devices=frozenset({"cpu"}),
        features=frozenset(
            {Capability.DENSE_ARRAY, Capability.DETERMINISTIC, Capability.JIT}
        ),
    )

    def calculate_evpi(self, net_benefits: np.ndarray) -> float:
        return float(net_benefits.max(axis=1).mean() - net_benefits.mean(axis=0).max())


class _UnsafeKernel:
    kernel_id = "unsafe"
    kernel_version = "1"
    method_maturity = "experimental"

    def requirements(self, spec, policy):
        return EvpiKernel().requirements(spec, policy)

    def calculate(self, spec, inputs, *, backend, policy, context):
        return np.asarray(inputs)


class _ExperimentalKernel(_UnsafeKernel):
    def calculate(self, spec, inputs, *, backend, policy, context):
        return EvpiKernel().calculate(
            spec,
            inputs,
            backend=backend,
            policy=policy,
            context=context,
        )


def _spec() -> AnalysisSpec:
    return AnalysisSpec(
        analysis_id="evpi-kernel-001",
        decision_problem_id="decision-001",
        method_family="evpi",
        method_contract_version="1.0.0",
        strategy_names=("A", "B"),
    )


def test_backend_selection_is_capability_aware_and_fail_closed() -> None:
    policy = NumericalPolicy(
        dtype="float64",
        required_capabilities=frozenset({Capability.DETERMINISTIC}),
    )
    requirements = EvpiKernel().requirements(_spec(), policy)
    with pytest.raises(UnsupportedCapabilityError, match="No backend satisfies"):
        select_backend([_UnsupportedBackend()], requirements)


def test_dispatch_requires_opt_in_and_discloses_backend_fallback() -> None:
    values = np.array([[10.0, 4.0], [2.0, 8.0]])
    strict_policy = NumericalPolicy(
        use_jit=True,
        backend_preference=("limited", "jit-capable"),
    )
    with pytest.raises(UnsupportedCapabilityError):
        dispatch_calculation(
            EvpiKernel(),
            _spec(),
            values,
            policy=strict_policy,
            backends=[_UnsupportedBackend(), _JitBackend()],
        )

    fallback = dispatch_calculation(
        EvpiKernel(),
        _spec(),
        values,
        policy=strict_policy.model_copy(update={"allow_fallback": True}),
        backends=[_UnsupportedBackend(), _JitBackend()],
    )
    assert fallback.run_context.selected_backend == "jit-capable"
    assert fallback.diagnostics.status == "degraded"
    assert fallback.diagnostics.degraded_paths == ("backend-fallback",)
    assert fallback.diagnostics.warnings[0].code == "backend_fallback"


def test_numpy_backend_adapter_declares_real_capabilities() -> None:
    backend: CapabilityBackend = adapt_backend(get_backend("numpy"))
    assert backend.capabilities.backend_name == "numpy"
    assert "evpi" in backend.capabilities.method_families
    assert Capability.DETERMINISTIC in backend.capabilities.features
    assert "float64" in backend.capabilities.dtypes


def test_evpi_kernel_preserves_legacy_scalar_numerical_result() -> None:
    values = np.array([[10.0, 4.0], [2.0, 8.0], [9.0, 3.0]])
    legacy = DecisionAnalysis(values, backend="numpy").evpi()
    envelope = run_evpi(
        values,
        spec=_spec(),
        policy=NumericalPolicy(backend_preference=("numpy",)),
    )
    assert envelope.payload.value == pytest.approx(legacy)
    assert envelope.run_context.selected_backend == "numpy"
    assert envelope.diagnostics.status == "ok"
    assert envelope.provenance.backend == "numpy"


def test_effective_policy_and_input_are_part_of_provenance() -> None:
    spec = _spec()
    result = run_evpi(
        np.array([[1.0, 2.0]], dtype=np.float64),
        spec=spec,
        policy=NumericalPolicy(dtype="float32", backend_preference=("numpy",)),
    )
    effective = spec.model_copy(update={"numerical_policy": result.numerical_policy})
    assert result.run_context.spec_digest == effective.contract_digest()
    assert result.run_context.input_digest is not None
    changed = run_evpi(
        np.array([[1.0, 3.0]], dtype=np.float64),
        spec=spec,
        policy=result.numerical_policy,
    )
    assert changed.run_context.input_digest != result.run_context.input_digest


def test_dispatch_rejects_non_contract_payloads_and_uses_kernel_maturity() -> None:
    with pytest.raises(TypeError, match="ContractModel"):
        dispatch_calculation(
            _UnsafeKernel(),
            _spec(),
            np.ones((1, 2)),
            policy=NumericalPolicy(),
            backends=[adapt_backend(get_backend("numpy"))],
        )
    experimental = dispatch_calculation(
        _ExperimentalKernel(),
        _spec().model_copy(update={"method_family": "evpi"}),
        np.ones((1, 2)),
        policy=NumericalPolicy(),
        backends=[adapt_backend(get_backend("numpy"))],
    )
    assert experimental.method_maturity == "experimental"
