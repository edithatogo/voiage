"""Generic calculation-kernel protocol and the additive EVPI pilot."""

# pyright: reportUnnecessaryIsInstance=false, reportUnreachable=false, reportExplicitAny=false, reportInvalidTypeVarUse=false

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003 - public dispatcher signature
from importlib.metadata import PackageNotFoundError, version
import logging
import platform
from typing import Literal, Protocol, TypeVar
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray  # noqa: TC002 - public API signature

from voiage.contracts.adapters import adapt_backend
from voiage.contracts.analysis import (
    AnalysisResult,
    AnalysisSpec,
    ContractModel,
    DiagnosticEnvelope,
    DiagnosticRecord,
    NumericalPolicy,
    Provenance,
    RunContext,
    ScalarPayload,
)
from voiage.contracts.capabilities import (
    Capability,
    CapabilityBackend,
    KernelRequirements,
    select_backend,
)
from voiage.contracts.digests import array_digest
from voiage.logging import analysis_log_context, analysis_log_context_from_result
from voiage.main_backends import get_backend

_LOGGER = logging.getLogger("voiage.contracts.kernel")

SpecT_contra = TypeVar("SpecT_contra", contravariant=True)
InputT_contra = TypeVar("InputT_contra", contravariant=True)
PayloadT_co = TypeVar("PayloadT_co", bound=ContractModel, covariant=True)
PayloadT = TypeVar("PayloadT", bound=ContractModel)


class CalculationKernel(Protocol[SpecT_contra, InputT_contra, PayloadT_co]):
    """Generic method implementation dispatched through backend capabilities."""

    kernel_id: str
    kernel_version: str

    @property
    def method_maturity(
        self,
    ) -> Literal[
        "stable", "fixture-backed", "approximate", "experimental", "backend-dependent"
    ]:
        """Return evidence-backed maturity for result attribution."""
        ...

    def requirements(
        self, spec: SpecT_contra, policy: NumericalPolicy
    ) -> KernelRequirements:
        """Return requirements for this specification and policy."""
        ...

    def calculate(
        self,
        spec: SpecT_contra,
        inputs: InputT_contra,
        *,
        backend: CapabilityBackend,
        policy: NumericalPolicy,
        context: RunContext,
    ) -> PayloadT_co:
        """Calculate a typed payload using an already-approved backend."""
        ...


class EvpiKernel:
    """Capability-aware EVPI kernel preserving the legacy scalar calculation."""

    kernel_id: str = "voiage.evpi"
    kernel_version: str = "1.0.0"
    method_maturity: Literal["stable"] = "stable"

    def requirements(
        self, spec: AnalysisSpec, policy: NumericalPolicy
    ) -> KernelRequirements:
        """Require deterministic dense-array support for the EVPI pilot."""
        if spec.method_family != "evpi":
            raise ValueError("EvpiKernel requires method_family='evpi'")
        required = {Capability.DENSE_ARRAY, Capability.DETERMINISTIC}
        required.update(Capability(item) for item in policy.required_capabilities)
        if policy.use_jit:
            required.add(Capability.JIT)
        return KernelRequirements(
            method_family="evpi",
            dtype=policy.dtype,
            device=policy.device,
            required_features=frozenset(required),
        )

    def calculate(
        self,
        spec: AnalysisSpec,
        inputs: NDArray[np.generic],
        *,
        backend: CapabilityBackend,
        policy: NumericalPolicy,
        context: RunContext,
    ) -> ScalarPayload:
        """Delegate EVPI and normalize the established scalar result."""
        del spec, policy, context
        return ScalarPayload(value=backend.calculate_evpi(inputs))


def _package_version() -> str:
    try:
        return version("voiage")
    except PackageNotFoundError:  # pragma: no cover - editable installs provide it
        return "0.0.0"


def dispatch_calculation[PayloadT: ContractModel](
    kernel: CalculationKernel[AnalysisSpec, NDArray[np.generic], PayloadT],
    spec: AnalysisSpec,
    inputs: NDArray[np.generic],
    *,
    policy: NumericalPolicy,
    backends: Sequence[CapabilityBackend],
) -> AnalysisResult[PayloadT]:
    """Select a capable backend, execute, and produce a typed envelope."""
    effective_spec = (
        spec
        if spec.numerical_policy == policy
        else spec.model_copy(update={"numerical_policy": policy})
    )
    requirements = kernel.requirements(effective_spec, policy)
    candidates = backends if policy.allow_fallback else backends[:1]
    backend = select_backend(candidates, requirements)
    capabilities = backend.capabilities
    fallback_used = bool(backends) and backend is not backends[0]
    package_version = _package_version()
    context = RunContext(
        run_id=uuid4().hex,
        spec_digest=effective_spec.contract_digest(),
        input_digest=array_digest(inputs),
        requested_backend=policy.backend_preference[0]
        if policy.backend_preference
        else None,
        selected_backend=capabilities.backend_name,
        backend_version=capabilities.backend_version,
        device=policy.device or sorted(capabilities.devices)[0],
        capabilities=frozenset(item.value for item in capabilities.features),
        package_version=package_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
    payload = kernel.calculate(
        effective_spec,
        inputs,
        backend=backend,
        policy=policy,
        context=context,
    )
    payload_object: object = payload
    if not isinstance(payload_object, ContractModel):
        raise TypeError("calculation kernels must return a ContractModel payload")
    return AnalysisResult(
        analysis_id=effective_spec.analysis_id,
        decision_problem_id=effective_spec.decision_problem_id,
        method_family=effective_spec.method_family,
        method_contract_version=effective_spec.method_contract_version,
        method_maturity=kernel.method_maturity,
        numerical_policy=policy,
        payload=payload,
        run_context=context,
        diagnostics=DiagnosticEnvelope(
            analysis_id=effective_spec.analysis_id,
            status="degraded" if fallback_used else "ok",
            backend=capabilities.backend_name,
            warnings=(
                DiagnosticRecord(
                    severity="warning",
                    code="backend_fallback",
                    message=(
                        f"Requested backend could not satisfy the kernel; "
                        f"used {capabilities.backend_name}."
                    ),
                    capability="kernel-requirements",
                    backend=capabilities.backend_name,
                ),
            )
            if fallback_used
            else (),
            degraded_paths=("backend-fallback",) if fallback_used else (),
        ),
        provenance=Provenance(
            backend=capabilities.backend_name,
            method_family=effective_spec.method_family,
            package_version=package_version,
            seed=policy.seed,
            input_artifact_ids=effective_spec.input_artifact_ids,
            details={
                "backend_fallback": fallback_used,
                "kernel_id": kernel.kernel_id,
                "kernel_version": kernel.kernel_version,
            },
        ),
    )


def run_evpi(
    net_benefits: ArrayLike,
    *,
    spec: AnalysisSpec,
    policy: NumericalPolicy | None = None,
) -> AnalysisResult[ScalarPayload]:
    """Run the opt-in EVPI contract without changing legacy APIs."""
    resolved_policy = policy or spec.numerical_policy
    dtype = np.float64 if resolved_policy.dtype == "float64" else np.float32
    values = np.asarray(net_benefits, dtype=dtype)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("net_benefits must be a non-empty 2D array")
    if spec.strategy_names and values.shape[1] != len(spec.strategy_names):
        raise ValueError("strategy_names must align with net_benefits columns")
    candidates = [
        adapt_backend(get_backend(name)) for name in resolved_policy.backend_preference
    ]
    result = dispatch_calculation(
        EvpiKernel(),
        spec,
        values,
        policy=resolved_policy,
        backends=candidates,
    )
    with analysis_log_context(analysis_log_context_from_result(result)):
        _LOGGER.info("analysis_completed")
    return result
