"""Capability descriptions and fail-closed backend selection."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003 - public protocol signature
from enum import StrEnum
from typing import Literal, Protocol

import numpy as np  # noqa: TC002 - public protocol signature

from voiage.contracts.analysis import ContractModel, Identifier


class Capability(StrEnum):
    """Stable backend capability identifiers."""

    DENSE_ARRAY = "dense-array"
    DETERMINISTIC = "deterministic"
    JIT = "jit"
    AUTODIFF = "autodiff"
    BATCHING = "batching"
    STREAMING = "streaming"
    FREE_THREADED = "free-threaded"
    ARROW_C_STREAM = "arrow-c-stream"


LiteralDtype = Literal["float32", "float64"]


class BackendCapabilities(ContractModel):
    """Machine-readable capabilities of one calculation backend."""

    backend_name: Identifier
    backend_version: str
    method_families: frozenset[Identifier]
    dtypes: frozenset[LiteralDtype]
    devices: frozenset[Identifier]
    features: frozenset[Capability] = frozenset()


class KernelRequirements(ContractModel):
    """Capabilities required to execute one kernel under a policy."""

    method_family: Identifier
    dtype: LiteralDtype
    device: Identifier | None = None
    required_features: frozenset[Capability] = frozenset()


class CapabilityReport(ContractModel):
    """Explain whether a backend satisfies a kernel requirement."""

    backend_name: Identifier
    supported: bool
    missing: tuple[Identifier, ...] = ()


class CapabilityBackend(Protocol):
    """Minimal backend surface required by the generic dispatcher."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return immutable capability metadata."""
        ...

    def calculate_evpi(self, net_benefits: np.ndarray) -> float:
        """Execute the EVPI pilot operation."""
        ...


class UnsupportedCapabilityError(ValueError):
    """Raised when no candidate backend satisfies a kernel contract."""


def evaluate_backend(
    backend: CapabilityBackend, requirements: KernelRequirements
) -> CapabilityReport:
    """Evaluate a backend without silently weakening requirements."""
    capabilities = backend.capabilities
    missing: list[str] = []
    if requirements.method_family not in capabilities.method_families:
        missing.append(f"method:{requirements.method_family}")
    if requirements.dtype not in capabilities.dtypes:
        missing.append(f"dtype:{requirements.dtype}")
    if (
        requirements.device is not None
        and requirements.device not in capabilities.devices
    ):
        missing.append(f"device:{requirements.device}")
    missing.extend(
        f"capability:{item.value}"
        for item in sorted(
            requirements.required_features - capabilities.features,
            key=lambda value: value.value,
        )
    )
    return CapabilityReport(
        backend_name=capabilities.backend_name,
        supported=not missing,
        missing=tuple(missing),
    )


def select_backend(
    backends: Sequence[CapabilityBackend], requirements: KernelRequirements
) -> CapabilityBackend:
    """Return the first capable backend or fail with complete evidence."""
    reports = tuple(evaluate_backend(item, requirements) for item in backends)
    for backend, report in zip(backends, reports, strict=True):
        if report.supported:
            return backend
    detail = "; ".join(
        f"{item.backend_name}: {', '.join(item.missing) or 'unavailable'}"
        for item in reports
    )
    raise UnsupportedCapabilityError(
        f"No backend satisfies kernel requirements ({detail or 'no candidates'})"
    )
