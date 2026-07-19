"""Canonical machine-readable VOIAGE analysis contracts."""

from .adapters import LegacyBackendAdapter, adapt_backend
from .analysis import (
    AnalysisResult,
    AnalysisSpec,
    ContractModel,
    DiagnosticEnvelope,
    DiagnosticRecord,
    NumericalPolicy,
    ParameterSpec,
    Provenance,
    RunContext,
    ScalarPayload,
)
from .capabilities import (
    BackendCapabilities,
    Capability,
    CapabilityBackend,
    CapabilityReport,
    KernelRequirements,
    UnsupportedCapabilityError,
    evaluate_backend,
    select_backend,
)
from .concerns import ConcernSpec, EvidenceReference
from .kernel import CalculationKernel, EvpiKernel, dispatch_calculation, run_evpi

__all__ = [
    "AnalysisResult",
    "AnalysisSpec",
    "BackendCapabilities",
    "CalculationKernel",
    "Capability",
    "CapabilityBackend",
    "CapabilityReport",
    "ConcernSpec",
    "ContractModel",
    "DiagnosticEnvelope",
    "DiagnosticRecord",
    "EvidenceReference",
    "EvpiKernel",
    "KernelRequirements",
    "LegacyBackendAdapter",
    "NumericalPolicy",
    "ParameterSpec",
    "Provenance",
    "RunContext",
    "ScalarPayload",
    "UnsupportedCapabilityError",
    "adapt_backend",
    "dispatch_calculation",
    "evaluate_backend",
    "run_evpi",
    "select_backend",
]
