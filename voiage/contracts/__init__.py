"""Canonical machine-readable VOIAGE analysis contracts."""

from .adapters import (
    LegacyBackendAdapter,
    adapt_backend,
    adapt_parameter_set,
    adapt_value_array,
    analysis_spec_from_inputs,
)
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
from .perspective import PerspectivePayload, adapt_perspective_result, run_perspective

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
    "PerspectivePayload",
    "Provenance",
    "RunContext",
    "ScalarPayload",
    "UnsupportedCapabilityError",
    "adapt_backend",
    "adapt_parameter_set",
    "adapt_perspective_result",
    "adapt_value_array",
    "analysis_spec_from_inputs",
    "dispatch_calculation",
    "evaluate_backend",
    "run_evpi",
    "run_perspective",
    "select_backend",
]
