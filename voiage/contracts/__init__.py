"""Canonical machine-readable VOIAGE analysis contracts."""

# pyright: reportUnknownVariableType=false

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
    InterchangeIdentity,
    NumericalPolicy,
    ParameterSpec,
    Provenance,
    RunContext,
    ScalarPayload,
)
from .bundle import (
    BundleVerificationError,
    ContractPerformanceBudget,
    SchemaEvolutionReport,
    VerifiedContractBundle,
    canonical_bundle_digest,
    validate_schema_evolution,
    verify_contract_bundle,
    verify_pinned_contract_bundle,
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
from .interchange import (
    analysis_result_table,
    schema_fingerprint,
    write_analysis_result_ipc,
    write_analysis_result_parquet,
)
from .kernel import CalculationKernel, EvpiKernel, dispatch_calculation, run_evpi
from .normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
    VOIBinding,
)
from .perspective import PerspectivePayload, adapt_perspective_result, run_perspective
from .preparation import PreparedAnalysisInputs, prepare_analysis_inputs

__all__ = [
    "AnalysisResult",
    "AnalysisSpec",
    "BackendCapabilities",
    "BundleVerificationError",
    "CalculationKernel",
    "Capability",
    "CapabilityBackend",
    "CapabilityReport",
    "ConcernSpec",
    "ContractModel",
    "ContractPerformanceBudget",
    "DatasetManifest",
    "DiagnosticEnvelope",
    "DiagnosticRecord",
    "EvidenceReference",
    "EvpiKernel",
    "FieldManifest",
    "InterchangeIdentity",
    "KernelRequirements",
    "LegacyBackendAdapter",
    "NormalizedInputBundle",
    "NumericalPolicy",
    "ParameterSpec",
    "PerspectivePayload",
    "PreparedAnalysisInputs",
    "Provenance",
    "RunContext",
    "ScalarPayload",
    "SchemaEvolutionReport",
    "SourceProvenance",
    "TableManifest",
    "UnsupportedCapabilityError",
    "VOIBinding",
    "VerifiedContractBundle",
    "adapt_backend",
    "adapt_parameter_set",
    "adapt_perspective_result",
    "adapt_value_array",
    "analysis_result_table",
    "analysis_spec_from_inputs",
    "canonical_bundle_digest",
    "dispatch_calculation",
    "evaluate_backend",
    "prepare_analysis_inputs",
    "run_evpi",
    "run_perspective",
    "schema_fingerprint",
    "select_backend",
    "validate_schema_evolution",
    "verify_contract_bundle",
    "verify_pinned_contract_bundle",
    "write_analysis_result_ipc",
    "write_analysis_result_parquet",
]
