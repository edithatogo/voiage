//! Rust core domain model and scalar contract helpers for voiage-core.

mod domain;
mod partial;
mod partial_information;
mod sample_information;
mod sample_information_stochastic;
mod scalar;

pub use domain::{
    validate_reporting_payload, AnalysisEnvelope, ApproximationStatus, DiagnosticSeverity,
    DiagnosticStatus, DiagnosticWarning, Diagnostics, DomainError, EnbsSummary, EvpiSummary,
    EvppiSummary, EvsiSummary, MethodMaturity, MethodMetadata, ParameterSet, Reporting, TrialArm,
    TrialDesign, ValueArray,
};
pub use partial::{
    calculate_ceaf, calculate_dominance, calculate_extended_dominance, calculate_icers,
    calculate_strong_dominance, cost_effectiveness_frontier, value_of_heterogeneity,
    CeafDiagnostics, CeafResult, DominanceDiagnostics, DominanceResult, HeterogeneityDiagnostics,
    HeterogeneityResult, PartialError, SummaryReportingEnvelope,
};
pub use partial_information::evppi_contract;
pub use sample_information::evsi_contract;
pub use sample_information_stochastic::evsi_stochastic_contract;
pub use scalar::{
    enbs, enbs_contract, evpi, evpi_contract, EnbsDiagnostics, EnbsResult, EvpiDiagnostics,
    EvpiResult, ReportingEnvelope, ScalarError,
};
