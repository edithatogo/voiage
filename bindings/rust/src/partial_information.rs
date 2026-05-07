//! Deterministic partial-information contracts for voiage-core.

use std::collections::BTreeSet;

use crate::domain::{
    validate_reporting_payload, AnalysisEnvelope, ApproximationStatus, DiagnosticStatus,
    Diagnostics, DomainError, EvppiSummary, MethodMaturity, MethodMetadata, Reporting,
};

const ANALYSIS_TYPE: &str = "evppi";
const METHOD_FAMILY: &str = "partial-information";
const DEFAULT_METHOD: &str = "deterministic-summary";
const DECISION_CONTEXT: &str = "parameter-set";

fn ensure_finite(value: f64, name: &'static str) -> Result<f64, DomainError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(DomainError::NonFinite(name))
    }
}

fn validate_parameter_names(parameter_names: &[String]) -> Result<Vec<String>, DomainError> {
    if parameter_names.is_empty() {
        return Err(DomainError::EmptyCollection("parameter_names"));
    }

    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(parameter_names.len());
    for name in parameter_names {
        if name.trim().is_empty() {
            return Err(DomainError::EmptyField("parameter_names"));
        }
        if !seen.insert(name.clone()) {
            return Err(DomainError::DuplicateValue("parameter_names"));
        }
        normalized.push(name.clone());
    }

    Ok(normalized)
}

/// Deterministic EVPPI summary contract built from fixed partial-information inputs.
pub fn evppi_contract(
    analysis_id: impl Into<String>,
    parameter_names: &[String],
    expected_current_value: f64,
    expected_partial_information_value: f64,
    expected_perfect_information: f64,
    method: Option<String>,
) -> Result<AnalysisEnvelope<EvppiSummary>, DomainError> {
    let analysis_id = analysis_id.into();
    if analysis_id.trim().is_empty() {
        return Err(DomainError::EmptyField("analysis_id"));
    }

    let parameter_names = validate_parameter_names(parameter_names)?;
    let expected_current_value = ensure_finite(expected_current_value, "expected_current_value")?;
    let expected_partial_information_value = ensure_finite(
        expected_partial_information_value,
        "expected_partial_information_value",
    )?;
    let expected_perfect_information =
        ensure_finite(expected_perfect_information, "expected_perfect_information")?;
    let evppi = (expected_partial_information_value - expected_current_value).max(0.0);
    let method_name = method.unwrap_or_else(|| DEFAULT_METHOD.to_string());

    let mut method_metadata = MethodMetadata::new(
        ANALYSIS_TYPE,
        METHOD_FAMILY,
        MethodMaturity::Stable,
        ApproximationStatus::Exact,
    )?;
    method_metadata.analysis_id = Some(analysis_id.clone());
    method_metadata.decision_context = Some(DECISION_CONTEXT.to_string());
    method_metadata
        .notes
        .push("Deterministic summary derived from fixed EVPPI inputs.".to_string());

    let mut diagnostics = Diagnostics::new(analysis_id.clone(), DiagnosticStatus::Ok)?;
    diagnostics.backend = None;
    let mut reporting =
        Reporting::cheers_voi(ANALYSIS_TYPE, METHOD_FAMILY, MethodMaturity::Stable)?;
    reporting.analysis_id = Some(analysis_id.clone());
    reporting.decision_context = Some(DECISION_CONTEXT.to_string());
    reporting.estimator = Some(method_name.clone());
    reporting.provenance.insert(
        "parameter_count".to_string(),
        parameter_names.len().to_string(),
    );
    reporting
        .provenance
        .insert("parameter_names".to_string(), parameter_names.join(","));
    reporting
        .reproducibility
        .insert("method".to_string(), method_name.clone());
    reporting.reproducibility.insert(
        "value_rule".to_string(),
        "partial_minus_current".to_string(),
    );
    reporting.diagnostics.insert(
        "parameter_count".to_string(),
        parameter_names.len().to_string(),
    );
    reporting
        .diagnostics
        .insert("status".to_string(), "ok".to_string());
    validate_reporting_payload(&reporting)?;

    let result = EvppiSummary {
        evppi,
        parameter_names,
        expected_current_value,
        expected_partial_information_value,
        expected_perfect_information,
        method: Some(method_name.clone()),
    };

    let mut envelope = AnalysisEnvelope::new(
        analysis_id,
        ANALYSIS_TYPE,
        method_metadata,
        diagnostics,
        reporting,
        result,
    )?;
    envelope.method = Some(method_name);
    Ok(envelope)
}
