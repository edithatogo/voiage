//! Deterministic sample-information contracts for voiage-core.

use std::collections::BTreeMap;

use crate::domain::{
    validate_reporting_payload, AnalysisEnvelope, ApproximationStatus, DiagnosticStatus,
    Diagnostics, DomainError, EvsiSummary, MethodMaturity, MethodMetadata, Reporting, TrialDesign,
    ValueArray,
};

const ANALYSIS_TYPE: &str = "evsi";
const METHOD_FAMILY: &str = "sample-information";
const DEFAULT_METHOD: &str = "deterministic-summary";
const DECISION_CONTEXT: &str = "trial-design";

fn summarize_trial_design(trial_design: &TrialDesign) -> Result<usize, DomainError> {
    if trial_design.arms.is_empty() {
        return Err(DomainError::EmptyCollection("arms"));
    }

    let mut seen_arm_ids = BTreeMap::new();
    let mut sample_size = 0usize;

    for arm in &trial_design.arms {
        if arm.arm_id.trim().is_empty() {
            return Err(DomainError::EmptyField("arm_id"));
        }
        if arm.name.trim().is_empty() {
            return Err(DomainError::EmptyField("name"));
        }
        if arm.sample_size == 0 {
            return Err(DomainError::EmptyCollection("sample_size"));
        }
        if seen_arm_ids.insert(arm.arm_id.clone(), ()).is_some() {
            return Err(DomainError::DuplicateValue("arm_id"));
        }
        sample_size = sample_size.saturating_add(arm.sample_size);
    }

    Ok(sample_size)
}

fn validate_value_array(value_array: &ValueArray) -> Result<(usize, usize), DomainError> {
    if value_array.value_array_id.trim().is_empty() {
        return Err(DomainError::EmptyField("value_array_id"));
    }
    if value_array.sample_count == 0 {
        return Err(DomainError::EmptyCollection("net_benefit"));
    }
    if value_array.strategy_names.is_empty() {
        return Err(DomainError::EmptyCollection("strategy_names"));
    }
    if value_array.sample_count != value_array.net_benefit.len() {
        return Err(DomainError::SampleCountMismatch {
            expected: value_array.net_benefit.len(),
            actual: value_array.sample_count,
        });
    }

    if value_array
        .strategy_names
        .iter()
        .any(|name| name.trim().is_empty())
    {
        return Err(DomainError::EmptyField("strategy_names"));
    }
    let mut seen_names = BTreeMap::new();
    for name in &value_array.strategy_names {
        if seen_names.insert(name.clone(), ()).is_some() {
            return Err(DomainError::DuplicateValue("strategy_names"));
        }
    }

    let width = value_array.net_benefit[0].len();
    if width == 0 {
        return Err(DomainError::EmptyCollection("net_benefit"));
    }
    if width != value_array.strategy_names.len() {
        return Err(DomainError::WidthMismatch {
            expected: value_array.strategy_names.len(),
            actual: width,
        });
    }

    for row in &value_array.net_benefit {
        if row.len() != width {
            return Err(DomainError::RaggedMatrix("net_benefit"));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(DomainError::NonFinite("net_benefit"));
        }
    }

    Ok((value_array.sample_count, width))
}

fn strategy_means(matrix: &[Vec<f64>]) -> Vec<f64> {
    let sample_count = matrix.len() as f64;
    let strategy_count = matrix[0].len();
    let mut totals = vec![0.0; strategy_count];
    for row in matrix {
        for (index, value) in row.iter().enumerate() {
            totals[index] += value;
        }
    }
    totals.into_iter().map(|sum| sum / sample_count).collect()
}

fn row_max_mean(matrix: &[Vec<f64>]) -> f64 {
    let sample_count = matrix.len() as f64;
    let mut total = 0.0;
    for row in matrix {
        let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        total += row_max;
    }
    total / sample_count
}

fn chunked_posterior_value(matrix: &[Vec<f64>], chunk_size: usize) -> f64 {
    let mut chunk_sum = 0.0;
    let mut chunk_count = 0usize;

    for chunk in matrix.chunks(chunk_size) {
        let means = strategy_means(chunk);
        let best = means.into_iter().fold(f64::NEG_INFINITY, f64::max);
        chunk_sum += best;
        chunk_count += 1;
    }

    chunk_sum / chunk_count as f64
}

/// Deterministic EVSI summary contract built from fixed trial-design inputs.
pub fn evsi_contract(
    analysis_id: impl Into<String>,
    trial_design: &TrialDesign,
    current_net_benefit: &ValueArray,
    method: Option<String>,
) -> Result<AnalysisEnvelope<EvsiSummary>, DomainError> {
    let analysis_id = analysis_id.into();
    if analysis_id.trim().is_empty() {
        return Err(DomainError::EmptyField("analysis_id"));
    }

    let sample_size = summarize_trial_design(trial_design)?;
    let (sample_count, strategy_count) = validate_value_array(current_net_benefit)?;
    let method_name = method.unwrap_or_else(|| DEFAULT_METHOD.to_string());

    let chunk_size = sample_size.clamp(1, sample_count);
    let expected_current_value = strategy_means(&current_net_benefit.net_benefit)
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let expected_perfect_information = row_max_mean(&current_net_benefit.net_benefit);
    let expected_sample_value =
        chunked_posterior_value(&current_net_benefit.net_benefit, chunk_size);
    let evsi = (expected_sample_value - expected_current_value).max(0.0);

    let mut method_metadata = MethodMetadata::new(
        ANALYSIS_TYPE,
        METHOD_FAMILY,
        MethodMaturity::Approximate,
        ApproximationStatus::Approximate,
    )?;
    method_metadata.analysis_id = Some(analysis_id.clone());
    method_metadata.decision_problem_id = Some(trial_design.trial_design_id.clone());
    method_metadata.decision_context = Some(DECISION_CONTEXT.to_string());
    method_metadata.notes.push(
        "Deterministic strided-batch EVSI kernel derived from sample net-benefit rows.".to_string(),
    );

    let mut diagnostics = Diagnostics::new(analysis_id.clone(), DiagnosticStatus::Ok)?;
    diagnostics.backend = None;

    let mut reporting =
        Reporting::cheers_voi(ANALYSIS_TYPE, METHOD_FAMILY, MethodMaturity::Approximate)?;
    reporting.analysis_id = Some(analysis_id.clone());
    reporting.decision_problem_id = Some(trial_design.trial_design_id.clone());
    reporting.decision_context = Some(DECISION_CONTEXT.to_string());
    reporting.estimator = Some(method_name.clone());
    reporting.provenance.insert(
        "trial_design_id".to_string(),
        trial_design.trial_design_id.clone(),
    );
    reporting.provenance.insert(
        "value_array_id".to_string(),
        current_net_benefit.value_array_id.clone(),
    );
    reporting
        .provenance
        .insert("sample_count".to_string(), sample_count.to_string());
    reporting
        .provenance
        .insert("strategy_count".to_string(), strategy_count.to_string());
    reporting.reproducibility.insert(
        "chunking_rule".to_string(),
        "contiguous_chunks_by_trial_sample_size".to_string(),
    );
    reporting
        .reproducibility
        .insert("method".to_string(), method_name.clone());
    reporting.reproducibility.insert(
        "sample_size_rule".to_string(),
        "sum_trial_arm_sample_sizes".to_string(),
    );
    reporting
        .diagnostics
        .insert("sample_count".to_string(), sample_count.to_string());
    reporting
        .diagnostics
        .insert("strategy_count".to_string(), strategy_count.to_string());
    reporting
        .diagnostics
        .insert("sample_size".to_string(), sample_size.to_string());
    reporting
        .diagnostics
        .insert("chunk_size".to_string(), chunk_size.to_string());
    reporting
        .diagnostics
        .insert("status".to_string(), "ok".to_string());
    validate_reporting_payload(&reporting)?;

    let result = EvsiSummary {
        evsi,
        trial_design_id: trial_design.trial_design_id.clone(),
        sample_size,
        expected_current_value,
        expected_sample_value,
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
    envelope.decision_problem_id = Some(trial_design.trial_design_id.clone());
    envelope.method = Some(method_name);
    Ok(envelope)
}
