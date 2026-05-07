//! Seed-driven stochastic EVSI contracts for voiage-core.

use std::collections::BTreeSet;

use crate::domain::{
    validate_reporting_payload, AnalysisEnvelope, ApproximationStatus, DiagnosticStatus,
    Diagnostics, DomainError, EvsiSummary, MethodMaturity, MethodMetadata, Reporting, TrialDesign,
    ValueArray,
};

const ANALYSIS_TYPE: &str = "evsi";
const METHOD_FAMILY: &str = "sample-information";
const DEFAULT_METHOD: &str = "seeded-bootstrap-summary";
const DECISION_CONTEXT: &str = "trial-design-sample-bootstrap";
const MIX64: u64 = 0x9E37_79B9_7F4A_7C15;

fn validate_trial_design(trial_design: &TrialDesign) -> Result<usize, DomainError> {
    if trial_design.trial_design_id.trim().is_empty() {
        return Err(DomainError::EmptyField("trial_design_id"));
    }
    if trial_design.arms.is_empty() {
        return Err(DomainError::EmptyCollection("arms"));
    }

    let mut seen_arm_ids = BTreeSet::new();
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
        if !seen_arm_ids.insert(arm.arm_id.clone()) {
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

    let mut seen_names = BTreeSet::new();
    for name in &value_array.strategy_names {
        if !seen_names.insert(name.clone()) {
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

fn next_index(state: &mut u64, sample_count: usize) -> usize {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let mixed = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    (mixed as usize) % sample_count
}

fn bootstrap_expected_value(
    matrix: &[Vec<f64>],
    draw_count: usize,
    resample_count: usize,
    seed: u64,
) -> f64 {
    let strategy_count = matrix[0].len();
    let mut bootstrap_total = 0.0;

    for resample_index in 0..resample_count {
        let mut totals = vec![0.0; strategy_count];
        let mut state = seed ^ ((resample_index as u64 + 1).wrapping_mul(MIX64));
        for _ in 0..draw_count {
            let index = next_index(&mut state, matrix.len());
            for (strategy_index, value) in matrix[index].iter().enumerate() {
                totals[strategy_index] += value;
            }
        }
        let best = totals
            .into_iter()
            .map(|sum| sum / draw_count as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        bootstrap_total += best;
    }

    bootstrap_total / resample_count as f64
}

/// Seed-driven two-loop EVSI contract built from sample net-benefit rows.
pub fn evsi_stochastic_contract(
    analysis_id: impl Into<String>,
    trial_design: &TrialDesign,
    current_net_benefit: &ValueArray,
    seed: u64,
    method: Option<String>,
) -> Result<AnalysisEnvelope<EvsiSummary>, DomainError> {
    let analysis_id = analysis_id.into();
    if analysis_id.trim().is_empty() {
        return Err(DomainError::EmptyField("analysis_id"));
    }

    let sample_size = validate_trial_design(trial_design)?;
    let (sample_count, strategy_count) = validate_value_array(current_net_benefit)?;
    let method_name = method.unwrap_or_else(|| DEFAULT_METHOD.to_string());

    let draw_count = sample_size.clamp(1, sample_count);
    let resample_count = sample_count.max(1);
    let expected_current_value = strategy_means(&current_net_benefit.net_benefit)
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let expected_perfect_information = row_max_mean(&current_net_benefit.net_benefit);
    let expected_sample_value = bootstrap_expected_value(
        &current_net_benefit.net_benefit,
        draw_count,
        resample_count,
        seed,
    );
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
        "Seeded bootstrap EVSI kernel with an outer resample loop and an inner draw loop."
            .to_string(),
    );

    let mut diagnostics = Diagnostics::new(analysis_id.clone(), DiagnosticStatus::Approximate)?;
    diagnostics.backend = Some("rust-core".to_string());
    diagnostics.approximation_caveats.push(
        "Seeded bootstrap estimate; outer loop can be parallelized independently.".to_string(),
    );

    let mut reporting =
        Reporting::cheers_voi(ANALYSIS_TYPE, METHOD_FAMILY, MethodMaturity::Approximate)?;
    reporting.analysis_id = Some(analysis_id.clone());
    reporting.decision_problem_id = Some(trial_design.trial_design_id.clone());
    reporting.decision_context = Some(DECISION_CONTEXT.to_string());
    reporting.estimator = Some(method_name.clone());
    reporting.seed = Some(seed);
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
        "draw_count_rule".to_string(),
        "min(sample_size, sample_count)".to_string(),
    );
    reporting.reproducibility.insert(
        "resample_count_rule".to_string(),
        "sample_count".to_string(),
    );
    reporting
        .reproducibility
        .insert("seed".to_string(), seed.to_string());
    reporting
        .reproducibility
        .insert("method".to_string(), method_name.clone());
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
        .insert("draw_count".to_string(), draw_count.to_string());
    reporting
        .diagnostics
        .insert("resample_count".to_string(), resample_count.to_string());
    reporting
        .diagnostics
        .insert("status".to_string(), "approximate".to_string());
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
