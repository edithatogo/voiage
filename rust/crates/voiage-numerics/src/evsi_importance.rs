use voiage_domain::{EvsiDiagnostics, SampleMatrix};

use crate::NumericalInputError;

/// Result for importance-sampling EVSI with explicit ESS and Pareto-k diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiImportanceResult {
    /// Stable estimator identifier.
    pub estimator: &'static str,
    /// Version of the importance-sampling EVSI contract.
    pub contract_version: u32,
    /// Expected value under current information.
    pub expected_current_value: f64,
    /// Expected value after importance-weighted study updates.
    pub expected_sample_value: f64,
    /// Unclamped raw EVSI (`expected_sample_value - expected_current_value`).
    pub raw_evsi: f64,
    /// Non-negative EVSI (`raw_evsi.max(0.0)`).
    pub evsi: f64,
    /// Minimum effective sample size observed across simulated datasets.
    pub min_ess: f64,
    /// Mean effective sample size across simulated datasets.
    pub mean_ess: f64,
    /// Maximum Pareto-k tail diagnostic observed across simulated datasets.
    pub max_khat: f64,
    /// Number of simulated outer datasets.
    pub dataset_count: usize,
    /// Number of prior draws.
    pub sample_count: usize,
    /// Coherent diagnostic summary.
    pub diagnostics: EvsiDiagnostics,
}

fn compute_row_weights_and_diagnostics(
    log_weights: &[f64],
) -> Result<(Vec<f64>, f64, f64), NumericalInputError> {
    if log_weights.iter().any(|w| w.is_nan() || *w == f64::INFINITY) {
        return Err(NumericalInputError::invalid(
            "log_weights",
            "importance log-weights must not be NaN or +inf",
        ));
    }
    let max_lw = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return Err(NumericalInputError::invalid(
            "log_weights",
            "importance log-weights are all -inf for at least one dataset",
        ));
    }

    let raw: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
    let sum_raw: f64 = raw.iter().sum();
    if !sum_raw.is_finite() || sum_raw <= 0.0 {
        return Err(NumericalInputError::invalid(
            "log_weights",
            "normalized importance weight sum is not positive and finite",
        ));
    }
    let weights: Vec<f64> = raw.into_iter().map(|w| w / sum_raw).collect();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    if !sum_sq.is_finite() || sum_sq <= 0.0 {
        return Err(NumericalInputError::invalid(
            "log_weights",
            "importance weight squared sum is invalid",
        ));
    }
    let ess = 1.0 / sum_sq;

    // Hill / empirical Pareto tail index on upper tail of sorted weights.
    let mut sorted = weights.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let tail_len = (n / 5).max(3).min(n - 1);
    let threshold = sorted[n - 1 - tail_len];
    let khat = if threshold <= 0.0 {
        f64::INFINITY
    } else {
        let sum_log_excess: f64 = sorted[n - tail_len..]
            .iter()
            .map(|&w| (w / threshold).ln())
            .sum();
        let tail_f64 = f64::from(u32::try_from(tail_len).map_err(|_| {
            NumericalInputError::invalid("log_weights", "sample count exceeds u32 range")
        })?);
        sum_log_excess / tail_f64
    };

    Ok((weights, ess, khat))
}

/// Evaluates importance-sampling EVSI and returns raw diagnostics without gating.
///
/// Use [`evsi_importance_sampling`] for fail-closed production evaluation.
///
/// # Errors
///
/// Returns [`NumericalInputError`] on shape mismatches, non-finite inputs, or
/// degenerate all-zero weights.
pub fn evsi_importance_sampling_raw(
    net_benefit: &SampleMatrix,
    log_weights: &SampleMatrix,
) -> Result<EvsiImportanceResult, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let [dataset_count, weight_cols] = log_weights.shape();
    if sample_count < 2 {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "importance sampling requires at least 2 prior samples",
        ));
    }
    if weight_cols != sample_count {
        return Err(NumericalInputError::dimension(
            "log_weights",
            sample_count,
            weight_cols,
            "log_weights columns must match net_benefit rows",
        ));
    }

    let sample_count_f64 = f64::from(u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid("net_benefit", "sample count exceeds u32 range")
    })?);
    let dataset_count_f64 = f64::from(u32::try_from(dataset_count).map_err(|_| {
        NumericalInputError::invalid("log_weights", "dataset count exceeds u32 range")
    })?);

    let mut prior_means = vec![0.0; strategy_count];
    for row in net_benefit.rows() {
        for (k, &val) in row.iter().enumerate() {
            prior_means[k] += val / sample_count_f64;
        }
    }
    let expected_current_value = prior_means
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut post_study_sum = 0.0;
    let mut min_ess = f64::INFINITY;
    let mut ess_sum = 0.0;
    let mut max_khat = f64::NEG_INFINITY;

    for lw_row in log_weights.rows() {
        let (weights, ess, khat) = compute_row_weights_and_diagnostics(lw_row)?;
        min_ess = min_ess.min(ess);
        ess_sum += ess;
        max_khat = max_khat.max(khat);

        let mut post_means = vec![0.0; strategy_count];
        for (nb_row, &w) in net_benefit.rows().zip(weights.iter()) {
            for (k, &nb) in nb_row.iter().enumerate() {
                post_means[k] += w * nb;
            }
        }
        let best = post_means
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        post_study_sum += best;
    }

    let expected_sample_value = post_study_sum / dataset_count_f64;
    let raw_evsi = expected_sample_value - expected_current_value;
    let evsi = raw_evsi.max(0.0);
    let mean_ess = ess_sum / dataset_count_f64;
    let khat_opt = if max_khat.is_finite() {
        Some(max_khat)
    } else {
        None
    };

    let diagnostics = EvsiDiagnostics {
        ess: Some(min_ess),
        khat: khat_opt,
        decision_error: None,
        monte_carlo_se: None,
        bias_estimate: None,
        rmse_estimate: None,
        coverage: None,
    };

    Ok(EvsiImportanceResult {
        estimator: "importance_sampling",
        contract_version: 1,
        expected_current_value,
        expected_sample_value,
        raw_evsi,
        evsi,
        min_ess,
        mean_ess,
        max_khat,
        dataset_count,
        sample_count,
        diagnostics,
    })
}

/// Computes importance-sampling EVSI and fails closed on ESS or Pareto-k degeneracy.
///
/// # Errors
///
/// Returns [`NumericalInputError`] if thresholds are invalid, inputs fail validation,
/// `min_ess < min_ess_threshold`, or `max_khat > max_khat_threshold` (or non-finite).
pub fn evsi_importance_sampling(
    net_benefit: &SampleMatrix,
    log_weights: &SampleMatrix,
    min_ess_threshold: f64,
    max_khat_threshold: f64,
) -> Result<EvsiImportanceResult, NumericalInputError> {
    if !min_ess_threshold.is_finite() || min_ess_threshold < 1.0 {
        return Err(NumericalInputError::invalid(
            "min_ess_threshold",
            "min_ess_threshold must be finite and >= 1.0",
        ));
    }
    if !max_khat_threshold.is_finite() || max_khat_threshold <= 0.0 {
        return Err(NumericalInputError::invalid(
            "max_khat_threshold",
            "max_khat_threshold must be finite and > 0.0",
        ));
    }
    let result = evsi_importance_sampling_raw(net_benefit, log_weights)?;
    if result.min_ess < min_ess_threshold {
        return Err(NumericalInputError::invalid(
            "importance_weights",
            "importance-sampling ESS degeneracy: observed min_ess below threshold",
        ));
    }
    if !result.max_khat.is_finite() || result.max_khat > max_khat_threshold {
        return Err(NumericalInputError::invalid(
            "importance_weights",
            "importance-sampling Pareto-k degeneracy: observed max_khat exceeds threshold",
        ));
    }
    Ok(result)
}
