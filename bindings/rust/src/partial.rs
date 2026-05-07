//! Deterministic summary-family VOI contract helpers.

use serde::{Deserialize, Serialize};

const CONTRACT_VERSION: &str = "rust-core-summary-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartialError {
    EmptyInput(&'static str),
    RaggedRows {
        expected: usize,
        actual: usize,
        row_index: usize,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
        field: &'static str,
    },
    NonFiniteInput(&'static str),
    InvalidConfidenceLevel,
    InvalidStrategyCount,
    InvalidThresholdCount,
}

impl PartialError {
    pub fn message(&self) -> &'static str {
        match self {
            Self::EmptyInput(_) => "input must not be empty",
            Self::RaggedRows { .. } => "rows must have a consistent width",
            Self::LengthMismatch { .. } => "input lengths must match",
            Self::NonFiniteInput(_) => "inputs must contain only finite values",
            Self::InvalidConfidenceLevel => "confidence_level must be between 0 and 1",
            Self::InvalidStrategyCount => "at least two strategies are required",
            Self::InvalidThresholdCount => "at least one threshold is required",
        }
    }
}

impl core::fmt::Display for PartialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for PartialError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryReportingEnvelope {
    pub contract_version: String,
    pub method: String,
    pub deterministic: bool,
    pub status: String,
    pub policy: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DominanceDiagnostics {
    pub strategy_count: usize,
    pub frontier_size: usize,
    pub strong_dominated_count: usize,
    pub extended_dominated_count: usize,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DominanceResult {
    pub strategy_names: Vec<String>,
    pub costs: Vec<f64>,
    pub effects: Vec<f64>,
    pub frontier_indices: Vec<usize>,
    pub strongly_dominated_indices: Vec<usize>,
    pub extended_dominated_indices: Vec<usize>,
    pub status: Vec<String>,
    pub incremental_costs: Vec<f64>,
    pub incremental_effects: Vec<f64>,
    pub icers: Vec<f64>,
    pub diagnostics: DominanceDiagnostics,
    pub reporting: SummaryReportingEnvelope,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeterogeneityDiagnostics {
    pub sample_count: usize,
    pub strategy_count: usize,
    pub subgroup_count: usize,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeterogeneityResult {
    pub value: f64,
    pub subgroup_labels: Vec<String>,
    pub subgroup_weights: Vec<f64>,
    pub subgroup_optimal_strategy_indices: Vec<usize>,
    pub subgroup_optimal_strategy_names: Vec<String>,
    pub subgroup_expected_net_benefits: Vec<f64>,
    pub overall_optimal_strategy_index: usize,
    pub overall_optimal_strategy_name: String,
    pub overall_expected_net_benefit: f64,
    pub diagnostics: HeterogeneityDiagnostics,
    pub reporting: SummaryReportingEnvelope,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CeafDiagnostics {
    pub sample_count: usize,
    pub strategy_count: usize,
    pub threshold_count: usize,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CeafResult {
    pub wtp_thresholds: Vec<f64>,
    pub optimal_strategy_indices: Vec<usize>,
    pub optimal_strategy_names: Vec<String>,
    pub acceptability_probabilities: Vec<f64>,
    pub probability_lower: Vec<f64>,
    pub probability_upper: Vec<f64>,
    pub expected_net_benefit: Vec<f64>,
    pub diagnostics: CeafDiagnostics,
    pub reporting: SummaryReportingEnvelope,
}

type FrontierSeries = (Vec<f64>, Vec<f64>, Vec<f64>);

fn reporting(method: &str, policy: &str) -> SummaryReportingEnvelope {
    SummaryReportingEnvelope {
        contract_version: CONTRACT_VERSION.to_string(),
        method: method.to_string(),
        deterministic: true,
        status: "complete".to_string(),
        policy: policy.to_string(),
    }
}

fn ensure_finite(values: &[f64], field: &'static str) -> Result<(), PartialError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PartialError::NonFiniteInput(field));
    }
    Ok(())
}

fn ensure_matrix(matrix: &[Vec<f64>], field: &'static str) -> Result<(usize, usize), PartialError> {
    if matrix.is_empty() {
        return Err(PartialError::EmptyInput(field));
    }
    let width = matrix[0].len();
    if width == 0 {
        return Err(PartialError::EmptyInput(field));
    }
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != width {
            return Err(PartialError::RaggedRows {
                expected: width,
                actual: row.len(),
                row_index,
            });
        }
        ensure_finite(row, field)?;
    }
    Ok((matrix.len(), width))
}

fn ensure_cube(
    cube: &[Vec<Vec<f64>>],
    field: &'static str,
) -> Result<(usize, usize, usize), PartialError> {
    if cube.is_empty() {
        return Err(PartialError::EmptyInput(field));
    }
    let strategy_count = cube[0].len();
    if strategy_count == 0 {
        return Err(PartialError::EmptyInput(field));
    }
    let threshold_count = cube[0][0].len();
    if threshold_count == 0 {
        return Err(PartialError::EmptyInput(field));
    }
    for (sample_index, sample) in cube.iter().enumerate() {
        if sample.len() != strategy_count {
            return Err(PartialError::RaggedRows {
                expected: strategy_count,
                actual: sample.len(),
                row_index: sample_index,
            });
        }
        for (strategy_index, strategy_slice) in sample.iter().enumerate() {
            if strategy_slice.len() != threshold_count {
                return Err(PartialError::RaggedRows {
                    expected: threshold_count,
                    actual: strategy_slice.len(),
                    row_index: strategy_index,
                });
            }
            ensure_finite(strategy_slice, field)?;
        }
    }
    Ok((cube.len(), strategy_count, threshold_count))
}

fn validate_labels(
    labels: &[String],
    expected: usize,
    field: &'static str,
) -> Result<(), PartialError> {
    if labels.len() != expected {
        return Err(PartialError::LengthMismatch {
            expected,
            actual: labels.len(),
            field,
        });
    }
    if labels.iter().any(|label| label.trim().is_empty()) {
        return Err(PartialError::EmptyInput(field));
    }
    Ok(())
}

/// Return the expected strategy frontier and ICERs for deterministic summary analysis.
pub fn calculate_dominance(
    costs: &[f64],
    effects: &[f64],
    strategy_names: Option<&[String]>,
) -> Result<DominanceResult, PartialError> {
    let strategy_count = costs.len();
    if strategy_count < 2 {
        return Err(PartialError::InvalidStrategyCount);
    }
    if effects.len() != strategy_count {
        return Err(PartialError::LengthMismatch {
            expected: strategy_count,
            actual: effects.len(),
            field: "effects",
        });
    }
    ensure_finite(costs, "costs")?;
    ensure_finite(effects, "effects")?;

    let strategy_names = strategy_names
        .map(|names| {
            if names.len() != strategy_count {
                return Err(PartialError::LengthMismatch {
                    expected: strategy_count,
                    actual: names.len(),
                    field: "strategy_names",
                });
            }
            if names.iter().any(|name| name.trim().is_empty()) {
                return Err(PartialError::EmptyInput("strategy_names"));
            }
            Ok(names.to_vec())
        })
        .transpose()?
        .unwrap_or_else(|| {
            (0..strategy_count)
                .map(|idx| format!("Strategy {idx}"))
                .collect()
        });

    let strong = calculate_strong_dominance(costs, effects)?;
    let frontier = cost_effectiveness_frontier(costs, effects)?;
    let extended = calculate_extended_dominance(costs, effects)?;
    let (incremental_costs, incremental_effects, icers) =
        calculate_icers(costs, effects, Some(&frontier))?;
    let frontier_size = frontier.len();
    let strong_dominated_count = strong.len();
    let extended_dominated_count = extended.len();

    let mut status = vec![String::from("frontier"); strategy_count];
    for idx in &strong {
        status[*idx] = String::from("strongly_dominated");
    }
    for idx in &extended {
        status[*idx] = String::from("extended_dominated");
    }

    Ok(DominanceResult {
        strategy_names,
        costs: costs.to_vec(),
        effects: effects.to_vec(),
        frontier_indices: frontier,
        strongly_dominated_indices: strong,
        extended_dominated_indices: extended,
        status,
        incremental_costs,
        incremental_effects,
        icers,
        diagnostics: DominanceDiagnostics {
            strategy_count,
            frontier_size,
            strong_dominated_count,
            extended_dominated_count,
            finite: true,
        },
        reporting: reporting("calculate_dominance", "deterministic_frontier"),
    })
}

/// Return the strong-dominance indices for deterministic summary analysis.
pub fn calculate_strong_dominance(
    costs: &[f64],
    effects: &[f64],
) -> Result<Vec<usize>, PartialError> {
    let strategy_count = costs.len();
    if strategy_count < 2 {
        return Err(PartialError::InvalidStrategyCount);
    }
    if effects.len() != strategy_count {
        return Err(PartialError::LengthMismatch {
            expected: strategy_count,
            actual: effects.len(),
            field: "effects",
        });
    }
    ensure_finite(costs, "costs")?;
    ensure_finite(effects, "effects")?;

    let mut dominated = Vec::new();
    for idx in 0..strategy_count {
        for other in 0..strategy_count {
            if idx == other {
                continue;
            }
            let no_more_costly = costs[other] <= costs[idx];
            let no_less_effective = effects[other] >= effects[idx];
            let strict_improvement = costs[other] < costs[idx] || effects[other] > effects[idx];
            if no_more_costly && no_less_effective && strict_improvement {
                dominated.push(idx);
                break;
            }
        }
    }
    dominated.sort_unstable();
    Ok(dominated)
}

/// Return the ordered frontier indices for deterministic summary analysis.
pub fn cost_effectiveness_frontier(
    costs: &[f64],
    effects: &[f64],
) -> Result<Vec<usize>, PartialError> {
    let strategy_count = costs.len();
    if strategy_count < 2 {
        return Err(PartialError::InvalidStrategyCount);
    }
    if effects.len() != strategy_count {
        return Err(PartialError::LengthMismatch {
            expected: strategy_count,
            actual: effects.len(),
            field: "effects",
        });
    }
    ensure_finite(costs, "costs")?;
    ensure_finite(effects, "effects")?;

    let strong = calculate_strong_dominance(costs, effects)?;
    let mut candidates: Vec<usize> = (0..strategy_count)
        .filter(|idx| !strong.contains(idx))
        .collect();
    candidates.sort_by(|lhs, rhs| {
        effects[*lhs]
            .total_cmp(&effects[*rhs])
            .then(costs[*lhs].total_cmp(&costs[*rhs]))
    });

    let mut frontier = Vec::new();
    for idx in candidates {
        if let Some(prev) = frontier.last() {
            let previous_index = *prev;
            let effect_gap: f64 = effects[idx] - effects[previous_index];
            if effect_gap.abs() <= f64::EPSILON {
                if costs[idx] < costs[previous_index] {
                    frontier.pop();
                    frontier.push(idx);
                }
                continue;
            }
        }
        frontier.push(idx);
    }

    let mut changed = true;
    while changed && frontier.len() >= 3 {
        changed = false;
        for position in 1..frontier.len() - 1 {
            let prev_idx = frontier[position - 1];
            let curr_idx = frontier[position];
            let next_idx = frontier[position + 1];
            let prev_icer = incremental_icer(
                costs[prev_idx],
                effects[prev_idx],
                costs[curr_idx],
                effects[curr_idx],
            );
            let next_icer = incremental_icer(
                costs[curr_idx],
                effects[curr_idx],
                costs[next_idx],
                effects[next_idx],
            );
            if next_icer <= prev_icer {
                frontier.remove(position);
                changed = true;
                break;
            }
        }
    }

    Ok(frontier)
}

/// Return the extended-dominance indices for deterministic summary analysis.
pub fn calculate_extended_dominance(
    costs: &[f64],
    effects: &[f64],
) -> Result<Vec<usize>, PartialError> {
    let strong = calculate_strong_dominance(costs, effects)?;
    let frontier = cost_effectiveness_frontier(costs, effects)?;
    Ok((0..costs.len())
        .filter(|idx| !strong.contains(idx) && !frontier.contains(idx))
        .collect())
}

/// Return incremental costs, effects, and ICERs for the frontier.
pub fn calculate_icers(
    costs: &[f64],
    effects: &[f64],
    frontier_indices: Option<&[usize]>,
) -> Result<FrontierSeries, PartialError> {
    let strategy_count = costs.len();
    if strategy_count < 2 {
        return Err(PartialError::InvalidStrategyCount);
    }
    if effects.len() != strategy_count {
        return Err(PartialError::LengthMismatch {
            expected: strategy_count,
            actual: effects.len(),
            field: "effects",
        });
    }
    ensure_finite(costs, "costs")?;
    ensure_finite(effects, "effects")?;

    let frontier = if let Some(frontier_indices) = frontier_indices {
        frontier_indices.to_vec()
    } else {
        cost_effectiveness_frontier(costs, effects)?
    };

    if frontier.len() < 2 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let mut incremental_costs = Vec::with_capacity(frontier.len() - 1);
    let mut incremental_effects = Vec::with_capacity(frontier.len() - 1);
    let mut icers = Vec::with_capacity(frontier.len() - 1);
    for window in frontier.windows(2) {
        let prev_idx = window[0];
        let next_idx = window[1];
        let inc_cost = costs[next_idx] - costs[prev_idx];
        let inc_effect = effects[next_idx] - effects[prev_idx];
        incremental_costs.push(inc_cost);
        incremental_effects.push(inc_effect);
        icers.push(if inc_effect > 0.0 {
            inc_cost / inc_effect
        } else {
            f64::INFINITY
        });
    }
    Ok((incremental_costs, incremental_effects, icers))
}

/// Calculate the value of tailoring decisions to subgroups.
pub fn value_of_heterogeneity(
    net_benefits: &[Vec<f64>],
    subgroups: &[String],
    strategy_names: Option<&[String]>,
) -> Result<HeterogeneityResult, PartialError> {
    let (sample_count, strategy_count) = ensure_matrix(net_benefits, "net_benefits")?;
    validate_labels(subgroups, sample_count, "subgroups")?;

    let strategy_names = strategy_names
        .map(|names| {
            if names.len() != strategy_count {
                return Err(PartialError::LengthMismatch {
                    expected: strategy_count,
                    actual: names.len(),
                    field: "strategy_names",
                });
            }
            if names.iter().any(|name| name.trim().is_empty()) {
                return Err(PartialError::EmptyInput("strategy_names"));
            }
            Ok(names.to_vec())
        })
        .transpose()?
        .unwrap_or_else(|| {
            (0..strategy_count)
                .map(|idx| format!("Strategy {idx}"))
                .collect()
        });

    let mut labels = subgroups.to_vec();
    labels.sort();
    labels.dedup();

    let mut subgroup_weights = Vec::with_capacity(labels.len());
    let mut subgroup_optimal_strategy_indices = Vec::with_capacity(labels.len());
    let mut subgroup_expected_net_benefits = Vec::with_capacity(labels.len());

    for label in &labels {
        let mut counts = 0usize;
        let mut subgroup_mean = vec![0.0; strategy_count];
        for (row, subgroup) in net_benefits.iter().zip(subgroups.iter()) {
            if subgroup == label {
                counts += 1;
                for (index, value) in row.iter().enumerate() {
                    subgroup_mean[index] += value;
                }
            }
        }
        if counts == 0 {
            return Err(PartialError::EmptyInput("subgroups"));
        }
        for value in &mut subgroup_mean {
            *value /= counts as f64;
        }
        let optimal_idx = subgroup_mean
            .iter()
            .enumerate()
            .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        subgroup_weights.push(counts as f64 / sample_count as f64);
        subgroup_optimal_strategy_indices.push(optimal_idx);
        subgroup_expected_net_benefits.push(subgroup_mean[optimal_idx]);
    }

    let subgroup_specific_enb: f64 = subgroup_weights
        .iter()
        .zip(subgroup_expected_net_benefits.iter())
        .map(|(weight, value)| weight * value)
        .sum();
    let mut overall_mean = vec![0.0; strategy_count];
    for row in net_benefits {
        for (index, value) in row.iter().enumerate() {
            overall_mean[index] += value;
        }
    }
    for value in &mut overall_mean {
        *value /= sample_count as f64;
    }
    let overall_optimal_strategy_index = overall_mean
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let overall_expected_net_benefit = overall_mean[overall_optimal_strategy_index];

    Ok(HeterogeneityResult {
        value: (subgroup_specific_enb - overall_expected_net_benefit).max(0.0),
        subgroup_labels: labels.clone(),
        subgroup_weights,
        subgroup_optimal_strategy_indices: subgroup_optimal_strategy_indices.clone(),
        subgroup_optimal_strategy_names: subgroup_optimal_strategy_indices
            .iter()
            .map(|idx| strategy_names[*idx].clone())
            .collect(),
        subgroup_expected_net_benefits,
        overall_optimal_strategy_index,
        overall_optimal_strategy_name: strategy_names[overall_optimal_strategy_index].clone(),
        overall_expected_net_benefit,
        diagnostics: HeterogeneityDiagnostics {
            sample_count,
            strategy_count,
            subgroup_count: labels.len(),
            finite: true,
        },
        reporting: reporting("value_of_heterogeneity", "subgroup_optimal_expected_value"),
    })
}

/// Calculate a cost-effectiveness acceptability frontier.
pub fn calculate_ceaf(
    net_benefits: &[Vec<Vec<f64>>],
    wtp_thresholds: &[f64],
    strategy_names: Option<&[String]>,
) -> Result<CeafResult, PartialError> {
    let (sample_count, strategy_count, threshold_count) =
        ensure_cube(net_benefits, "net_benefits")?;
    if wtp_thresholds.is_empty() {
        return Err(PartialError::InvalidThresholdCount);
    }
    if wtp_thresholds.len() != threshold_count {
        return Err(PartialError::LengthMismatch {
            expected: threshold_count,
            actual: wtp_thresholds.len(),
            field: "wtp_thresholds",
        });
    }
    ensure_finite(wtp_thresholds, "wtp_thresholds")?;

    let strategy_names = strategy_names
        .map(|names| {
            if names.len() != strategy_count {
                return Err(PartialError::LengthMismatch {
                    expected: strategy_count,
                    actual: names.len(),
                    field: "strategy_names",
                });
            }
            if names.iter().any(|name| name.trim().is_empty()) {
                return Err(PartialError::EmptyInput("strategy_names"));
            }
            Ok(names.to_vec())
        })
        .transpose()?
        .unwrap_or_else(|| {
            (0..strategy_count)
                .map(|idx| format!("Strategy {idx}"))
                .collect()
        });

    let mut optimal_strategy_indices = Vec::with_capacity(threshold_count);
    let mut optimal_strategy_names = Vec::with_capacity(threshold_count);
    let mut acceptability_probabilities = Vec::with_capacity(threshold_count);
    let mut expected_net_benefit = Vec::with_capacity(threshold_count);
    let mut probability_lower = Vec::with_capacity(threshold_count);
    let mut probability_upper = Vec::with_capacity(threshold_count);

    for threshold_index in 0..threshold_count {
        let mut mean_nb = vec![0.0; strategy_count];
        for sample in net_benefits {
            for (strategy_index, strategy_slice) in sample.iter().enumerate() {
                mean_nb[strategy_index] += strategy_slice[threshold_index];
            }
        }
        for value in &mut mean_nb {
            *value /= sample_count as f64;
        }

        let optimal_strategy_index = mean_nb
            .iter()
            .enumerate()
            .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
            .map(|(index, _)| index)
            .unwrap_or(0);

        let mut sample_optimal_count = 0usize;
        for sample in net_benefits {
            let mut sample_best_index = 0usize;
            let mut sample_best_value = f64::NEG_INFINITY;
            for (strategy_index, strategy_slice) in sample.iter().enumerate() {
                let value = strategy_slice[threshold_index];
                if value > sample_best_value {
                    sample_best_value = value;
                    sample_best_index = strategy_index;
                }
            }
            if sample_best_index == optimal_strategy_index {
                sample_optimal_count += 1;
            }
        }

        let probability = sample_optimal_count as f64 / sample_count as f64;
        let standard_error = (probability * (1.0 - probability) / sample_count as f64).sqrt();
        let z_value = 1.959_963_984_540_054_f64;

        optimal_strategy_indices.push(optimal_strategy_index);
        optimal_strategy_names.push(strategy_names[optimal_strategy_index].clone());
        acceptability_probabilities.push(probability);
        expected_net_benefit.push(mean_nb[optimal_strategy_index]);
        probability_lower.push((probability - z_value * standard_error).clamp(0.0, 1.0));
        probability_upper.push((probability + z_value * standard_error).clamp(0.0, 1.0));
    }

    Ok(CeafResult {
        wtp_thresholds: wtp_thresholds.to_vec(),
        optimal_strategy_indices,
        optimal_strategy_names,
        acceptability_probabilities,
        probability_lower,
        probability_upper,
        expected_net_benefit,
        diagnostics: CeafDiagnostics {
            sample_count,
            strategy_count,
            threshold_count,
            finite: true,
        },
        reporting: reporting("calculate_ceaf", "frontier_probability"),
    })
}

fn incremental_icer(prev_cost: f64, prev_effect: f64, next_cost: f64, next_effect: f64) -> f64 {
    let incremental_effect = next_effect - prev_effect;
    if incremental_effect <= 0.0 {
        return f64::INFINITY;
    }
    (next_cost - prev_cost) / incremental_effect
}
