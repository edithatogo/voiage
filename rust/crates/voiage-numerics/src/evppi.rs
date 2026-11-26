use voiage_domain::SampleMatrix;

use crate::NumericalInputError;

/// Computes regression-based EVPPI from net-benefit and parameter samples.
///
/// Each strategy is fitted independently using ordinary least squares with an
/// intercept. The fitted conditional net benefits are evaluated at every
/// supplied parameter sample, matching the stable Python linear estimator.
///
/// # Errors
///
/// Returns a dimension error when net-benefit and parameter samples do not
/// contain the same number of rows.
#[allow(clippy::too_many_lines)]
pub fn evppi(
    net_benefit: &SampleMatrix,
    parameter_samples: &SampleMatrix,
) -> Result<f64, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let [parameter_sample_count, parameter_count] = parameter_samples.shape();
    if sample_count != parameter_sample_count {
        return Err(NumericalInputError::dimension(
            "parameter_samples",
            sample_count,
            parameter_sample_count,
            "net-benefit and parameter sample counts must match",
        ));
    }

    if strategy_count <= 1 {
        return Ok(0.0);
    }

    let (parameter_means, parameter_scales) = parameter_location_and_scale(parameter_samples)?;
    let coefficient_count = parameter_count + 1;
    let mut normal_matrix = vec![vec![0.0; coefficient_count]; coefficient_count];
    for row in parameter_samples.rows() {
        let design_row = design_row(row, &parameter_means, &parameter_scales);
        for (left, left_value) in design_row.iter().copied().enumerate() {
            for (right, right_value) in design_row.iter().copied().enumerate() {
                let updated = normal_matrix[left][right] + left_value * right_value;
                if !updated.is_finite() {
                    return Err(NumericalInputError::invalid(
                        "parameter_samples",
                        "EVPPI regression design is not finite",
                    ));
                }
                normal_matrix[left][right] = updated;
            }
        }
    }

    let mut conditional_maximum_total = 0.0;
    let mut strategy_totals = vec![0.0; strategy_count];
    let mut fitted_values = vec![vec![0.0; sample_count]; strategy_count];
    for strategy_index in 0..strategy_count {
        let right_hand_side = (0..coefficient_count)
            .map(|coefficient| {
                let sum = parameter_samples
                    .rows()
                    .zip(net_benefit.rows())
                    .map(|(parameters, values)| {
                        design_row_value(
                            parameters,
                            coefficient,
                            &parameter_means,
                            &parameter_scales,
                        ) * values[strategy_index]
                    })
                    .try_fold(0.0, |total, value| {
                        let updated = total + value;
                        updated.is_finite().then_some(updated)
                    })
                    .ok_or_else(|| {
                        NumericalInputError::invalid(
                            "net_benefit",
                            "EVPPI regression response is not finite",
                        )
                    })?;
                Ok(sum)
            })
            .collect::<Result<Vec<f64>, NumericalInputError>>()?;
        let coefficients = solve_singular_system(&normal_matrix, &right_hand_side);
        for (sample_index, parameter_row) in parameter_samples.rows().enumerate() {
            fitted_values[strategy_index][sample_index] =
                design_row(parameter_row, &parameter_means, &parameter_scales)
                    .iter()
                    .zip(coefficients.iter())
                    .map(|(design_value, coefficient)| design_value * coefficient)
                    .sum();
            if !fitted_values[strategy_index][sample_index].is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "EVPPI fitted response is not finite",
                ));
            }
        }
        for value in net_benefit.rows().map(|row| row[strategy_index]) {
            let updated = strategy_totals[strategy_index] + value;
            if !updated.is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "EVPPI current-value total is not finite",
                ));
            }
            strategy_totals[strategy_index] = updated;
        }
    }
    for sample_index in 0..sample_count {
        conditional_maximum_total += fitted_values
            .iter()
            .map(|strategy| strategy[sample_index])
            .fold(f64::NEG_INFINITY, f64::max);
    }

    let divisor = f64::from(u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported numerical range",
        )
    })?);
    let max_expected_net_benefit = strategy_totals
        .into_iter()
        .map(|total| total / divisor)
        .fold(f64::NEG_INFINITY, f64::max);
    let result = (conditional_maximum_total / divisor - max_expected_net_benefit).max(0.0);
    if !result.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "EVPPI result is not finite",
        ));
    }
    Ok(result)
}

fn parameter_location_and_scale(
    parameter_samples: &SampleMatrix,
) -> Result<(Vec<f64>, Vec<f64>), NumericalInputError> {
    let parameter_count = parameter_samples.shape()[1];
    let sample_count = f64::from(u32::try_from(parameter_samples.shape()[0]).map_err(|_| {
        NumericalInputError::invalid(
            "parameter_samples",
            "sample count exceeds the supported numerical range",
        )
    })?);
    let mut means = vec![0.0; parameter_count];
    for row in parameter_samples.rows() {
        for (index, value) in row.iter().copied().enumerate() {
            means[index] += value / sample_count;
        }
    }
    if means.iter().any(|value| !value.is_finite()) {
        return Err(NumericalInputError::invalid(
            "parameter_samples",
            "EVPPI parameter mean is not finite",
        ));
    }
    let mut scales = vec![0.0_f64; parameter_count];
    for row in parameter_samples.rows() {
        for (index, value) in row.iter().copied().enumerate() {
            let centered = value - means[index];
            if !centered.is_finite() {
                return Err(NumericalInputError::invalid(
                    "parameter_samples",
                    "EVPPI parameter centering is not finite",
                ));
            }
            scales[index] = scales[index].max(centered.abs());
        }
    }
    for scale in &mut scales {
        if *scale == 0.0 {
            *scale = 1.0;
        }
    }
    Ok((means, scales))
}

fn design_row(parameters: &[f64], means: &[f64], scales: &[f64]) -> Vec<f64> {
    std::iter::once(1.0)
        .chain(
            parameters
                .iter()
                .zip(means.iter().zip(scales.iter()))
                .map(|(value, (mean, scale))| (value - mean) / scale),
        )
        .collect()
}

fn design_row_value(parameters: &[f64], coefficient: usize, means: &[f64], scales: &[f64]) -> f64 {
    if coefficient == 0 {
        1.0
    } else {
        (parameters[coefficient - 1] - means[coefficient - 1]) / scales[coefficient - 1]
    }
}

fn solve_singular_system(matrix: &[Vec<f64>], right_hand_side: &[f64]) -> Vec<f64> {
    let size = right_hand_side.len();
    let mut augmented = matrix
        .iter()
        .zip(right_hand_side.iter())
        .map(|(row, value)| {
            let mut augmented_row = row.clone();
            augmented_row.push(*value);
            augmented_row
        })
        .collect::<Vec<_>>();
    let mut pivot_row = 0;
    for column in 0..size {
        let Some((best_row, best_value)) = augmented[pivot_row..]
            .iter()
            .enumerate()
            .map(|(offset, row)| (pivot_row + offset, row[column].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
        else {
            break;
        };
        let scale = augmented
            .iter()
            .map(|row| row[column].abs())
            .fold(0.0, f64::max);
        if best_value <= scale * 1.0e-12 {
            continue;
        }
        augmented.swap(pivot_row, best_row);
        let pivot = augmented[pivot_row][column];
        for value in &mut augmented[pivot_row][column..] {
            *value /= pivot;
        }
        for row in 0..size {
            if row == pivot_row {
                continue;
            }
            let factor = augmented[row][column];
            let pivot_values = augmented[pivot_row][column..=size].to_vec();
            for (value, pivot_value) in augmented[row][column..=size]
                .iter_mut()
                .zip(pivot_values.iter())
            {
                *value -= factor * pivot_value;
            }
        }
        pivot_row += 1;
        if pivot_row == size {
            break;
        }
    }
    let mut solution = vec![0.0; size];
    for row in &augmented {
        if let Some((column, value)) = row[..size]
            .iter()
            .enumerate()
            .find(|(_, value)| value.abs() > 1.0e-12)
        {
            solution[column] = row[size] / *value;
        }
    }
    solution
}

const MAX_NONLINEAR_BASIS_WIDTH: usize = 256;
const MAX_SPLINE_KNOTS: usize = 20;
const MARS_ARTIFACT_SCHEMA_V1: &str = "voiage.mars-artifact.v1";

/// Basis family for Rust nonlinear EVPPI regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvppiBasis {
    /// Intercept, standardized linear terms, quadratic squares, and pairwise interactions.
    QuadraticInteraction,
    /// Intercept, standardized linear terms, empirical-quantile hinge splines, and interactions.
    PiecewiseLinearSpline {
        /// Number of empirical-quantile knots evaluated per parameter on each training fold.
        knots_per_parameter: usize,
    },
}

/// Configuration for cross-fitted Rust nonlinear EVPPI evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvppiNonlinearConfig {
    /// Nonlinear basis expansion family.
    pub basis: EvppiBasis,
    /// Number of deterministic modulo cross-fitting folds.
    pub fold_count: usize,
    /// Non-negative L2 penalty applied to non-intercept basis coefficients.
    pub ridge_lambda: f64,
}

/// Diagnostic and decision-quality output from nonlinear EVPPI estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct EvppiNonlinearResult {
    /// Stable nonlinear estimator identifier.
    pub estimator: &'static str,
    /// Numerical contract version.
    pub contract_version: u32,
    /// Full-sample fitted EVPPI estimate (clamped at zero).
    pub evppi_in_sample: f64,
    /// Out-of-fold cross-fitted EVPPI estimate (clamped at zero).
    pub evppi_held_out: f64,
    /// Held-out realized decision gain relative to current best strategy (unclamped).
    pub held_out_realized_gain: f64,
    /// Decision loss from held-out policy selection relative to perfect information.
    pub held_out_decision_loss_vs_perfect: f64,
    /// Full-sample linear OLS EVPPI reference value.
    pub linear_evppi_reference: f64,
    /// Baseline expected net benefit under current information.
    pub expected_current_value: f64,
    /// Expected net benefit under perfect information.
    pub expected_perfect_information: f64,
    /// Held-out coefficient of determination per strategy.
    pub held_out_r2_by_strategy: Vec<f64>,
    /// Held-out root mean squared error per strategy.
    pub held_out_rmse_by_strategy: Vec<f64>,
    /// Number of basis columns (including intercept).
    pub basis_width: usize,
    /// Number of aligned PSA draws evaluated.
    pub sample_count: usize,
    /// Number of decision strategies evaluated.
    pub strategy_count: usize,
    /// Number of parameters of interest evaluated.
    pub parameter_count: usize,
    /// Number of cross-fitting folds evaluated.
    pub fold_count: usize,
    /// Ridge regularization penalty applied to non-intercept weights.
    pub ridge_lambda: f64,
    /// Explicit applicability and diagnostic warnings surfaced by the kernel.
    pub applicability_warnings: Vec<String>,
}

/// Single hinge factor inside a producer-fitted MARS basis term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarsHingeFactor {
    /// Zero-based index into `parameter_samples` columns.
    pub parameter_index: usize,
    /// Threshold knot value in raw parameter units.
    pub knot: f64,
    /// Hinge direction (`1` for `(x - knot)+`, `-1` for `(knot - x)+`).
    pub direction: i8,
}

/// Multiplicative basis term in a producer-fitted MARS surrogate.
#[derive(Clone, Debug, PartialEq)]
pub struct MarsTerm {
    /// Fitted scalar weight multiplying the hinge product.
    pub coefficient: f64,
    /// Non-empty list of hinge factors forming this basis term.
    pub factors: Vec<MarsHingeFactor>,
}

/// Fitted MARS regression expression for one decision strategy.
#[derive(Clone, Debug, PartialEq)]
pub struct MarsStrategyModel {
    /// Additive intercept for the strategy net-benefit surrogate.
    pub intercept: f64,
    /// Basis terms added to `intercept`.
    pub terms: Vec<MarsTerm>,
}

/// Producer-owned MARS fitted artifact replayed by Rust without training inside `voiage`.
#[derive(Clone, Debug, PartialEq)]
pub struct MarsArtifact {
    /// Version tag (`voiage.mars-artifact.v1`).
    pub schema_version: String,
    /// Non-empty untrimmed identifier of the external training producer.
    pub producer: String,
    /// Strategy surrogates aligned with `net_benefit` strategy columns.
    pub strategies: Vec<MarsStrategyModel>,
}

/// EVPPI and surrogate diagnostics from replaying a producer-owned MARS artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct EvppiMarsResult {
    /// Stable MARS replay estimator identifier.
    pub estimator: &'static str,
    /// Numerical contract version.
    pub contract_version: u32,
    /// Replayed surrogate EVPPI (clamped at zero).
    pub evppi: f64,
    /// Realized net-benefit gain from acting on MARS predictions vs current best.
    pub realized_decision_gain: f64,
    /// Decision loss from acting on MARS predictions vs perfect information.
    pub decision_loss_vs_perfect: f64,
    /// Baseline expected net benefit under current information.
    pub expected_current_value: f64,
    /// Expected net benefit under perfect information.
    pub expected_perfect_information: f64,
    /// Coefficient of determination per strategy over the supplied draws.
    pub r2_by_strategy: Vec<f64>,
    /// Root mean squared error per strategy over the supplied draws.
    pub rmse_by_strategy: Vec<f64>,
    /// Number of PSA draws evaluated.
    pub sample_count: usize,
    /// Number of strategies evaluated.
    pub strategy_count: usize,
    /// Number of parameters evaluated.
    pub parameter_count: usize,
}

fn checked_count_f64(
    count: usize,
    field: &'static str,
    message: &'static str,
) -> Result<f64, NumericalInputError> {
    let exact = u32::try_from(count).map_err(|_| NumericalInputError::invalid(field, message))?;
    Ok(f64::from(exact))
}

fn basis_width_for(
    basis: EvppiBasis,
    parameter_count: usize,
) -> Result<usize, NumericalInputError> {
    let interaction_count = if parameter_count <= 8 && parameter_count > 1 {
        parameter_count * (parameter_count - 1) / 2
    } else {
        0
    };
    let width = match basis {
        EvppiBasis::QuadraticInteraction => {
            1 + parameter_count
                + parameter_count
                + parameter_count.saturating_mul(parameter_count.saturating_sub(1)) / 2
        }
        EvppiBasis::PiecewiseLinearSpline {
            knots_per_parameter,
        } => {
            if knots_per_parameter == 0 || knots_per_parameter > MAX_SPLINE_KNOTS {
                return Err(NumericalInputError::invalid(
                    "knots_per_parameter",
                    "knots_per_parameter must be between 1 and 20",
                ));
            }
            1 + parameter_count + parameter_count * knots_per_parameter + interaction_count
        }
    };
    if width > MAX_NONLINEAR_BASIS_WIDTH {
        return Err(NumericalInputError::invalid(
            "parameter_samples",
            "nonlinear EVPPI basis width exceeds supported limit of 256 columns",
        ));
    }
    Ok(width)
}

fn row_at(matrix: &SampleMatrix, row_idx: usize) -> &[f64] {
    matrix.row(row_idx).expect("row index within matrix bounds")
}

fn subset_location_and_scale(
    parameter_samples: &SampleMatrix,
    indices: &[usize],
) -> Result<(Vec<f64>, Vec<f64>, bool), NumericalInputError> {
    let parameter_count = parameter_samples.shape()[1];
    let n_rows = checked_count_f64(
        indices.len(),
        "parameter_samples",
        "training sample count exceeds numerical range",
    )?;
    let mut means: Vec<f64> = vec![0.0; parameter_count];
    for &row_idx in indices {
        let row = row_at(parameter_samples, row_idx);
        for (col, &val) in row.iter().enumerate() {
            means[col] += val / n_rows;
        }
    }
    if means.iter().any(|value: &f64| !value.is_finite()) {
        return Err(NumericalInputError::invalid(
            "parameter_samples",
            "nonlinear EVPPI parameter mean is not finite",
        ));
    }
    let mut scales = vec![0.0_f64; parameter_count];
    for &row_idx in indices {
        let row = row_at(parameter_samples, row_idx);
        for (col, &val) in row.iter().enumerate() {
            let centered = val - means[col];
            if !centered.is_finite() {
                return Err(NumericalInputError::invalid(
                    "parameter_samples",
                    "nonlinear EVPPI parameter centering is not finite",
                ));
            }
            scales[col] = scales[col].max(centered.abs());
        }
    }
    let mut has_constant = false;
    for scale in &mut scales {
        if *scale == 0.0 {
            has_constant = true;
            *scale = 1.0;
        }
    }
    Ok((means, scales, has_constant))
}

fn training_spline_knots(
    parameter_samples: &SampleMatrix,
    train_indices: &[usize],
    means: &[f64],
    scales: &[f64],
    knots_per_parameter: usize,
) -> Vec<Vec<f64>> {
    let parameter_count = parameter_samples.shape()[1];
    let mut all_knots = Vec::with_capacity(parameter_count);
    for col in 0..parameter_count {
        let mut sorted: Vec<f64> = train_indices
            .iter()
            .map(|&row_idx| (row_at(parameter_samples, row_idx)[col] - means[col]) / scales[col])
            .collect();
        sorted.sort_by(f64::total_cmp);
        let max_idx = sorted.len().saturating_sub(1);
        let denom = f64::from(u32::try_from(knots_per_parameter + 1).unwrap_or(u32::MAX));
        let max_pos = f64::from(u32::try_from(max_idx).unwrap_or(u32::MAX));
        let mut col_knots = Vec::with_capacity(knots_per_parameter);
        for m in 1..=knots_per_parameter {
            let quantile = f64::from(u32::try_from(m).unwrap_or(u32::MAX)) / denom;
            let pos = quantile * max_pos;
            let lower = pos.floor();
            let frac = pos - lower;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let lo_idx = (lower as usize).min(max_idx);
            let hi_idx = (lo_idx + 1).min(max_idx);
            let knot = sorted[lo_idx] + frac * (sorted[hi_idx] - sorted[lo_idx]);
            col_knots.push(knot);
        }
        all_knots.push(col_knots);
    }
    all_knots
}

fn eval_nonlinear_design_row(
    row: &[f64],
    means: &[f64],
    scales: &[f64],
    basis: EvppiBasis,
    knots: &[Vec<f64>],
) -> Vec<f64> {
    let p = row.len();
    let z: Vec<f64> = row
        .iter()
        .zip(means.iter().zip(scales.iter()))
        .map(|(val, (mean, scale))| (val - mean) / scale)
        .collect();
    let mut design = Vec::with_capacity(MAX_NONLINEAR_BASIS_WIDTH);
    design.push(1.0);
    design.extend_from_slice(&z);
    match basis {
        EvppiBasis::QuadraticInteraction => {
            for &zj in &z {
                design.push(zj * zj);
            }
            for j in 0..p {
                for k in (j + 1)..p {
                    design.push(z[j] * z[k]);
                }
            }
        }
        EvppiBasis::PiecewiseLinearSpline { .. } => {
            for (j, col_knots) in knots.iter().enumerate() {
                for &knot in col_knots {
                    design.push((z[j] - knot).max(0.0));
                }
            }
            if p <= 8 {
                for j in 0..p {
                    for k in (j + 1)..p {
                        design.push(z[j] * z[k]);
                    }
                }
            }
        }
    }
    design
}

fn fit_and_predict_subset(
    net_benefit: &SampleMatrix,
    parameter_samples: &SampleMatrix,
    train_indices: &[usize],
    eval_indices: &[usize],
    basis: EvppiBasis,
    ridge_lambda: f64,
    basis_width: usize,
) -> Result<Vec<Vec<f64>>, NumericalInputError> {
    let [_, strategy_count] = net_benefit.shape();
    let (means, scales, _) = subset_location_and_scale(parameter_samples, train_indices)?;
    let knots = match basis {
        EvppiBasis::QuadraticInteraction => Vec::new(),
        EvppiBasis::PiecewiseLinearSpline {
            knots_per_parameter,
        } => training_spline_knots(
            parameter_samples,
            train_indices,
            &means,
            &scales,
            knots_per_parameter,
        ),
    };
    let n_train = checked_count_f64(
        train_indices.len(),
        "net_benefit",
        "training fold count exceeds numerical range",
    )?;
    let mut normal_matrix = vec![vec![0.0; basis_width]; basis_width];
    let mut rhs_by_strategy = vec![vec![0.0; basis_width]; strategy_count];
    for &row_idx in train_indices {
        let p_slice = row_at(parameter_samples, row_idx);
        let nb_slice = row_at(net_benefit, row_idx);
        let design = eval_nonlinear_design_row(p_slice, &means, &scales, basis, &knots);
        for left in 0..basis_width {
            let scaled_left = design[left] / n_train;
            for right in 0..basis_width {
                let updated = normal_matrix[left][right] + scaled_left * design[right];
                if !updated.is_finite() {
                    return Err(NumericalInputError::invalid(
                        "parameter_samples",
                        "nonlinear EVPPI normal matrix is not finite",
                    ));
                }
                normal_matrix[left][right] = updated;
            }
            for strategy_idx in 0..strategy_count {
                let updated =
                    rhs_by_strategy[strategy_idx][left] + scaled_left * nb_slice[strategy_idx];
                if !updated.is_finite() {
                    return Err(NumericalInputError::invalid(
                        "net_benefit",
                        "nonlinear EVPPI regression response is not finite",
                    ));
                }
                rhs_by_strategy[strategy_idx][left] = updated;
            }
        }
    }
    for (diag, row) in normal_matrix.iter_mut().enumerate().skip(1) {
        row[diag] += ridge_lambda;
    }

    let coefficients: Vec<Vec<f64>> = rhs_by_strategy
        .iter()
        .map(|rhs| solve_singular_system(&normal_matrix, rhs))
        .collect();
    let mut predictions = vec![vec![0.0; strategy_count]; eval_indices.len()];
    for (out_idx, &row_idx) in eval_indices.iter().enumerate() {
        let p_slice = row_at(parameter_samples, row_idx);
        let design = eval_nonlinear_design_row(p_slice, &means, &scales, basis, &knots);
        for strategy_idx in 0..strategy_count {
            let pred: f64 = design
                .iter()
                .zip(coefficients[strategy_idx].iter())
                .map(|(d, b)| d * b)
                .sum();
            if !pred.is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "nonlinear EVPPI predicted response is not finite",
                ));
            }
            predictions[out_idx][strategy_idx] = pred;
        }
    }
    Ok(predictions)
}

struct DecisionFitSummary {
    evppi: f64,
    realized_gain: f64,
    decision_loss: f64,
    expected_current: f64,
    expected_perfect: f64,
    r2_by_strategy: Vec<f64>,
    rmse_by_strategy: Vec<f64>,
}

fn summarize_decision_and_fit(
    net_benefit: &SampleMatrix,
    predictions: &[Vec<f64>],
) -> Result<DecisionFitSummary, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let divisor = checked_count_f64(
        sample_count,
        "net_benefit",
        "sample count exceeds the supported numerical range",
    )?;
    let mut strategy_totals = vec![0.0; strategy_count];
    let mut surrogate_totals = vec![0.0; strategy_count];
    let mut perfect_total = 0.0;
    let mut conditional_max_total = 0.0;
    let mut realized_total = 0.0;
    let mut ss_res = vec![0.0; strategy_count];

    for (row_idx, pred_row) in predictions.iter().enumerate() {
        let actual_row = row_at(net_benefit, row_idx);
        let mut row_nb_best = f64::NEG_INFINITY;
        let mut pred_best = f64::NEG_INFINITY;
        let mut chosen_strategy = 0;
        for d in 0..strategy_count {
            let actual = actual_row[d];
            let pred = pred_row[d];
            strategy_totals[d] += actual / divisor;
            surrogate_totals[d] += pred / divisor;
            row_nb_best = row_nb_best.max(actual);
            if pred > pred_best {
                pred_best = pred;
                chosen_strategy = d;
            }
            let diff = pred - actual;
            ss_res[d] += diff * diff;
        }
        perfect_total += row_nb_best / divisor;
        conditional_max_total += pred_best / divisor;
        realized_total += actual_row[chosen_strategy] / divisor;
    }

    let expected_current = strategy_totals
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let surrogate_expected_current = surrogate_totals
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut r2_by_strategy = Vec::with_capacity(strategy_count);
    let mut rmse_by_strategy = Vec::with_capacity(strategy_count);
    for d in 0..strategy_count {
        let mean_d = strategy_totals[d];
        let mut ss_tot = 0.0;
        for row_idx in 0..sample_count {
            let dev = row_at(net_benefit, row_idx)[d] - mean_d;
            ss_tot += dev * dev;
        }
        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res[d] / ss_tot
        } else {
            0.0
        };
        let rmse = (ss_res[d] / divisor).sqrt();
        if !r2.is_finite() || !rmse.is_finite() {
            return Err(NumericalInputError::invalid(
                "net_benefit",
                "surrogate fit diagnostics are not finite",
            ));
        }
        r2_by_strategy.push(r2);
        rmse_by_strategy.push(rmse);
    }

    let evppi_val = (conditional_max_total - surrogate_expected_current).max(0.0);
    let realized_gain = realized_total - expected_current;
    let decision_loss = (perfect_total - realized_total).max(0.0);
    if ![
        evppi_val,
        realized_gain,
        decision_loss,
        expected_current,
        perfect_total,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "EVPPI decision metrics are not finite",
        ));
    }
    Ok(DecisionFitSummary {
        evppi: evppi_val,
        realized_gain,
        decision_loss,
        expected_current,
        expected_perfect: perfect_total,
        r2_by_strategy,
        rmse_by_strategy,
    })
}

fn collect_applicability_warnings(
    sample_count: usize,
    basis_width: usize,
    has_constant_param: bool,
    evppi_in_sample: f64,
    oof: &DecisionFitSummary,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if sample_count < 10 * basis_width {
        warnings.push("low_sample_to_basis_ratio".to_string());
    }
    if has_constant_param {
        warnings.push("constant_parameter_column".to_string());
    }
    if oof.r2_by_strategy.iter().any(|&r2| r2 < 0.0) {
        warnings.push("negative_held_out_r2".to_string());
    }
    let evpi_gap_scale = (oof.expected_perfect - oof.expected_current)
        .max(0.0)
        .max(1.0);
    if evppi_in_sample - oof.realized_gain > 0.1 * evpi_gap_scale {
        warnings.push("optimism_gap_detected".to_string());
    }
    warnings
}

/// Evaluates cross-fitted nonlinear EVPPI and out-of-fold decision quality in Rust.
///
/// Both parameter standardization and spline quantile knot selection are
/// estimated strictly inside each training fold to prevent leakage across held-out
/// predictions.
///
/// # Errors
///
/// Returns an input or dimension error when sample counts fail to align, fold or
/// ridge settings are invalid, parameter/response reductions overflow, or the
/// expanded basis exceeds `256` columns.
pub fn evppi_nonlinear(
    net_benefit: &SampleMatrix,
    parameter_samples: &SampleMatrix,
    config: &EvppiNonlinearConfig,
) -> Result<EvppiNonlinearResult, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let [parameter_sample_count, parameter_count] = parameter_samples.shape();
    if sample_count != parameter_sample_count {
        return Err(NumericalInputError::dimension(
            "parameter_samples",
            sample_count,
            parameter_sample_count,
            "net-benefit and parameter sample counts must match",
        ));
    }
    if sample_count < 2 {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "nonlinear EVPPI requires at least 2 samples",
        ));
    }
    if config.fold_count < 2 || config.fold_count > sample_count {
        return Err(NumericalInputError::invalid(
            "fold_count",
            "fold_count must be between 2 and the sample count",
        ));
    }
    if !config.ridge_lambda.is_finite() || config.ridge_lambda < 0.0 {
        return Err(NumericalInputError::invalid(
            "ridge_lambda",
            "ridge_lambda must be finite and non-negative",
        ));
    }
    let basis_width = basis_width_for(config.basis, parameter_count)?;
    let full_indices: Vec<usize> = (0..sample_count).collect();
    let (_, _, has_constant_param) = subset_location_and_scale(parameter_samples, &full_indices)?;
    let linear_ref = evppi(net_benefit, parameter_samples)?;

    let estimator = match config.basis {
        EvppiBasis::QuadraticInteraction => "quadratic_interaction",
        EvppiBasis::PiecewiseLinearSpline { .. } => "piecewise_linear_spline",
    };

    let in_sample_preds = fit_and_predict_subset(
        net_benefit,
        parameter_samples,
        &full_indices,
        &full_indices,
        config.basis,
        config.ridge_lambda,
        basis_width,
    )?;
    let mut oof_preds = vec![vec![0.0; strategy_count]; sample_count];
    for fold in 0..config.fold_count {
        let mut train_indices = Vec::with_capacity(sample_count);
        let mut val_indices = Vec::with_capacity(sample_count / config.fold_count + 1);
        for row_idx in 0..sample_count {
            if row_idx % config.fold_count == fold {
                val_indices.push(row_idx);
            } else {
                train_indices.push(row_idx);
            }
        }
        let fold_preds = fit_and_predict_subset(
            net_benefit,
            parameter_samples,
            &train_indices,
            &val_indices,
            config.basis,
            config.ridge_lambda,
            basis_width,
        )?;
        for (local_idx, row_idx) in val_indices.into_iter().enumerate() {
            oof_preds[row_idx].clone_from(&fold_preds[local_idx]);
        }
    }

    let in_sample = summarize_decision_and_fit(net_benefit, &in_sample_preds)?;
    let oof = summarize_decision_and_fit(net_benefit, &oof_preds)?;
    let warnings = collect_applicability_warnings(
        sample_count,
        basis_width,
        has_constant_param,
        in_sample.evppi,
        &oof,
    );

    Ok(EvppiNonlinearResult {
        estimator,
        contract_version: 1,
        evppi_in_sample: in_sample.evppi,
        evppi_held_out: oof.evppi,
        held_out_realized_gain: oof.realized_gain,
        held_out_decision_loss_vs_perfect: oof.decision_loss,
        linear_evppi_reference: linear_ref,
        expected_current_value: oof.expected_current,
        expected_perfect_information: oof.expected_perfect,
        held_out_r2_by_strategy: oof.r2_by_strategy,
        held_out_rmse_by_strategy: oof.rmse_by_strategy,
        basis_width,
        sample_count,
        strategy_count,
        parameter_count,
        fold_count: config.fold_count,
        ridge_lambda: config.ridge_lambda,
        applicability_warnings: warnings,
    })
}

fn validate_mars_artifact(
    artifact: &MarsArtifact,
    strategy_count: usize,
    parameter_count: usize,
) -> Result<(), NumericalInputError> {
    if artifact.schema_version != MARS_ARTIFACT_SCHEMA_V1 {
        return Err(NumericalInputError::invalid(
            "schema_version",
            "MARS artifact schema_version must be 'voiage.mars-artifact.v1'",
        ));
    }
    if artifact.producer.trim().is_empty() || artifact.producer != artifact.producer.trim() {
        return Err(NumericalInputError::invalid(
            "producer",
            "MARS artifact producer must be a non-empty untrimmed identifier",
        ));
    }
    if artifact.strategies.len() != strategy_count {
        return Err(NumericalInputError::dimension(
            "artifact.strategies",
            strategy_count,
            artifact.strategies.len(),
            "MARS artifact strategy count must match net_benefit columns",
        ));
    }
    for model in &artifact.strategies {
        if !model.intercept.is_finite() {
            return Err(NumericalInputError::invalid(
                "intercept",
                "MARS strategy intercept must be finite",
            ));
        }
        for term in &model.terms {
            if !term.coefficient.is_finite() {
                return Err(NumericalInputError::invalid(
                    "coefficient",
                    "MARS term coefficient must be finite",
                ));
            }
            if term.factors.is_empty() {
                return Err(NumericalInputError::invalid(
                    "factors",
                    "MARS term must contain at least one hinge factor",
                ));
            }
            for factor in &term.factors {
                if factor.parameter_index >= parameter_count {
                    return Err(NumericalInputError::invalid(
                        "parameter_index",
                        "MARS hinge parameter_index is out of bounds",
                    ));
                }
                if !factor.knot.is_finite() {
                    return Err(NumericalInputError::invalid(
                        "knot",
                        "MARS hinge knot must be finite",
                    ));
                }
                if factor.direction != 1 && factor.direction != -1 {
                    return Err(NumericalInputError::invalid(
                        "direction",
                        "MARS hinge direction must be 1 or -1",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Replays a producer-owned MARS fitted surrogate artifact against PSA samples.
///
/// # Errors
///
/// Returns an input or dimension error when the artifact schema or producer
/// identity is invalid, strategy/parameter/row dimensions mismatch, hinge
/// parameters or directions are invalid, or evaluation produces non-finite
/// predictions.
pub fn evppi_from_mars_artifact(
    net_benefit: &SampleMatrix,
    parameter_samples: &SampleMatrix,
    artifact: &MarsArtifact,
) -> Result<EvppiMarsResult, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let [parameter_sample_count, parameter_count] = parameter_samples.shape();
    if sample_count != parameter_sample_count {
        return Err(NumericalInputError::dimension(
            "parameter_samples",
            sample_count,
            parameter_sample_count,
            "net-benefit and parameter sample counts must match",
        ));
    }
    validate_mars_artifact(artifact, strategy_count, parameter_count)?;

    let mut predictions = vec![vec![0.0; strategy_count]; sample_count];
    for (row_idx, pred_row) in predictions.iter_mut().enumerate() {
        let params = row_at(parameter_samples, row_idx);
        for (strategy_idx, model) in artifact.strategies.iter().enumerate() {
            let mut y_hat = model.intercept;
            for term in &model.terms {
                let mut prod = term.coefficient;
                for factor in &term.factors {
                    let x_val = params[factor.parameter_index];
                    let hinge = if factor.direction == 1 {
                        (x_val - factor.knot).max(0.0)
                    } else {
                        (factor.knot - x_val).max(0.0)
                    };
                    prod *= hinge;
                }
                y_hat += prod;
            }
            if !y_hat.is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "MARS surrogate evaluation produced a non-finite prediction",
                ));
            }
            pred_row[strategy_idx] = y_hat;
        }
    }

    let summary = summarize_decision_and_fit(net_benefit, &predictions)?;

    Ok(EvppiMarsResult {
        estimator: "mars_artifact_replay",
        contract_version: 1,
        evppi: summary.evppi,
        realized_decision_gain: summary.realized_gain,
        decision_loss_vs_perfect: summary.decision_loss,
        expected_current_value: summary.expected_current,
        expected_perfect_information: summary.expected_perfect,
        r2_by_strategy: summary.r2_by_strategy,
        rmse_by_strategy: summary.rmse_by_strategy,
        sample_count,
        strategy_count,
        parameter_count,
    })
}
