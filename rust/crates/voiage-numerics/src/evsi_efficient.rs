use voiage_domain::SampleMatrix;

use crate::NumericalInputError;

/// Deterministic result returned by a contract-bound EVSI approximation.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiApproximationResult {
    /// Stable estimator identifier.
    pub estimator: &'static str,
    /// Version of the efficient-linear numerical contract.
    pub contract_version: u32,
    /// Expected value under current information.
    pub expected_current_value: f64,
    /// Expected value after the estimator-specific approximation.
    pub expected_sample_value: f64,
    /// Expected value with perfect information.
    pub expected_perfect_information: f64,
    /// Fraction of information provided by the trial-size proxy.
    pub information_fraction: f64,
    /// Non-negative expected value of sample information.
    pub evsi: f64,
    /// Number of parameter samples.
    pub sample_count: usize,
    /// Number of strategies.
    pub strategy_count: usize,
    /// Number of parameters.
    pub parameter_count: usize,
}

/// Computes the deterministic efficient-linear EVSI approximation.
///
/// The kernel fits an ordinary least-squares model with an intercept for each
/// strategy, predicts on the supplied parameter samples, and applies the
/// versioned information-fraction and clipping contract. Trial-arm parsing,
/// callbacks, and population scaling remain outside this numerical kernel.
///
/// # Errors
///
/// Returns an input error for invalid trial sizes, non-finite intermediate
/// values, or rank-deficient parameter designs, and a dimension error when
/// sample counts differ.
#[allow(clippy::too_many_lines)]
pub fn evsi_efficient_linear(
    net_benefit: &SampleMatrix,
    parameter_samples: &SampleMatrix,
    trial_sample_size: usize,
) -> Result<EvsiApproximationResult, NumericalInputError> {
    if trial_sample_size == 0 {
        return Err(NumericalInputError::invalid(
            "trial_sample_size",
            "trial sample size must be positive",
        ));
    }
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
    let sample_count_u32 = u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported numerical range",
        )
    })?;
    let trial_sample_size_u32 = u32::try_from(trial_sample_size).map_err(|_| {
        NumericalInputError::invalid(
            "trial_sample_size",
            "trial sample size exceeds the supported numerical range",
        )
    })?;
    let sample_divisor = f64::from(sample_count_u32);
    let (means, scales) = location_and_scale(parameter_samples, sample_divisor)?;
    let coefficient_count = parameter_count + 1;
    let mut normal = vec![vec![0.0; coefficient_count]; coefficient_count];
    for row in parameter_samples.rows() {
        let design = design_row(row, &means, &scales);
        for left in 0..coefficient_count {
            for right in 0..coefficient_count {
                add_scaled_checked(
                    &mut normal[left][right],
                    design[left] * design[right],
                    sample_divisor,
                    "parameter_samples",
                    "efficient-linear normal matrix is not finite",
                )?;
            }
        }
    }

    let mut predictions = vec![vec![0.0; strategy_count]; sample_count];
    for strategy in 0..strategy_count {
        let mut rhs = vec![0.0; coefficient_count];
        for (row, values) in parameter_samples.rows().zip(net_benefit.rows()) {
            let design = design_row(row, &means, &scales);
            for coefficient in 0..coefficient_count {
                add_checked(
                    &mut rhs[coefficient],
                    (design[coefficient] / sample_divisor) * values[strategy],
                    "net_benefit",
                    "efficient-linear response is not finite",
                )?;
            }
        }
        let coefficients = solve_full_rank(&normal, &rhs)?;
        for (index, row) in parameter_samples.rows().enumerate() {
            let design = design_row(row, &means, &scales);
            predictions[index][strategy] = design
                .iter()
                .zip(coefficients.iter())
                .map(|(left, right)| left * right)
                .sum();
            if !predictions[index][strategy].is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "efficient-linear prediction is not finite",
                ));
            }
        }
    }

    let current_means = strategy_means(net_benefit)?;
    let expected_current_value = current_means
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let expected_perfect_information = row_max_mean(net_benefit)?;
    let information_fraction = f64::from(trial_sample_size_u32)
        / (f64::from(trial_sample_size_u32) + f64::from(sample_count_u32));
    let conditional_total = predictions
        .iter()
        .enumerate()
        .map(|(index, prediction)| {
            let expected = prediction
                .iter()
                .enumerate()
                .map(|(strategy, value)| {
                    let prior = current_means[strategy];
                    prior + information_fraction * (value - prior)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            if expected.is_finite() {
                let count = f64::from(u32::try_from(index + 1).ok()?);
                Some((expected, count))
            } else {
                None
            }
        })
        .try_fold(0.0, |mean, value| {
            value.and_then(|(value, count)| {
                let updated = mean + (value - mean) / count;
                updated.is_finite().then_some(updated)
            })
        })
        .ok_or_else(|| {
            NumericalInputError::invalid(
                "net_benefit",
                "efficient-linear expected value is not finite",
            )
        })?;
    let raw_sample_value = conditional_total;
    let expected_sample_value = raw_sample_value.clamp(
        expected_current_value,
        expected_current_value.max(expected_perfect_information),
    );
    let evsi = (expected_sample_value - expected_current_value).max(0.0);
    if !evsi.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "efficient-linear EVSI is not finite",
        ));
    }
    Ok(EvsiApproximationResult {
        estimator: "efficient_linear",
        contract_version: 1,
        expected_current_value,
        expected_sample_value,
        expected_perfect_information,
        information_fraction,
        evsi,
        sample_count,
        strategy_count,
        parameter_count,
    })
}

fn add_checked(
    target: &mut f64,
    value: f64,
    field: &'static str,
    message: &'static str,
) -> Result<(), NumericalInputError> {
    let updated = *target + value;
    if !updated.is_finite() {
        return Err(NumericalInputError::invalid(field, message));
    }
    *target = updated;
    Ok(())
}

fn add_scaled_checked(
    target: &mut f64,
    value: f64,
    divisor: f64,
    field: &'static str,
    message: &'static str,
) -> Result<(), NumericalInputError> {
    add_checked(target, value / divisor, field, message)
}

fn location_and_scale(
    samples: &SampleMatrix,
    divisor: f64,
) -> Result<(Vec<f64>, Vec<f64>), NumericalInputError> {
    let parameter_count = samples.shape()[1];
    let mut means = vec![0.0; parameter_count];
    for row in samples.rows() {
        for (index, value) in row.iter().copied().enumerate() {
            means[index] += value / divisor;
        }
    }
    let mut scales = vec![0.0_f64; parameter_count];
    for row in samples.rows() {
        for (index, value) in row.iter().copied().enumerate() {
            scales[index] = scales[index].max((value - means[index]).abs());
        }
    }
    for scale in &mut scales {
        if *scale == 0.0 {
            *scale = 1.0;
        }
    }
    if means
        .iter()
        .chain(scales.iter())
        .any(|value| !value.is_finite())
    {
        return Err(NumericalInputError::invalid(
            "parameter_samples",
            "efficient-linear parameter normalization is not finite",
        ));
    }
    Ok((means, scales))
}

fn design_row(row: &[f64], means: &[f64], scales: &[f64]) -> Vec<f64> {
    std::iter::once(1.0)
        .chain(
            row.iter()
                .zip(means.iter().zip(scales.iter()))
                .map(|(value, (mean, scale))| (value - mean) / scale),
        )
        .collect()
}

fn solve_full_rank(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, NumericalInputError> {
    let size = rhs.len();
    let mut augmented = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, value)| {
            let mut row = row.clone();
            row.push(*value);
            row
        })
        .collect::<Vec<_>>();
    for column in 0..size {
        let (pivot_row, pivot_value) = (column..size)
            .map(|row| (row, augmented[row][column].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("normal matrix has at least one row");
        let scale = augmented[column..]
            .iter()
            .map(|row| row[column].abs())
            .fold(0.0, f64::max);
        if pivot_value <= scale * 1.0e-12 || !pivot_value.is_finite() {
            return Err(NumericalInputError::invalid(
                "parameter_samples",
                "efficient-linear design is rank deficient",
            ));
        }
        augmented.swap(column, pivot_row);
        let pivot = augmented[column][column];
        for value in &mut augmented[column][column..=size] {
            *value /= pivot;
        }
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            let pivot_values = augmented[column].clone();
            for (index, value) in augmented[row]
                .iter_mut()
                .enumerate()
                .take(size + 1)
                .skip(column)
            {
                *value -= factor * pivot_values[index];
            }
        }
    }
    Ok(augmented.into_iter().map(|row| row[size]).collect())
}

fn strategy_means(matrix: &SampleMatrix) -> Result<Vec<f64>, NumericalInputError> {
    let strategy_count = matrix.shape()[1];
    (0..strategy_count)
        .map(|strategy| strategy_mean_checked(matrix, strategy))
        .collect()
}

fn strategy_mean_checked(
    matrix: &SampleMatrix,
    strategy: usize,
) -> Result<f64, NumericalInputError> {
    let mut mean = 0.0;
    for (index, row) in matrix.rows().enumerate() {
        let count = f64::from(u32::try_from(index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "net_benefit",
                "sample count exceeds the supported numerical range",
            )
        })?);
        let updated = mean + (row[strategy] - mean) / count;
        if !updated.is_finite() {
            return Err(NumericalInputError::invalid(
                "net_benefit",
                "strategy total is not finite",
            ));
        }
        mean = updated;
    }
    Ok(mean)
}

fn row_max_mean(matrix: &SampleMatrix) -> Result<f64, NumericalInputError> {
    let mut mean = 0.0;
    for (index, row) in matrix.rows().enumerate() {
        let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let count = f64::from(u32::try_from(index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "net_benefit",
                "sample count exceeds the supported numerical range",
            )
        })?);
        let updated = mean + (row_max - mean) / count;
        if !updated.is_finite() {
            return Err(NumericalInputError::invalid(
                "net_benefit",
                "perfect-information total is not finite",
            ));
        }
        mean = updated;
    }
    Ok(mean)
}
