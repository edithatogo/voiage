use voiage_domain::SampleMatrix;

use crate::{EvsiApproximationResult, NumericalInputError};

/// Computes the deterministic centered linear-plus-quadratic EVSI approximation.
///
/// The design columns are ordered as intercept, centered linear terms, centered
/// squares, and pairwise centered interactions. The native contract fails
/// closed for rank-deficient designs; the Python compatibility path retains
/// `NumPy`'s minimum-norm behavior until a matching SVD contract is specified.
///
/// # Errors
///
/// Returns an input error for invalid sizes, non-finite intermediates, or a
/// rank-deficient design, and a dimension error when sample counts differ.
#[allow(clippy::too_many_lines)]
pub fn evsi_moment_based(
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
    let (means, design_width) = parameter_means_and_width(parameter_samples, parameter_count)?;
    let mut normal = vec![vec![0.0; design_width]; design_width];
    for row in parameter_samples.rows() {
        let design = design_row(row, &means);
        for left in 0..design_width {
            for right in 0..design_width {
                add_scaled(
                    &mut normal[left][right],
                    design[left] * design[right],
                    sample_divisor,
                    "parameter_samples",
                    "moment-based normal matrix is not finite",
                )?;
            }
        }
    }

    let mut predictions = vec![vec![0.0; strategy_count]; sample_count];
    for strategy in 0..strategy_count {
        let mut rhs = vec![0.0; design_width];
        for (row, values) in parameter_samples.rows().zip(net_benefit.rows()) {
            let design = design_row(row, &means);
            for column in 0..design_width {
                add_checked(
                    &mut rhs[column],
                    (design[column] / sample_divisor) * values[strategy],
                    "net_benefit",
                    "moment-based response is not finite",
                )?;
            }
        }
        let coefficients = solve_full_rank(&normal, &rhs)?;
        for (index, row) in parameter_samples.rows().enumerate() {
            let design = design_row(row, &means);
            predictions[index][strategy] = design
                .iter()
                .zip(coefficients.iter())
                .map(|(left, right)| left * right)
                .sum();
            if !predictions[index][strategy].is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "moment-based prediction is not finite",
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
    let expected_sample_value = predictions
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
            let count = f64::from(u32::try_from(index + 1).ok()?);
            expected.is_finite().then_some((expected, count))
        })
        .try_fold(0.0, |mean, value| {
            value.and_then(|(value, count)| {
                let updated = mean + (value - mean) / count;
                updated.is_finite().then_some(updated)
            })
        })
        .ok_or_else(|| {
            NumericalInputError::invalid("net_benefit", "moment-based expected value is not finite")
        })?
        .clamp(
            expected_current_value,
            expected_current_value.max(expected_perfect_information),
        );
    let evsi = (expected_sample_value - expected_current_value).max(0.0);
    if !evsi.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "moment-based EVSI is not finite",
        ));
    }
    Ok(EvsiApproximationResult {
        estimator: "moment_based",
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

fn parameter_means_and_width(
    samples: &SampleMatrix,
    parameter_count: usize,
) -> Result<(Vec<f64>, usize), NumericalInputError> {
    let design_width = parameter_count
        .checked_mul(2)
        .and_then(|value| {
            parameter_count
                .checked_mul(parameter_count.saturating_sub(1))
                .map(|interactions| value + interactions / 2)
        })
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            NumericalInputError::invalid("parameter_samples", "moment-based design width overflows")
        })?;
    let mut means = vec![0.0; parameter_count];
    for (index, row) in samples.rows().enumerate() {
        let count = f64::from(u32::try_from(index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "parameter_samples",
                "sample count exceeds the supported numerical range",
            )
        })?);
        for (column, value) in row.iter().copied().enumerate() {
            means[column] += (value - means[column]) / count;
        }
    }
    if means.iter().any(|value| !value.is_finite()) {
        return Err(NumericalInputError::invalid(
            "parameter_samples",
            "moment-based parameter mean is not finite",
        ));
    }
    Ok((means, design_width))
}

fn design_row(row: &[f64], means: &[f64]) -> Vec<f64> {
    let centered = row
        .iter()
        .zip(means)
        .map(|(value, mean)| value - mean)
        .collect::<Vec<_>>();
    let mut design = Vec::with_capacity(
        1 + centered.len() * 2 + centered.len() * centered.len().saturating_sub(1) / 2,
    );
    design.push(1.0);
    design.extend(centered.iter().copied());
    design.extend(centered.iter().map(|value| value * value));
    for left in 0..centered.len() {
        for right in (left + 1)..centered.len() {
            design.push(centered[left] * centered[right]);
        }
    }
    design
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

fn add_scaled(
    target: &mut f64,
    value: f64,
    divisor: f64,
    field: &'static str,
    message: &'static str,
) -> Result<(), NumericalInputError> {
    add_checked(target, value / divisor, field, message)
}

fn solve_full_rank(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, NumericalInputError> {
    let size = rhs.len();
    let mut augmented = matrix
        .iter()
        .zip(rhs)
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
            .expect("normal matrix is non-empty");
        let scale = augmented[column..]
            .iter()
            .map(|row| row[column].abs())
            .fold(0.0, f64::max);
        if pivot_value <= scale * 1.0e-12 || !pivot_value.is_finite() {
            return Err(NumericalInputError::invalid(
                "parameter_samples",
                "moment-based design is rank deficient",
            ));
        }
        augmented.swap(column, pivot_row);
        let pivot = augmented[column][column];
        for value in &mut augmented[column][column..=size] {
            *value /= pivot;
        }
        let pivot_values = augmented[column].clone();
        for (row, augmented_row) in augmented.iter_mut().enumerate().take(size) {
            if row == column {
                continue;
            }
            let factor = augmented_row[column];
            for (index, value) in augmented_row
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
    (0..matrix.shape()[1])
        .map(|strategy| {
            let mut mean = 0.0;
            for (index, row) in matrix.rows().enumerate() {
                let count = f64::from(u32::try_from(index + 1).map_err(|_| {
                    NumericalInputError::invalid(
                        "net_benefit",
                        "sample count exceeds the supported numerical range",
                    )
                })?);
                mean += (row[strategy] - mean) / count;
            }
            mean.is_finite().then_some(mean).ok_or_else(|| {
                NumericalInputError::invalid("net_benefit", "strategy mean is not finite")
            })
        })
        .collect()
}

fn row_max_mean(matrix: &SampleMatrix) -> Result<f64, NumericalInputError> {
    let mut mean = 0.0;
    for (index, row) in matrix.rows().enumerate() {
        let count = f64::from(u32::try_from(index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "net_benefit",
                "sample count exceeds the supported numerical range",
            )
        })?);
        let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        mean += (row_max - mean) / count;
    }
    mean.is_finite().then_some(mean).ok_or_else(|| {
        NumericalInputError::invalid("net_benefit", "perfect-information mean is not finite")
    })
}
