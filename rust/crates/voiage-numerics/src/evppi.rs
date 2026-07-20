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
