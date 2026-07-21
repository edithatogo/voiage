use voiage_domain::SampleMatrix;

use crate::NumericalInputError;

/// Versioned result for the callback-driven regression aggregation boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiRegressionResult {
    /// Stable estimator identifier.
    pub estimator: &'static str,
    /// Version of the regression aggregation contract.
    pub contract_version: u32,
    /// Mean prediction over the requested prediction samples.
    pub expected_sample_value: f64,
    /// Number of fitted target rows.
    pub sample_count: usize,
    /// Number of prediction rows.
    pub prediction_count: usize,
    /// Number of predictor columns.
    pub parameter_count: usize,
}

/// Fit an intercept OLS model to callback-produced targets and predict them.
///
/// Callback execution and trial simulation remain Python-owned. This kernel
/// owns only the deterministic, finite regression aggregation contract.
///
/// # Errors
///
/// Returns an input error for non-finite regression values or a rank-deficient
/// design, and a dimension error when matrix shapes do not align.
pub fn evsi_regression(
    targets: &SampleMatrix,
    parameter_samples: &SampleMatrix,
    prediction_samples: &SampleMatrix,
) -> Result<EvsiRegressionResult, NumericalInputError> {
    let [sample_count, target_columns] = targets.shape();
    if target_columns != 1 {
        return Err(NumericalInputError::invalid(
            "targets",
            "regression targets must have exactly one column",
        ));
    }
    let [parameter_count_rows, parameter_count] = parameter_samples.shape();
    let [prediction_count, prediction_parameters] = prediction_samples.shape();
    if sample_count != parameter_count_rows {
        return Err(NumericalInputError::dimension(
            "parameter_samples",
            sample_count,
            parameter_count_rows,
            "targets and parameter samples must have matching rows",
        ));
    }
    if parameter_count != prediction_parameters {
        return Err(NumericalInputError::dimension(
            "prediction_samples",
            parameter_count,
            prediction_parameters,
            "parameter columns must match prediction samples",
        ));
    }
    let coefficient_count = parameter_count + 1;
    let mut normal = vec![vec![0.0; coefficient_count]; coefficient_count];
    let mut rhs = vec![0.0; coefficient_count];
    for (row, target) in parameter_samples.rows().zip(targets.rows()) {
        let design = std::iter::once(1.0)
            .chain(row.iter().copied())
            .collect::<Vec<_>>();
        for left in 0..coefficient_count {
            for right in 0..coefficient_count {
                add(
                    &mut normal[left][right],
                    design[left] * design[right],
                    "normal matrix",
                )?;
            }
            add(
                &mut rhs[left],
                design[left] * target[0],
                "regression target",
            )?;
        }
    }
    let coefficients = solve(&normal, &rhs)?;
    let mut mean = 0.0;
    for (index, row) in prediction_samples.rows().enumerate() {
        let prediction = std::iter::once(1.0)
            .chain(row.iter().copied())
            .zip(coefficients.iter().copied())
            .map(|(left, right)| left * right)
            .sum::<f64>();
        if !prediction.is_finite() {
            return Err(NumericalInputError::invalid(
                "prediction_samples",
                "regression prediction is not finite",
            ));
        }
        let count = f64::from(u32::try_from(index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "prediction_samples",
                "prediction count exceeds the supported numerical range",
            )
        })?);
        mean += (prediction - mean) / count;
    }
    if !mean.is_finite() {
        return Err(NumericalInputError::invalid(
            "prediction_samples",
            "regression expected sample value is not finite",
        ));
    }
    Ok(EvsiRegressionResult {
        estimator: "regression",
        contract_version: 1,
        expected_sample_value: mean,
        sample_count,
        prediction_count,
        parameter_count,
    })
}

fn add(target: &mut f64, value: f64, field: &'static str) -> Result<(), NumericalInputError> {
    *target += value;
    if target.is_finite() {
        Ok(())
    } else {
        Err(NumericalInputError::invalid(
            field,
            "regression value is not finite",
        ))
    }
}

fn solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, NumericalInputError> {
    let size = rhs.len();
    let mut augmented = matrix
        .iter()
        .zip(rhs.iter().copied())
        .map(|(row, value)| {
            let mut row = row.clone();
            row.push(value);
            row
        })
        .collect::<Vec<_>>();
    for pivot in 0..size {
        let Some((best, magnitude)) = (pivot..size)
            .map(|row| (row, augmented[row][pivot].abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
        else {
            return Err(NumericalInputError::invalid(
                "parameter_samples",
                "regression design is rank deficient",
            ));
        };
        if magnitude <= f64::EPSILON || !magnitude.is_finite() {
            return Err(NumericalInputError::invalid(
                "parameter_samples",
                "regression design is rank deficient",
            ));
        }
        augmented.swap(pivot, best);
        let divisor = augmented[pivot][pivot];
        for value in augmented[pivot].iter_mut().skip(pivot) {
            *value /= divisor;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            let pivot_values = augmented[pivot][pivot..=size].to_vec();
            for (value, pivot_value) in augmented[row][pivot..=size].iter_mut().zip(pivot_values) {
                *value -= factor * pivot_value;
            }
        }
    }
    Ok(augmented.into_iter().map(|row| row[size]).collect())
}
