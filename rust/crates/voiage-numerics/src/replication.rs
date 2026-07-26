use crate::NumericalInputError;

/// Between-run uncertainty and split-sample convergence for independent estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicationSummary {
    /// Mean estimate across independent runs.
    pub mean: f64,
    /// Unbiased between-run sample variance.
    pub variance: f64,
    /// Standard error of the replicated mean.
    pub standard_error: f64,
    /// Number of independent runs.
    pub replications: usize,
    /// Relative difference between the first- and second-half means.
    pub split_mean_relative_difference: f64,
    /// Whether the split-mean diagnostic satisfies the requested tolerance.
    pub converged: bool,
}

/// Summarizes independently replicated estimator results.
///
/// The input order must be fixed before evaluation. Convergence compares the
/// first and second contiguous halves using a scale bounded below by one, so
/// estimates close to zero remain well-defined.
///
/// # Errors
///
/// Returns an input error for fewer than two estimates, non-finite estimates,
/// an invalid tolerance, or non-finite derived statistics.
pub fn summarize_replications(
    estimates: &[f64],
    relative_tolerance: f64,
) -> Result<ReplicationSummary, NumericalInputError> {
    if estimates.len() < 2 {
        return Err(NumericalInputError::invalid(
            "estimates",
            "at least two independent estimates are required",
        ));
    }
    if estimates.iter().any(|value| !value.is_finite()) {
        return Err(NumericalInputError::invalid(
            "estimates",
            "replicated estimates must be finite",
        ));
    }
    if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
        return Err(NumericalInputError::invalid(
            "relative_tolerance",
            "relative tolerance must be finite and positive",
        ));
    }
    let count = exact_count(estimates.len())?;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (index, value) in estimates.iter().copied().enumerate() {
        let observed = exact_count(index + 1)?;
        let delta = value - mean;
        mean += delta / observed;
        m2 += delta * (value - mean);
    }
    let variance = (m2 / (count - 1.0)).max(0.0);
    let standard_error = (variance / count).sqrt();
    let midpoint = estimates.len() / 2;
    let first_mean = stable_mean(&estimates[..midpoint])?;
    let second_mean = stable_mean(&estimates[midpoint..])?;
    let scale = mean
        .abs()
        .max(first_mean.abs())
        .max(second_mean.abs())
        .max(1.0);
    let split_mean_relative_difference = (first_mean - second_mean).abs() / scale;
    if [
        mean,
        variance,
        standard_error,
        split_mean_relative_difference,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        Ok(ReplicationSummary {
            mean,
            variance,
            standard_error,
            replications: estimates.len(),
            split_mean_relative_difference,
            converged: split_mean_relative_difference <= relative_tolerance,
        })
    } else {
        Err(NumericalInputError::invalid(
            "estimates",
            "replication summary is not finite",
        ))
    }
}

fn stable_mean(values: &[f64]) -> Result<f64, NumericalInputError> {
    let mut mean = 0.0;
    for (index, value) in values.iter().copied().enumerate() {
        mean += (value - mean) / exact_count(index + 1)?;
    }
    Ok(mean)
}

fn exact_count(count: usize) -> Result<f64, NumericalInputError> {
    u32::try_from(count).map(f64::from).map_err(|_| {
        NumericalInputError::invalid(
            "estimates",
            "replication count exceeds the supported exact range",
        )
    })
}
