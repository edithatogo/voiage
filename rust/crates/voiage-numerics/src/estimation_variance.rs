use std::collections::BTreeMap;

use voiage_domain::SampleVector;

use crate::NumericalInputError;

/// Rust-owned scalar variance-reduction result shared by estimation methods.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimationVarianceKernelResult {
    /// Population variance of the declared target under current information.
    pub prior_variance: f64,
    /// Expected conditional or posterior population variance.
    pub expected_posterior_variance: f64,
    /// Unclipped finite-sample difference.
    pub raw_reduction: f64,
    /// Reported reduction after clipping a negative finite-sample estimate.
    pub absolute_reduction: f64,
    /// Reduction divided by prior variance, or `None` for zero prior variance.
    pub relative_reduction: Option<f64>,
    /// Number of prior target samples.
    pub prior_sample_count: usize,
    /// Number of conditioning groups or posterior variance evaluations.
    pub posterior_evaluation_count: usize,
}

/// Estimate scalar `EVPPI_var` by exact aggregation over discrete groups.
///
/// Population variance is used because the supplied equally weighted samples
/// represent the declared empirical probability mass. The conditioning labels
/// may be strings or serialized discrete states; lexical grouping only affects
/// deterministic traversal, not the result.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for fewer than two target samples, a label
/// count mismatch, blank labels, or an intermediate non-finite result.
pub fn evppi_variance(
    target_samples: &SampleVector,
    conditioning_groups: &[String],
) -> Result<EstimationVarianceKernelResult, NumericalInputError> {
    require_prior_samples(target_samples)?;
    if target_samples.len() != conditioning_groups.len() {
        return Err(NumericalInputError::dimension(
            "conditioning_groups",
            target_samples.len(),
            conditioning_groups.len(),
            "conditioning-group count must match target sample count",
        ));
    }

    let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (value, label) in target_samples
        .as_slice()
        .iter()
        .copied()
        .zip(conditioning_groups)
    {
        if label.trim().is_empty() {
            return Err(NumericalInputError::invalid(
                "conditioning_groups",
                "conditioning-group labels must not be blank",
            ));
        }
        groups.entry(label).or_default().push(value);
    }

    let prior_variance = population_variance(target_samples.as_slice(), "target_samples")?;
    let sample_count = exact_count(target_samples.len(), "target_samples")?;
    let mut expected_posterior_variance = 0.0;
    for members in groups.values() {
        let weight = exact_count(members.len(), "conditioning_groups")? / sample_count;
        let within = population_variance(members, "target_samples")?;
        expected_posterior_variance = checked_sum(
            expected_posterior_variance,
            weight * within,
            "target_samples",
        )?;
    }
    make_result(
        prior_variance,
        expected_posterior_variance,
        target_samples.len(),
        groups.len(),
    )
}

/// Aggregate scalar `EVSI_var` from posterior-variance evaluations.
///
/// The caller owns the declared sampling model and supplies one finite
/// posterior variance for each simulated or enumerated dataset. This kernel
/// averages those values and applies the governed negative-estimate and
/// zero-prior-variance policies.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for fewer than two prior target samples,
/// negative posterior variances, or an intermediate non-finite result.
pub fn evsi_variance(
    prior_target_samples: &SampleVector,
    posterior_variances: &SampleVector,
) -> Result<EstimationVarianceKernelResult, NumericalInputError> {
    require_prior_samples(prior_target_samples)?;
    if posterior_variances
        .as_slice()
        .iter()
        .any(|variance| *variance < 0.0)
    {
        return Err(NumericalInputError::invalid(
            "posterior_variances",
            "posterior variances must be nonnegative",
        ));
    }

    let prior_variance =
        population_variance(prior_target_samples.as_slice(), "prior_target_samples")?;
    let expected_posterior_variance = mean(posterior_variances.as_slice(), "posterior_variances")?;
    make_result(
        prior_variance,
        expected_posterior_variance,
        prior_target_samples.len(),
        posterior_variances.len(),
    )
}

fn require_prior_samples(samples: &SampleVector) -> Result<(), NumericalInputError> {
    if samples.len() < 2 {
        return Err(NumericalInputError::invalid(
            "target_samples",
            "at least two prior target samples are required",
        ));
    }
    Ok(())
}

fn population_variance(values: &[f64], field: &'static str) -> Result<f64, NumericalInputError> {
    let mut count = 0.0;
    let mut mean = 0.0;
    let mut sum_squared_deviations = 0.0;
    for value in values {
        count += 1.0;
        let delta = value - mean;
        mean += delta / count;
        let updated = sum_squared_deviations + delta * (value - mean);
        if !mean.is_finite() || !updated.is_finite() {
            return Err(NumericalInputError::invalid(
                field,
                "variance calculation exceeded the finite numerical range",
            ));
        }
        sum_squared_deviations = updated;
    }
    let variance = (sum_squared_deviations / count).max(0.0);
    if variance.is_finite() {
        Ok(variance)
    } else {
        Err(NumericalInputError::invalid(
            field,
            "variance result must be finite",
        ))
    }
}

fn mean(values: &[f64], field: &'static str) -> Result<f64, NumericalInputError> {
    let count = exact_count(values.len(), field)?;
    values
        .iter()
        .try_fold(0.0, |total, value| checked_sum(total, value / count, field))
}

fn checked_sum(left: f64, right: f64, field: &'static str) -> Result<f64, NumericalInputError> {
    let result = left + right;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(NumericalInputError::invalid(
            field,
            "aggregation exceeded the finite numerical range",
        ))
    }
}

fn exact_count(count: usize, field: &'static str) -> Result<f64, NumericalInputError> {
    u32::try_from(count).map(f64::from).map_err(|_| {
        NumericalInputError::invalid(field, "sample count exceeds the supported exact range")
    })
}

fn make_result(
    prior_variance: f64,
    expected_posterior_variance: f64,
    prior_sample_count: usize,
    posterior_evaluation_count: usize,
) -> Result<EstimationVarianceKernelResult, NumericalInputError> {
    let raw_reduction = prior_variance - expected_posterior_variance;
    if !raw_reduction.is_finite() {
        return Err(NumericalInputError::invalid(
            "variance_reduction",
            "variance reduction must be finite",
        ));
    }
    let absolute_reduction = raw_reduction.max(0.0);
    let relative_reduction = if prior_variance == 0.0 {
        None
    } else {
        Some(absolute_reduction / prior_variance)
    };
    Ok(EstimationVarianceKernelResult {
        prior_variance,
        expected_posterior_variance,
        raw_reduction,
        absolute_reduction,
        relative_reduction,
        prior_sample_count,
        posterior_evaluation_count,
    })
}

#[cfg(test)]
mod tests {
    use voiage_domain::SampleVector;

    use super::{evppi_variance, evsi_variance};

    #[test]
    fn discrete_evppi_variance_matches_enumerable_reference() {
        let samples = SampleVector::try_from(vec![0.0, 2.0, 1.0, 3.0]).unwrap();
        let groups = vec!["a".into(), "a".into(), "b".into(), "b".into()];
        let result = evppi_variance(&samples, &groups).unwrap();
        assert!((result.prior_variance - 1.25).abs() < 1.0e-12);
        assert!((result.expected_posterior_variance - 1.0).abs() < 1.0e-12);
        assert!((result.absolute_reduction - 0.25).abs() < 1.0e-12);
        assert_eq!(result.relative_reduction, Some(0.2));
        assert_eq!(result.posterior_evaluation_count, 2);
    }

    #[test]
    fn posterior_variance_aggregation_retains_negative_raw_estimate() {
        let samples = SampleVector::try_from(vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let posterior = SampleVector::try_from(vec![1.5, 1.5]).unwrap();
        let result = evsi_variance(&samples, &posterior).unwrap();
        assert!((result.prior_variance - 1.25).abs() < 1.0e-12);
        assert!((result.expected_posterior_variance - 1.5).abs() < 1.0e-12);
        assert!((result.raw_reduction + 0.25).abs() < 1.0e-12);
        assert!(result.absolute_reduction.abs() < 1.0e-12);
        assert_eq!(result.relative_reduction, Some(0.0));
    }

    #[test]
    fn zero_and_perfect_information_policies_are_explicit() {
        let constant = SampleVector::try_from(vec![2.0, 2.0]).unwrap();
        let posterior = SampleVector::try_from(vec![0.0]).unwrap();
        let zero = evsi_variance(&constant, &posterior).unwrap();
        assert!(zero.absolute_reduction.abs() < 1.0e-12);
        assert_eq!(zero.relative_reduction, None);

        let samples = SampleVector::try_from(vec![0.0, 2.0]).unwrap();
        let groups = vec!["left".into(), "right".into()];
        let perfect = evppi_variance(&samples, &groups).unwrap();
        assert!(perfect.expected_posterior_variance.abs() < 1.0e-12);
        assert_eq!(perfect.relative_reduction, Some(1.0));
    }

    #[test]
    fn malformed_estimation_inputs_fail_closed() {
        let one_sample = SampleVector::try_from(vec![1.0]).unwrap();
        assert!(evppi_variance(&one_sample, &["a".into()]).is_err());

        let samples = SampleVector::try_from(vec![0.0, 1.0]).unwrap();
        assert!(evppi_variance(&samples, &["a".into()]).is_err());
        assert!(evppi_variance(&samples, &[" ".into(), "b".into()]).is_err());
        let negative = SampleVector::try_from(vec![-0.1]).unwrap();
        assert!(evsi_variance(&samples, &negative).is_err());
    }
}
