use std::collections::BTreeMap;

use voiage_domain::SampleVector;

use crate::NumericalInputError;

const MIX64: u64 = 0x9E37_79B9_7F4A_7C15;

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
    /// Number of seeded bootstrap replicates used for assurance.
    pub bootstrap_replicates: usize,
    /// Bootstrap standard error of the raw variance reduction.
    pub monte_carlo_standard_error: Option<f64>,
    /// Bootstrap percentile interval for the raw variance reduction.
    pub confidence_interval: Option<(f64, f64)>,
    /// Whether MCSE is within the declared fraction of prior variance.
    pub converged: bool,
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

/// Estimate `EVPPI_var` with deterministic paired bootstrap assurance.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for the same invalid inputs as
/// [`evppi_variance`], fewer than two bootstrap replicates, or a non-positive
/// convergence threshold.
pub fn evppi_variance_with_assurance(
    target_samples: &SampleVector,
    conditioning_groups: &[String],
    bootstrap_replicates: usize,
    seed: u64,
    convergence_threshold: f64,
) -> Result<EstimationVarianceKernelResult, NumericalInputError> {
    let result = evppi_variance(target_samples, conditioning_groups)?;
    validate_assurance(bootstrap_replicates, convergence_threshold)?;
    let mut reductions = Vec::with_capacity(bootstrap_replicates);
    for replicate in 0..bootstrap_replicates {
        let mut state = bootstrap_state(seed, replicate);
        let mut samples = Vec::with_capacity(target_samples.len());
        let mut groups = Vec::with_capacity(target_samples.len());
        for _ in 0..target_samples.len() {
            let index = next_index(&mut state, target_samples.len());
            samples.push(target_samples.as_slice()[index]);
            groups.push(conditioning_groups[index].clone());
        }
        let samples = SampleVector::try_from(samples).map_err(|_| {
            NumericalInputError::invalid(
                "target_samples",
                "bootstrap produced invalid target samples",
            )
        })?;
        reductions.push(evppi_variance(&samples, &groups)?.raw_reduction);
    }
    apply_assurance(result, reductions, convergence_threshold)
}

/// Estimate `EVSI_var` with deterministic independent bootstrap assurance.
///
/// Prior target samples and posterior-variance evaluations represent distinct
/// Monte Carlo stages and are therefore resampled independently within each
/// replicate.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for the same invalid inputs as
/// [`evsi_variance`], fewer than two bootstrap replicates, or a non-positive
/// convergence threshold.
pub fn evsi_variance_with_assurance(
    prior_target_samples: &SampleVector,
    posterior_variances: &SampleVector,
    bootstrap_replicates: usize,
    seed: u64,
    convergence_threshold: f64,
) -> Result<EstimationVarianceKernelResult, NumericalInputError> {
    let result = evsi_variance(prior_target_samples, posterior_variances)?;
    validate_assurance(bootstrap_replicates, convergence_threshold)?;
    let mut reductions = Vec::with_capacity(bootstrap_replicates);
    for replicate in 0..bootstrap_replicates {
        let mut state = bootstrap_state(seed, replicate);
        let prior = (0..prior_target_samples.len())
            .map(|_| {
                prior_target_samples.as_slice()[next_index(&mut state, prior_target_samples.len())]
            })
            .collect::<Vec<_>>();
        let posterior = (0..posterior_variances.len())
            .map(|_| {
                posterior_variances.as_slice()[next_index(&mut state, posterior_variances.len())]
            })
            .collect::<Vec<_>>();
        let prior = SampleVector::try_from(prior).map_err(|_| {
            NumericalInputError::invalid(
                "prior_target_samples",
                "bootstrap produced invalid prior target samples",
            )
        })?;
        let posterior = SampleVector::try_from(posterior).map_err(|_| {
            NumericalInputError::invalid(
                "posterior_variances",
                "bootstrap produced invalid posterior variances",
            )
        })?;
        reductions.push(evsi_variance(&prior, &posterior)?.raw_reduction);
    }
    apply_assurance(result, reductions, convergence_threshold)
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
        bootstrap_replicates: 0,
        monte_carlo_standard_error: None,
        confidence_interval: None,
        converged: true,
    })
}

fn validate_assurance(
    bootstrap_replicates: usize,
    convergence_threshold: f64,
) -> Result<(), NumericalInputError> {
    if bootstrap_replicates < 2 {
        return Err(NumericalInputError::invalid(
            "bootstrap_replicates",
            "bootstrap assurance requires at least two replicates",
        ));
    }
    if !convergence_threshold.is_finite() || convergence_threshold <= 0.0 {
        return Err(NumericalInputError::invalid(
            "convergence_threshold",
            "convergence threshold must be positive and finite",
        ));
    }
    Ok(())
}

fn bootstrap_state(seed: u64, replicate: usize) -> u64 {
    seed ^ ((replicate as u64 + 1).wrapping_mul(MIX64))
}

fn next_index(state: &mut u64, sample_count: usize) -> usize {
    if *state == 0 {
        *state = MIX64;
    }
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let mixed = (*state).wrapping_mul(0x2545_F491_4F6C_DD1D);
    let modulus = u64::try_from(sample_count).expect("validated sample count fits u64");
    usize::try_from(mixed % modulus).expect("bootstrap index is bounded by sample count")
}

fn apply_assurance(
    mut result: EstimationVarianceKernelResult,
    mut reductions: Vec<f64>,
    convergence_threshold: f64,
) -> Result<EstimationVarianceKernelResult, NumericalInputError> {
    let count = exact_count(reductions.len(), "bootstrap_replicates")?;
    let center = mean(&reductions, "bootstrap_reductions")?;
    let sum_squares = reductions.iter().try_fold(0.0, |total, value| {
        checked_sum(
            total,
            (value - center) * (value - center),
            "bootstrap_reductions",
        )
    })?;
    let denominator = exact_count(reductions.len() - 1, "bootstrap_replicates")?;
    let standard_error = (sum_squares / denominator / count).sqrt();
    if !standard_error.is_finite() {
        return Err(NumericalInputError::invalid(
            "bootstrap_reductions",
            "bootstrap standard error must be finite",
        ));
    }
    reductions.sort_by(f64::total_cmp);
    let last = reductions.len() - 1;
    let lower_index = last.saturating_mul(25) / 1_000;
    let upper_index = last.saturating_mul(975).div_ceil(1_000);
    let confidence_interval = (reductions[lower_index], reductions[upper_index.min(last)]);
    let convergence_scale = result.prior_variance.max(f64::EPSILON);
    result.bootstrap_replicates = reductions.len();
    result.monte_carlo_standard_error = Some(standard_error);
    result.confidence_interval = Some(confidence_interval);
    result.converged = standard_error <= convergence_threshold * convergence_scale;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use voiage_domain::SampleVector;

    use super::{
        evppi_variance, evppi_variance_with_assurance, evsi_variance, evsi_variance_with_assurance,
    };

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

    #[test]
    fn seeded_assurance_is_reproducible_and_reports_convergence() {
        let samples = SampleVector::try_from(vec![0.0, 2.0, 1.0, 3.0]).unwrap();
        let groups = vec!["a".into(), "a".into(), "b".into(), "b".into()];
        let first = evppi_variance_with_assurance(&samples, &groups, 128, 17, 1.0).unwrap();
        let second = evppi_variance_with_assurance(&samples, &groups, 128, 17, 1.0).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.bootstrap_replicates, 128);
        assert!(first.monte_carlo_standard_error.is_some());
        assert!(first.confidence_interval.is_some());
        assert!(first.converged);

        let posterior = SampleVector::try_from(vec![0.8, 1.0, 1.2, 1.0]).unwrap();
        let evsi = evsi_variance_with_assurance(&samples, &posterior, 128, 17, 1.0).unwrap();
        assert!(evsi.monte_carlo_standard_error.is_some());
    }

    #[test]
    fn malformed_assurance_settings_fail_closed() {
        let samples = SampleVector::try_from(vec![0.0, 1.0]).unwrap();
        let groups = vec!["a".into(), "b".into()];
        assert!(evppi_variance_with_assurance(&samples, &groups, 1, 0, 0.01).is_err());
        assert!(evppi_variance_with_assurance(&samples, &groups, 2, 0, f64::NAN).is_err());
    }
}
