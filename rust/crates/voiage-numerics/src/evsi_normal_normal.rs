use std::f64::consts::{PI, SQRT_2};

use crate::NumericalInputError;

fn standard_normal_density(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (2.0 * PI).sqrt()
}

fn standard_normal_cdf(value: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26. The approximation error is below
    // 1.5e-7 and is negligible relative to Monte Carlo error in VOI models.
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs() / SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    let erf = sign * (1.0 - polynomial * (-x * x).exp());
    0.5 * (1.0 + erf)
}

fn expected_positive_normal(mean: f64, standard_deviation: f64) -> f64 {
    if standard_deviation == 0.0 {
        return mean.max(0.0);
    }
    let z = mean / standard_deviation;
    standard_deviation * standard_normal_density(z) + mean * standard_normal_cdf(z)
}

/// Calculate EVSI for a declared equal-allocation, two-arm normal study.
///
/// The uncertain incremental effect has a normal prior. The proposed study
/// allocates half of `total_sample_size` to each arm and assumes a common known
/// individual outcome standard deviation. Incremental net benefit is linear in
/// the effect: `net_benefit_slope * effect + net_benefit_intercept`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] when an input is non-finite, either standard
/// deviation is not positive, the total sample size is zero or odd, the sample
/// size cannot be represented exactly by the kernel, or the result is
/// non-finite.
pub fn normal_normal_two_arm_evsi(
    prior_mean: f64,
    prior_standard_deviation: f64,
    outcome_standard_deviation: f64,
    total_sample_size: usize,
    net_benefit_slope: f64,
    net_benefit_intercept: f64,
) -> Result<f64, NumericalInputError> {
    if !prior_mean.is_finite()
        || !prior_standard_deviation.is_finite()
        || !outcome_standard_deviation.is_finite()
        || !net_benefit_slope.is_finite()
        || !net_benefit_intercept.is_finite()
    {
        return Err(NumericalInputError::invalid(
            "normal_normal_two_arm_evsi",
            "normal-normal EVSI inputs must be finite",
        ));
    }
    if prior_standard_deviation <= 0.0 {
        return Err(NumericalInputError::invalid(
            "prior_standard_deviation",
            "prior standard deviation must be positive",
        ));
    }
    if outcome_standard_deviation <= 0.0 {
        return Err(NumericalInputError::invalid(
            "outcome_standard_deviation",
            "outcome standard deviation must be positive",
        ));
    }
    if total_sample_size == 0 || total_sample_size % 2 != 0 {
        return Err(NumericalInputError::invalid(
            "total_sample_size",
            "total sample size must be a positive even integer",
        ));
    }
    let sample_size = u32::try_from(total_sample_size).map_err(|_| {
        NumericalInputError::invalid(
            "total_sample_size",
            "total sample size exceeds the supported u32 range",
        )
    })?;

    let prior_variance = prior_standard_deviation * prior_standard_deviation;
    let outcome_variance = outcome_standard_deviation * outcome_standard_deviation;
    let sampling_variance = 4.0 * outcome_variance / f64::from(sample_size);
    let posterior_mean_variance =
        prior_variance * prior_variance / (prior_variance + sampling_variance);
    let incremental_net_benefit_mean = net_benefit_slope * prior_mean + net_benefit_intercept;
    let incremental_net_benefit_standard_deviation =
        net_benefit_slope.abs() * posterior_mean_variance.sqrt();
    let expected_after_study = expected_positive_normal(
        incremental_net_benefit_mean,
        incremental_net_benefit_standard_deviation,
    );
    let current_value = incremental_net_benefit_mean.max(0.0);
    let result = (expected_after_study - current_value).max(0.0);
    if !result.is_finite() {
        return Err(NumericalInputError::invalid(
            "normal_normal_two_arm_evsi",
            "normal-normal EVSI result is not finite",
        ));
    }
    Ok(result)
}
