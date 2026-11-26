use std::f64::consts::PI;

use voiage_domain::SampleSizeConvention;

use crate::NumericalInputError;

#[allow(clippy::excessive_precision)]
fn ln_gamma(z: f64) -> f64 {
    // Lanczos approximation (g = 7, n = 9) accurate to ~1e-14 for z > 0.
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return PI.ln() - (PI * z).sin().ln() - ln_gamma(1.0 - z);
    }
    let zm1 = z - 1.0;
    let mut x = COEFFS[0];
    for (idx, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (zm1 + f64::from(u32::try_from(idx).unwrap_or(1)));
    }
    let t = zm1 + 7.5;
    0.5 * (2.0 * PI).ln() + (zm1 + 0.5) * t.ln() - t + x.ln()
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

fn beta_binomial_pmf(k: u32, n: u32, alpha: f64, beta: f64) -> f64 {
    let kf = f64::from(k);
    let nf = f64::from(n);
    let ln_binom = ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0);
    let ln_p = ln_binom + ln_beta(kf + alpha, nf - kf + beta) - ln_beta(alpha, beta);
    ln_p.exp()
}

fn validate_positive_finite(value: f64, name: &'static str) -> Result<(), NumericalInputError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(NumericalInputError::invalid(
            name,
            "study-model parameter must be finite and positive",
        ));
    }
    Ok(())
}

/// Computes exact/tail-certified EVSI for a two-arm binary study under independent Beta priors.
///
/// Incremental net benefit is `net_benefit_slope * (p_1 - p_0) + net_benefit_intercept`,
/// where `p_0 ~ Beta(alpha_0, beta_0)` and `p_1 ~ Beta(alpha_1, beta_1)`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] if priors, slopes, or sample-size conventions are invalid.
pub fn beta_binomial_two_arm_evsi(
    alpha_0: f64,
    beta_0: f64,
    alpha_1: f64,
    beta_1: f64,
    sample_size: &SampleSizeConvention,
    net_benefit_slope: f64,
    net_benefit_intercept: f64,
) -> Result<f64, NumericalInputError> {
    validate_positive_finite(alpha_0, "alpha_0")?;
    validate_positive_finite(beta_0, "beta_0")?;
    validate_positive_finite(alpha_1, "alpha_1")?;
    validate_positive_finite(beta_1, "beta_1")?;
    if !net_benefit_slope.is_finite() || !net_benefit_intercept.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit_slope",
            "net-benefit parameters must be finite",
        ));
    }

    let arm_sizes = sample_size
        .per_arm_sizes()
        .map_err(|_| NumericalInputError::invalid("sample_size", "invalid sample-size spec"))?;
    if arm_sizes.len() != 2 {
        return Err(NumericalInputError::invalid(
            "sample_size",
            "two-arm Beta-Binomial EVSI requires exactly two arms",
        ));
    }
    let n0 = arm_sizes[0];
    let n1 = arm_sizes[1];

    let prior_mean_0 = alpha_0 / (alpha_0 + beta_0);
    let prior_mean_1 = alpha_1 / (alpha_1 + beta_1);
    let current_inb = net_benefit_slope * (prior_mean_1 - prior_mean_0) + net_benefit_intercept;
    let current_value = current_inb.max(0.0);

    let n0_f = f64::from(n0);
    let n1_f = f64::from(n1);
    let mut expected_after = 0.0;

    for y0 in 0..=n0 {
        let p_y0 = beta_binomial_pmf(y0, n0, alpha_0, beta_0);
        if p_y0 <= 0.0 {
            continue;
        }
        let post_p0 = (alpha_0 + f64::from(y0)) / (alpha_0 + beta_0 + n0_f);
        for y1 in 0..=n1 {
            let p_y1 = beta_binomial_pmf(y1, n1, alpha_1, beta_1);
            let post_p1 = (alpha_1 + f64::from(y1)) / (alpha_1 + beta_1 + n1_f);
            let post_inb = net_benefit_slope * (post_p1 - post_p0) + net_benefit_intercept;
            expected_after += p_y0 * p_y1 * post_inb.max(0.0);
        }
    }

    Ok((expected_after - current_value).max(0.0))
}

/// Computes tail-certified conjugate Gamma-Poisson EVSI for a count/rate study.
///
/// The uncertain event rate `lambda ~ Gamma(prior_shape, prior_rate)` is observed over
/// total exposure `exposure_per_unit * total_sample_size`. Incremental net benefit is
/// `net_benefit_slope * lambda + net_benefit_intercept`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for invalid prior, exposure, or sample-size inputs.
pub fn gamma_poisson_evsi(
    prior_shape: f64,
    prior_rate: f64,
    exposure_per_unit: f64,
    sample_size: &SampleSizeConvention,
    net_benefit_slope: f64,
    net_benefit_intercept: f64,
) -> Result<f64, NumericalInputError> {
    validate_positive_finite(prior_shape, "prior_shape")?;
    validate_positive_finite(prior_rate, "prior_rate")?;
    validate_positive_finite(exposure_per_unit, "exposure_per_unit")?;
    if !net_benefit_slope.is_finite() || !net_benefit_intercept.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit_slope",
            "net-benefit parameters must be finite",
        ));
    }

    let total_n = sample_size
        .total_sample_size()
        .map_err(|_| NumericalInputError::invalid("sample_size", "invalid sample-size spec"))?;
    let total_exposure = exposure_per_unit * f64::from(total_n);

    let current_inb = net_benefit_slope * (prior_shape / prior_rate) + net_benefit_intercept;
    let current_value = current_inb.max(0.0);

    if (net_benefit_slope >= 0.0 && net_benefit_intercept >= 0.0)
        || (net_benefit_slope <= 0.0 && net_benefit_intercept <= 0.0)
    {
        return Ok(0.0);
    }

    // Negative Binomial predictive: Y ~ NB(r = prior_shape, p = prior_rate / (prior_rate + E)).
    let p = prior_rate / (prior_rate + total_exposure);
    let ln_p = p.ln();
    let ln_1mp = (1.0 - p).ln();

    let mean_y = prior_shape * total_exposure / prior_rate;
    let var_y = mean_y * (1.0 + total_exposure / prior_rate);
    let sd_y = var_y.sqrt();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let y_start = (mean_y - 12.0 * sd_y).floor().max(0.0) as u64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let y_end = ((mean_y + 25.0 * sd_y).ceil().max(10_000.0)) as u64;
    if y_end - y_start > 20_000_000 {
        return Err(NumericalInputError::invalid(
            "total_sample_size",
            "Gamma-Poisson predictive support exceeds exact summation budget",
        ));
    }

    #[allow(clippy::cast_precision_loss)]
    let y0_f = y_start as f64;
    let mut pmf = (ln_gamma(y0_f + prior_shape) - ln_gamma(y0_f + 1.0) - ln_gamma(prior_shape)
        + prior_shape * ln_p
        + y0_f * ln_1mp)
        .exp();
    let one_minus_p = 1.0 - p;
    let mut cdf = 0.0;
    let mut expected_after = 0.0;
    for y in y_start..=y_end {
        #[allow(clippy::cast_precision_loss)]
        let yf = y as f64;
        let post_mean = (prior_shape + yf) / (prior_rate + total_exposure);
        let post_inb = net_benefit_slope * post_mean + net_benefit_intercept;
        expected_after += pmf * post_inb.max(0.0);
        cdf += pmf;
        if y >= y_start + 32 && cdf >= 1.0 - 1e-11 && yf > mean_y {
            break;
        }
        pmf *= ((yf + prior_shape) / (yf + 1.0)) * one_minus_p;
        if pmf == 0.0 && yf > mean_y {
            break;
        }
    }
    if !cdf.is_finite() || cdf < 1.0 - 1e-9 {
        return Err(NumericalInputError::invalid(
            "total_sample_size",
            "Gamma-Poisson predictive tail mass unresolved",
        ));
    }

    Ok((expected_after - current_value).max(0.0))
}

/// Computes endpoint-aware tanh-sinh quadrature EVSI for an exponential time-to-event study.
///
/// The hazard rate `lambda ~ Gamma(prior_shape, prior_rate)` is observed until
/// `expected_events` (`d > 0`) events occur, yielding total follow-up time `T ~ Gamma(d, lambda)`.
/// Under the Beta-prime predictive distribution for `T / prior_rate`, `U = T / (T + prior_rate)`
/// follows `Beta(d, prior_shape)` and posterior mean hazard is `(prior_shape + d) * (1 - U) / prior_rate`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for invalid prior, event count, net-benefit parameters,
/// or if quadrature mass deviates from `1.0` beyond `1e-6`.
pub fn exponential_time_to_event_evsi(
    prior_shape: f64,
    prior_rate: f64,
    expected_events: f64,
    net_benefit_slope: f64,
    net_benefit_intercept: f64,
) -> Result<f64, NumericalInputError> {
    validate_positive_finite(prior_shape, "prior_shape")?;
    validate_positive_finite(prior_rate, "prior_rate")?;
    validate_positive_finite(expected_events, "expected_events")?;
    if !net_benefit_slope.is_finite() || !net_benefit_intercept.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit_slope",
            "net-benefit parameters must be finite",
        ));
    }

    if (net_benefit_slope >= 0.0 && net_benefit_intercept >= 0.0)
        || (net_benefit_slope <= 0.0 && net_benefit_intercept <= 0.0)
    {
        return Ok(0.0);
    }

    let current_inb = net_benefit_slope * (prior_shape / prior_rate) + net_benefit_intercept;
    let current_value = current_inb.max(0.0);

    // Tanh-sinh quadrature on u in (0, 1) handles Beta(d, alpha) endpoint singularities
    // accurately even when d < 1 or alpha < 1.
    let n_grid: i32 = 512;
    let t_max = 4.5_f64;
    let dt = (2.0 * t_max) / f64::from(n_grid);
    let half_pi = 0.5 * PI;
    let norm_const = ln_beta(expected_events, prior_shape);
    let mut weight_sum = 0.0;
    let mut expected_after = 0.0;

    for idx in 0..n_grid {
        let t = -t_max + (f64::from(idx) + 0.5) * dt;
        let s = half_pi * t.sinh();
        if s.abs() > 35.0 {
            continue;
        }
        let u = f64::midpoint(1.0, s.tanh());
        let one_minus_u = 0.5 * (1.0 - s.tanh());
        if u <= 0.0 || one_minus_u <= 0.0 {
            continue;
        }
        let cosh_s = s.cosh();
        let du_dt = half_pi * t.cosh() / (2.0 * cosh_s * cosh_s);
        let ln_density =
            (expected_events - 1.0) * u.ln() + (prior_shape - 1.0) * one_minus_u.ln() - norm_const;
        let w = ln_density.exp() * du_dt * dt;
        if !w.is_finite() {
            continue;
        }
        let post_mean = (prior_shape + expected_events) * one_minus_u / prior_rate;
        let post_inb = net_benefit_slope * post_mean + net_benefit_intercept;
        expected_after += w * post_inb.max(0.0);
        weight_sum += w;
    }
    if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1e-6 {
        return Err(NumericalInputError::invalid(
            "expected_events",
            "Beta predictive quadrature mass failed accuracy tolerance",
        ));
    }
    Ok((expected_after - current_value).max(0.0))
}
