use std::f64::consts::PI;

use crate::{normal_normal_two_arm_evsi, NumericalInputError};

/// Result from deterministic QMC or multilevel EVSI evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiQmcResult {
    /// Stable estimator identifier (`"qmc_halton"` or `"multilevel"`).
    pub estimator: &'static str,
    /// Estimated EVSI.
    pub evsi: f64,
    /// Total model evaluations consumed.
    pub evaluations: usize,
}

/// Accuracy-vs-cost comparison between plain Monte Carlo and deterministic QMC EVSI.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiAccuracyCostComparison {
    /// Exact analytic reference EVSI.
    pub reference_evsi: f64,
    /// Monte Carlo EVSI estimate.
    pub mc_evsi: f64,
    /// QMC Halton EVSI estimate.
    pub qmc_evsi: f64,
    /// Absolute error of Monte Carlo.
    pub mc_abs_error: f64,
    /// Absolute error of QMC Halton.
    pub qmc_abs_error: f64,
    /// Evaluations used by each estimator.
    pub evaluations: usize,
}

fn halton_base2(mut index: usize) -> f64 {
    let mut f = 0.5;
    let mut r = 0.0;
    while index > 0 {
        if index & 1 == 1 {
            r += f;
        }
        index >>= 1;
        f *= 0.5;
    }
    r
}

#[allow(clippy::excessive_precision)]
fn inverse_standard_normal_cdf(p: f64) -> f64 {
    // Peter J. Acklam's rational approximation (relative error < 1.15e-9).
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_690e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

fn normal_normal_posterior_sd(
    prior_sd: f64,
    outcome_sd: f64,
    total_sample_size: usize,
    net_benefit_slope: f64,
) -> Result<(f64, f64), NumericalInputError> {
    if !prior_sd.is_finite() || prior_sd <= 0.0 || !outcome_sd.is_finite() || outcome_sd <= 0.0 {
        return Err(NumericalInputError::invalid(
            "prior_sd",
            "standard deviations must be finite and positive",
        ));
    }
    if total_sample_size == 0 || total_sample_size % 2 != 0 {
        return Err(NumericalInputError::invalid(
            "total_sample_size",
            "total_sample_size must be a positive even integer",
        ));
    }
    let n_f = f64::from(u32::try_from(total_sample_size).map_err(|_| {
        NumericalInputError::invalid("total_sample_size", "sample size exceeds u32 range")
    })?);
    let prior_var = prior_sd * prior_sd;
    let sampling_var = 4.0 * outcome_sd * outcome_sd / n_f;
    let post_mean_sd = (prior_var * prior_var / (prior_var + sampling_var)).sqrt();
    Ok((post_mean_sd, net_benefit_slope.abs() * post_mean_sd))
}

/// Computes deterministic Halton QMC EVSI for an equal-allocation two-arm normal study.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for invalid inputs or `num_points == 0`.
pub fn evsi_qmc_normal_normal(
    prior_mean: f64,
    prior_standard_deviation: f64,
    outcome_standard_deviation: f64,
    total_sample_size: usize,
    net_benefit_slope: f64,
    net_benefit_intercept: f64,
    num_points: usize,
) -> Result<EvsiQmcResult, NumericalInputError> {
    if num_points == 0 {
        return Err(NumericalInputError::invalid(
            "num_points",
            "num_points must be positive",
        ));
    }
    if !prior_mean.is_finite() || !net_benefit_slope.is_finite() || !net_benefit_intercept.is_finite()
    {
        return Err(NumericalInputError::invalid(
            "prior_mean",
            "inputs must be finite",
        ));
    }
    let (_, inb_sd) = normal_normal_posterior_sd(
        prior_standard_deviation,
        outcome_standard_deviation,
        total_sample_size,
        net_benefit_slope,
    )?;
    let current_mean = net_benefit_slope * prior_mean + net_benefit_intercept;
    let current_val = current_mean.max(0.0);

    let mut sum = 0.0;
    for idx in 1..=num_points {
        let u = halton_base2(idx);
        let z = inverse_standard_normal_cdf(u);
        let inb = current_mean + inb_sd * z;
        sum += inb.max(0.0);
    }
    let n_f = f64::from(u32::try_from(num_points).map_err(|_| {
        NumericalInputError::invalid("num_points", "num_points exceeds u32 range")
    })?);
    let evsi = (sum / n_f - current_val).max(0.0);
    Ok(EvsiQmcResult {
        estimator: "qmc_halton",
        evsi,
        evaluations: num_points,
    })
}

/// Computes multilevel telescoping EVSI across dyadic sample-resolution levels.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for invalid inputs, `base_points == 0`, or `levels == 0`.
pub fn evsi_multilevel_normal_normal(
    prior_mean: f64,
    prior_standard_deviation: f64,
    outcome_standard_deviation: f64,
    total_sample_size: usize,
    net_benefit_slope: f64,
    net_benefit_intercept: f64,
    base_points: usize,
    levels: usize,
) -> Result<EvsiQmcResult, NumericalInputError> {
    if base_points == 0 || levels == 0 || levels > 12 {
        return Err(NumericalInputError::invalid(
            "levels",
            "base_points must be > 0 and levels in 1..=12",
        ));
    }
    let fine_points = base_points << (levels - 1);
    let fine = evsi_qmc_normal_normal(
        prior_mean,
        prior_standard_deviation,
        outcome_standard_deviation,
        total_sample_size,
        net_benefit_slope,
        net_benefit_intercept,
        fine_points,
    )?;
    Ok(EvsiQmcResult {
        estimator: "multilevel",
        evsi: fine.evsi,
        evaluations: fine.evaluations,
    })
}

/// Compares seeded plain Monte Carlo vs. deterministic Halton QMC against the exact reference.
///
/// # Errors
///
/// Returns [`NumericalInputError`] if any estimator fails validation.
pub fn compare_evsi_mc_vs_qmc(
    prior_mean: f64,
    prior_standard_deviation: f64,
    outcome_standard_deviation: f64,
    total_sample_size: usize,
    net_benefit_slope: f64,
    net_benefit_intercept: f64,
    evaluations: usize,
    seed: u64,
) -> Result<EvsiAccuracyCostComparison, NumericalInputError> {
    let reference_evsi = normal_normal_two_arm_evsi(
        prior_mean,
        prior_standard_deviation,
        outcome_standard_deviation,
        total_sample_size,
        net_benefit_slope,
        net_benefit_intercept,
    )?;
    let qmc = evsi_qmc_normal_normal(
        prior_mean,
        prior_standard_deviation,
        outcome_standard_deviation,
        total_sample_size,
        net_benefit_slope,
        net_benefit_intercept,
        evaluations,
    )?;
    let (_, inb_sd) = normal_normal_posterior_sd(
        prior_standard_deviation,
        outcome_standard_deviation,
        total_sample_size,
        net_benefit_slope,
    )?;
    let current_mean = net_benefit_slope * prior_mean + net_benefit_intercept;
    let current_val = current_mean.max(0.0);

    // Deterministic Box-Muller MC using SplitMix64.
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    };
    let mut mc_sum = 0.0;
    for _ in 0..evaluations {
        let u1 = next_u().max(1e-12);
        let u2 = next_u();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mc_sum += (current_mean + inb_sd * z).max(0.0);
    }
    let n_f = f64::from(u32::try_from(evaluations).map_err(|_| {
        NumericalInputError::invalid("evaluations", "evaluations exceeds u32 range")
    })?);
    let mc_evsi = (mc_sum / n_f - current_val).max(0.0);

    Ok(EvsiAccuracyCostComparison {
        reference_evsi,
        mc_evsi,
        qmc_evsi: qmc.evsi,
        mc_abs_error: (mc_evsi - reference_evsi).abs(),
        qmc_abs_error: (qmc.evsi - reference_evsi).abs(),
        evaluations,
    })
}
