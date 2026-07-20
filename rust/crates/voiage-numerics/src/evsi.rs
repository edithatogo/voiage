use voiage_domain::SampleMatrix;

use crate::NumericalInputError;

const MIX64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Approximate seeded-bootstrap EVSI summary.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiKernelResult {
    /// Stable estimator identifier.
    pub estimator: &'static str,
    /// Version of the seeded-bootstrap numerical contract.
    pub contract_version: u32,
    /// Expected value under current information.
    pub expected_current_value: f64,
    /// Expected value after the seeded bootstrap study summary.
    pub expected_sample_value: f64,
    /// Expected value with perfect information.
    pub expected_perfect_information: f64,
    /// Non-negative expected value of sample information.
    pub evsi: f64,
    /// Number of rows drawn per bootstrap resample.
    pub draw_count: usize,
    /// Number of bootstrap resamples performed.
    pub resample_count: usize,
}

/// Computes the production seeded-bootstrap EVSI summary.
///
/// This is the bounded Rust estimator corresponding to the existing
/// `seeded-bootstrap-summary` contract, using the versioned `xorshift64-star-v1`
/// algorithm for indexed
/// resampling. Trial-design parsing and reporting
/// remain outside this binding-independent numerical kernel.
///
/// # Errors
///
/// Returns an input error for zero loop sizes, unsupported sample-count
/// magnitudes, non-finite totals, or non-finite output.
pub fn evsi_stochastic(
    net_benefit: &SampleMatrix,
    trial_sample_size: usize,
    resample_count: usize,
    seed: u64,
) -> Result<EvsiKernelResult, NumericalInputError> {
    if trial_sample_size == 0 {
        return Err(NumericalInputError::invalid(
            "trial_sample_size",
            "trial sample size must be positive",
        ));
    }
    if resample_count == 0 {
        return Err(NumericalInputError::invalid(
            "resample_count",
            "resample count must be positive",
        ));
    }
    let [sample_count, strategy_count] = net_benefit.shape();
    let draw_count = trial_sample_size.min(sample_count);
    let divisor = f64::from(u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported numerical range",
        )
    })?);
    let expected_current_value = strategy_means(net_benefit, divisor, strategy_count)?
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let expected_perfect_information = row_max_mean(net_benefit, divisor);
    if !expected_perfect_information.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "EVSI perfect-information total is not finite",
        ));
    }
    if strategy_count == 1 {
        return Ok(EvsiKernelResult {
            estimator: "seeded_bootstrap",
            contract_version: 1,
            expected_current_value,
            expected_sample_value: expected_current_value,
            expected_perfect_information,
            evsi: 0.0,
            draw_count,
            resample_count,
        });
    }
    let expected_sample_value = bootstrap_expected_value(
        net_benefit,
        draw_count,
        resample_count,
        seed,
        strategy_count,
    )?;
    let evsi = (expected_sample_value - expected_current_value).max(0.0);
    Ok(EvsiKernelResult {
        estimator: "seeded_bootstrap",
        contract_version: 1,
        expected_current_value,
        expected_sample_value,
        expected_perfect_information,
        evsi,
        draw_count,
        resample_count,
    })
}

fn strategy_means(
    matrix: &SampleMatrix,
    divisor: f64,
    strategy_count: usize,
) -> Result<Vec<f64>, NumericalInputError> {
    let mut means = vec![0.0; strategy_count];
    for (row_index, row) in matrix.rows().enumerate() {
        let count = f64::from(u32::try_from(row_index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "net_benefit",
                "sample count exceeds the supported numerical range",
            )
        })?);
        for (index, value) in row.iter().copied().enumerate() {
            let updated = means[index] + (value - means[index]) / count;
            if !updated.is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "EVSI strategy total is not finite",
                ));
            }
            means[index] = updated;
        }
    }
    let _ = divisor;
    Ok(means)
}

fn row_max_mean(matrix: &SampleMatrix, divisor: f64) -> f64 {
    let mut mean = 0.0;
    for (index, row) in matrix.rows().enumerate() {
        let count = f64::from(u32::try_from(index + 1).unwrap_or(u32::MAX));
        let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        mean += (row_max - mean) / count;
    }
    let _ = divisor;
    mean
}

fn next_index(state: &mut u64, sample_count: usize) -> usize {
    if *state == 0 {
        *state = MIX64;
    }
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let mixed = (*state).wrapping_mul(0x2545_F491_4F6C_DD1D);
    let modulus = u64::try_from(sample_count).expect("usize fits in u64");
    usize::try_from(mixed % modulus).expect("modulo is bounded by sample count")
}

fn bootstrap_expected_value(
    matrix: &SampleMatrix,
    draw_count: usize,
    resample_count: usize,
    seed: u64,
    strategy_count: usize,
) -> Result<f64, NumericalInputError> {
    let _draw_divisor = f64::from(u32::try_from(draw_count).map_err(|_| {
        NumericalInputError::invalid(
            "trial_sample_size",
            "draw count exceeds the supported numerical range",
        )
    })?);
    let _resample_divisor = f64::from(u32::try_from(resample_count).map_err(|_| {
        NumericalInputError::invalid(
            "resample_count",
            "resample count exceeds the supported numerical range",
        )
    })?);
    let mut bootstrap_mean = 0.0;
    for resample_index in 0..resample_count {
        let mut means = vec![0.0; strategy_count];
        let mut state = seed ^ ((resample_index as u64 + 1).wrapping_mul(MIX64));
        for (draw_index, _) in (0..draw_count).enumerate() {
            let draw_number = f64::from(u32::try_from(draw_index + 1).map_err(|_| {
                NumericalInputError::invalid(
                    "trial_sample_size",
                    "draw count exceeds the supported numerical range",
                )
            })?);
            let index = next_index(&mut state, matrix.shape()[0]);
            let row = matrix.row(index).expect("validated sample matrix row");
            for (strategy_index, value) in row.iter().copied().enumerate() {
                let updated = means[strategy_index] + (value - means[strategy_index]) / draw_number;
                if !updated.is_finite() {
                    return Err(NumericalInputError::invalid(
                        "net_benefit",
                        "EVSI bootstrap total is not finite",
                    ));
                }
                means[strategy_index] = updated;
            }
        }
        let best = means.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let resample_number = f64::from(u32::try_from(resample_index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "resample_count",
                "resample count exceeds the supported numerical range",
            )
        })?);
        bootstrap_mean += (best - bootstrap_mean) / resample_number;
    }
    let result = bootstrap_mean;
    if !result.is_finite() {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "EVSI bootstrap result is not finite",
        ));
    }
    Ok(result)
}
