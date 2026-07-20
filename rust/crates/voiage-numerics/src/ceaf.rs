use voiage_domain::{SampleCube, SampleVector};

use crate::NumericalInputError;

/// Binding-independent cost-effectiveness acceptability frontier result.
#[derive(Clone, Debug, PartialEq)]
pub struct CeafKernelResult {
    /// Willingness-to-pay labels in original order.
    pub wtp_thresholds: Vec<f64>,
    /// Expected-optimal original strategy index at each threshold.
    pub optimal_strategy_indices: Vec<usize>,
    /// Probability that the expected-optimal strategy is sample-optimal.
    pub acceptability_probabilities: Vec<f64>,
    /// Lower bound of the two-sided Wald probability interval.
    pub probability_lower: Vec<f64>,
    /// Upper bound of the two-sided Wald probability interval.
    pub probability_upper: Vec<f64>,
    /// Expected net benefit of the selected strategy at each threshold.
    pub expected_net_benefit: Vec<f64>,
}

/// Computes a cost-effectiveness acceptability frontier.
///
/// Ties select the lowest original strategy index. Threshold values are labels
/// and are preserved in input order.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for a threshold dimension mismatch or a
/// confidence level outside the finite open unit interval. Cube and vector
/// finiteness are enforced by the domain contracts.
pub fn ceaf(
    net_benefit: &SampleCube,
    wtp_thresholds: &SampleVector,
    confidence_level: f64,
) -> Result<CeafKernelResult, NumericalInputError> {
    if !confidence_level.is_finite() || confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(NumericalInputError::invalid(
            "confidence_level",
            "confidence level must be finite and strictly between zero and one",
        ));
    }
    let [sample_count, strategy_count, threshold_count] = net_benefit.shape();
    if wtp_thresholds.len() != threshold_count {
        return Err(NumericalInputError::dimension(
            "wtp_thresholds",
            threshold_count,
            wtp_thresholds.len(),
            "threshold count must match the net-benefit cube",
        ));
    }
    let sample_divisor = f64::from(u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported numerical range",
        )
    })?);
    let planes = net_benefit.planes().collect::<Vec<_>>();
    let z_value = inverse_standard_normal(0.5 + confidence_level / 2.0);
    let mut optimal_strategy_indices = Vec::with_capacity(threshold_count);
    let mut acceptability_probabilities = Vec::with_capacity(threshold_count);
    let mut probability_lower = Vec::with_capacity(threshold_count);
    let mut probability_upper = Vec::with_capacity(threshold_count);
    let mut expected_net_benefit = Vec::with_capacity(threshold_count);

    for threshold in 0..threshold_count {
        let mut means = vec![0.0; strategy_count];
        for plane in &planes {
            for strategy in 0..strategy_count {
                means[strategy] += plane[strategy][threshold];
            }
        }
        for mean in &mut means {
            *mean /= sample_divisor;
        }
        let expected_optimal = first_argmax(&means);
        let matching_samples = planes
            .iter()
            .filter(|plane| first_argmax_at_threshold(plane, threshold) == expected_optimal)
            .count();
        let probability = f64::from(u32::try_from(matching_samples).map_err(|_| {
            NumericalInputError::invalid(
                "net_benefit",
                "matching sample count exceeds the supported numerical range",
            )
        })?) / sample_divisor;
        let standard_error = (probability * (1.0 - probability) / sample_divisor).sqrt();

        optimal_strategy_indices.push(expected_optimal);
        acceptability_probabilities.push(probability);
        probability_lower.push((probability - z_value * standard_error).clamp(0.0, 1.0));
        probability_upper.push((probability + z_value * standard_error).clamp(0.0, 1.0));
        expected_net_benefit.push(means[expected_optimal]);
    }

    Ok(CeafKernelResult {
        wtp_thresholds: wtp_thresholds.as_slice().to_vec(),
        optimal_strategy_indices,
        acceptability_probabilities,
        probability_lower,
        probability_upper,
        expected_net_benefit,
    })
}

fn first_argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn first_argmax_at_threshold(strategies: &[Vec<f64>], threshold: usize) -> usize {
    let mut best = 0;
    for index in 1..strategies.len() {
        if strategies[index][threshold] > strategies[best][threshold] {
            best = index;
        }
    }
    best
}

#[allow(clippy::excessive_precision)]
fn inverse_standard_normal(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOWER: f64 = 0.024_25;
    const UPPER: f64 = 1.0 - LOWER;

    if probability < LOWER {
        let q = (-2.0 * probability.ln()).sqrt();
        polynomial(&C, q) / (polynomial(&D, q) * q + 1.0)
    } else if probability <= UPPER {
        let q = probability - 0.5;
        let r = q * q;
        polynomial(&A, r) * q / (polynomial(&B, r) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -polynomial(&C, q) / (polynomial(&D, q) * q + 1.0)
    }
}

fn polynomial(coefficients: &[f64], value: f64) -> f64 {
    coefficients
        .iter()
        .copied()
        .reduce(|result, coefficient| result * value + coefficient)
        .unwrap_or(0.0)
}
