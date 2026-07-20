use voiage_domain::SampleMatrix;

use crate::NumericalInputError;

/// Computes expected value of perfect information from `[sample][strategy]`
/// net-benefit values.
///
/// # Errors
///
/// The matrix is validated by [`SampleMatrix`]. This result-bearing signature
/// preserves a uniform numerical-kernel boundary for future checked failures.
pub fn evpi(net_benefit: &SampleMatrix) -> Result<f64, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let mut strategy_totals = vec![0.0; strategy_count];
    let mut perfect_information_total = 0.0;

    for row in net_benefit.rows() {
        let mut row_maximum = f64::NEG_INFINITY;
        for (strategy_index, value) in row.iter().copied().enumerate() {
            strategy_totals[strategy_index] += value;
            row_maximum = row_maximum.max(value);
        }
        perfect_information_total += row_maximum;
    }

    let divisor = f64::from(u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported numerical range",
        )
    })?);
    let perfect_information_mean = perfect_information_total / divisor;
    let current_information_mean = strategy_totals
        .into_iter()
        .map(|total| total / divisor)
        .fold(f64::NEG_INFINITY, f64::max);

    Ok((perfect_information_mean - current_information_mean).max(0.0))
}
