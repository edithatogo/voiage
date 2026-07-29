use voiage_domain::SampleMatrix;

use crate::{expected_loss, NumericalInputError};

/// EVPI estimate with sample-average uncertainty metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct EvpiKernelResult {
    /// Non-negative expected value of perfect information.
    pub value: f64,
    /// Number of uncertainty samples.
    pub sample_count: usize,
    /// Number of strategies.
    pub strategy_count: usize,
    /// Unbiased variance of selected-strategy opportunity loss.
    pub opportunity_loss_variance: Option<f64>,
    /// Monte Carlo standard error of EVPI.
    pub monte_carlo_standard_error: Option<f64>,
}

/// Computes expected value of perfect information from `[sample][strategy]`
/// net-benefit values.
///
/// # Errors
///
/// The matrix is validated by [`SampleMatrix`]. This result-bearing signature
/// preserves a uniform numerical-kernel boundary for future checked failures.
pub fn evpi(net_benefit: &SampleMatrix) -> Result<f64, NumericalInputError> {
    evpi_with_assurance(net_benefit).map(|result| result.value)
}

/// Computes EVPI together with sample-average uncertainty metadata.
///
/// # Errors
///
/// Returns an input error when the shared expected-loss kernel cannot produce
/// a finite result.
pub fn evpi_with_assurance(
    net_benefit: &SampleMatrix,
) -> Result<EvpiKernelResult, NumericalInputError> {
    let result = expected_loss(net_benefit)?;
    Ok(EvpiKernelResult {
        value: result.minimum_expected_opportunity_loss,
        sample_count: result.sample_count,
        strategy_count: result.strategy_count,
        opportunity_loss_variance: result.opportunity_loss_variance,
        monte_carlo_standard_error: result.monte_carlo_standard_error,
    })
}
