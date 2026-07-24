use voiage_domain::SampleMatrix;

use crate::NumericalInputError;

/// Expected opportunity-loss summary for a sample-by-strategy value matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedLossKernelResult {
    /// Mean net or generalized benefit for each strategy.
    pub expected_net_benefit_by_strategy: Vec<f64>,
    /// Mean sample-level opportunity loss for each strategy.
    pub expected_opportunity_loss_by_strategy: Vec<f64>,
    /// Lowest-index strategy with the greatest expected net benefit.
    pub optimal_strategy_index: usize,
    /// Expected opportunity loss of the current-information optimum.
    pub minimum_expected_opportunity_loss: f64,
    /// Number of uncertainty samples.
    pub sample_count: usize,
    /// Number of decision strategies.
    pub strategy_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedSum {
    total: f64,
    compensation: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.compensation;
        let next = self.total + corrected;
        self.compensation = (next - self.total) - corrected;
        self.total = next;
    }
}

/// Compute expected opportunity loss for every strategy.
///
/// Opportunity loss is computed sample by sample as the difference between
/// the best available net benefit in that sample and each strategy's net
/// benefit. Exact ties in expected net benefit resolve to the lowest strategy
/// index. The minimum expected opportunity loss therefore equals EVPI for the
/// same matrix, subject only to floating-point rounding.
///
/// # Errors
///
/// Returns [`NumericalInputError`] if an intermediate or result is not finite.
pub fn expected_loss(
    net_benefit: &SampleMatrix,
) -> Result<ExpectedLossKernelResult, NumericalInputError> {
    let [sample_count, strategy_count] = net_benefit.shape();
    let divisor = f64::from(u32::try_from(sample_count).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported numerical range",
        )
    })?);
    let mut benefit_totals = vec![CompensatedSum::default(); strategy_count];
    let mut loss_totals = vec![CompensatedSum::default(); strategy_count];

    for row in net_benefit.rows() {
        let sample_optimum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (strategy, value) in row.iter().copied().enumerate() {
            benefit_totals[strategy].add(value);
            let loss = sample_optimum - value;
            if !loss.is_finite() {
                return Err(NumericalInputError::invalid(
                    "net_benefit",
                    "opportunity-loss difference is not finite",
                ));
            }
            loss_totals[strategy].add(loss);
        }
    }

    let expected_net_benefit_by_strategy = benefit_totals
        .into_iter()
        .map(|sum| sum.total / divisor)
        .collect::<Vec<_>>();
    let expected_opportunity_loss_by_strategy = loss_totals
        .into_iter()
        .map(|sum| sum.total / divisor)
        .collect::<Vec<_>>();
    if expected_net_benefit_by_strategy
        .iter()
        .chain(&expected_opportunity_loss_by_strategy)
        .any(|value| !value.is_finite())
    {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "expected benefit or opportunity loss is not finite",
        ));
    }

    let mut optimal_strategy_index = 0;
    for strategy in 1..strategy_count {
        if expected_net_benefit_by_strategy[strategy]
            > expected_net_benefit_by_strategy[optimal_strategy_index]
        {
            optimal_strategy_index = strategy;
        }
    }
    let minimum_expected_opportunity_loss =
        expected_opportunity_loss_by_strategy[optimal_strategy_index];

    Ok(ExpectedLossKernelResult {
        expected_net_benefit_by_strategy,
        expected_opportunity_loss_by_strategy,
        optimal_strategy_index,
        minimum_expected_opportunity_loss,
        sample_count,
        strategy_count,
    })
}
