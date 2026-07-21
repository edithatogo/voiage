use std::collections::BTreeMap;

use crate::NumericalInputError;

/// Rust-owned subgroup aggregation for value of heterogeneity.
#[derive(Clone, Debug, PartialEq)]
pub struct HeterogeneityKernelResult {
    /// Gain from subgroup-specific strategy selection.
    pub value: f64,
    /// Labels in deterministic lexical order.
    pub subgroup_labels: Vec<String>,
    /// Fraction of samples in each subgroup.
    pub subgroup_weights: Vec<f64>,
    /// Optimal strategy index for each subgroup.
    pub subgroup_optimal_strategy_indices: Vec<usize>,
    /// Expected net benefit of each subgroup-optimal strategy.
    pub subgroup_expected_net_benefits: Vec<f64>,
    /// Optimal strategy index for the pooled population.
    pub overall_optimal_strategy_index: usize,
    /// Expected net benefit of the pooled-population optimum.
    pub overall_expected_net_benefit: f64,
}

/// Compute the gain from subgroup-specific strategy selection.
///
/// # Errors
///
/// Returns [`NumericalInputError`] when inputs are empty, non-rectangular,
/// non-finite, dimensionally inconsistent, or too large for exact sample-count
/// conversion.
pub fn heterogeneity(
    net_benefit: &[Vec<f64>],
    subgroups: &[String],
) -> Result<HeterogeneityKernelResult, NumericalInputError> {
    let strategy_count = net_benefit.first().map_or(0, Vec::len);
    if net_benefit.is_empty() || strategy_count == 0 {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "at least one sample and one strategy are required",
        ));
    }
    if net_benefit.len() != subgroups.len() {
        return Err(NumericalInputError::dimension(
            "subgroups",
            net_benefit.len(),
            subgroups.len(),
            "subgroup count must match sample count",
        ));
    }
    if net_benefit
        .iter()
        .any(|row| row.len() != strategy_count || row.iter().any(|value| !value.is_finite()))
    {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "net-benefit values must be finite and rectangular",
        ));
    }

    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, label) in subgroups.iter().enumerate() {
        if label.is_empty() {
            return Err(NumericalInputError::invalid(
                "subgroups",
                "subgroup labels must not be empty",
            ));
        }
        groups.entry(label.as_str()).or_default().push(index);
    }

    let overall_means =
        means_for_indices(net_benefit, &(0..net_benefit.len()).collect::<Vec<_>>())?;
    let overall_optimal_strategy_index = argmax(&overall_means);
    let overall_expected_net_benefit = overall_means[overall_optimal_strategy_index];
    let mut subgroup_weights = Vec::with_capacity(groups.len());
    let mut subgroup_optimal_strategy_indices = Vec::with_capacity(groups.len());
    let mut subgroup_expected_net_benefits = Vec::with_capacity(groups.len());
    let mut subgroup_specific = 0.0;
    for indices in groups.values() {
        let means = means_for_indices(net_benefit, indices)?;
        let optimal = argmax(&means);
        let weight = sample_count(indices.len())? / sample_count(net_benefit.len())?;
        subgroup_weights.push(weight);
        subgroup_optimal_strategy_indices.push(optimal);
        subgroup_expected_net_benefits.push(means[optimal]);
        subgroup_specific += weight * means[optimal];
    }

    Ok(HeterogeneityKernelResult {
        value: (subgroup_specific - overall_expected_net_benefit).max(0.0),
        subgroup_labels: groups.keys().map(|label| (*label).to_owned()).collect(),
        subgroup_weights,
        subgroup_optimal_strategy_indices,
        subgroup_expected_net_benefits,
        overall_optimal_strategy_index,
        overall_expected_net_benefit,
    })
}

fn means_for_indices(
    net_benefit: &[Vec<f64>],
    indices: &[usize],
) -> Result<Vec<f64>, NumericalInputError> {
    let mut means = vec![0.0; net_benefit[0].len()];
    for &index in indices {
        for (strategy, value) in net_benefit[index].iter().enumerate() {
            means[strategy] += value;
        }
    }
    let count = sample_count(indices.len())?;
    for value in &mut means {
        *value /= count;
    }
    Ok(means)
}

fn sample_count(count: usize) -> Result<f64, NumericalInputError> {
    u32::try_from(count).map(f64::from).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "sample count exceeds the supported exact range",
        )
    })
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::heterogeneity;

    #[test]
    fn subgroup_aggregation_matches_python_contract() {
        let result = heterogeneity(
            &[
                vec![10.0, 8.0],
                vec![11.0, 7.0],
                vec![6.0, 12.0],
                vec![5.0, 13.0],
            ],
            &["low".into(), "low".into(), "high".into(), "high".into()],
        )
        .expect("valid heterogeneity input");
        assert_eq!(result.subgroup_labels, ["high", "low"]);
        assert_eq!(result.subgroup_optimal_strategy_indices, [1, 0]);
        assert_eq!(result.overall_optimal_strategy_index, 1);
        assert!((result.value - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        assert!(heterogeneity(&[vec![1.0]], &[]).is_err());
        assert!(heterogeneity(&[vec![1.0], vec![f64::NAN]], &["a".into(), "a".into()]).is_err());
    }
}
