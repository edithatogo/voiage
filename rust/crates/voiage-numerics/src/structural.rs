use voiage_domain::{SampleCube, SampleVector};

use crate::NumericalInputError;

/// Computes structural EVPI after model evaluators have produced one plane per structure.
pub fn structural_evpi(
    net_benefit_by_structure: &SampleCube,
    structure_probabilities: &SampleVector,
) -> Result<f64, NumericalInputError> {
    let [structure_count, sample_count, strategy_count] = net_benefit_by_structure.shape();
    if structure_probabilities.len() != structure_count {
        return Err(NumericalInputError::dimension(
            "structure_probabilities",
            structure_count,
            structure_probabilities.len(),
            "structure probability count must match structure count",
        ));
    }
    let probabilities = structure_probabilities.as_slice();
    if probabilities.iter().any(|probability| *probability < 0.0)
        || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1.0e-12
    {
        return Err(NumericalInputError::invalid(
            "structure_probabilities",
            "structure probabilities must be non-negative and sum to one",
        ));
    }

    let mut perfect_information = 0.0;
    let mut pooled_means = vec![0.0; strategy_count];
    for (structure_index, plane) in net_benefit_by_structure.planes().enumerate() {
        let mut means = vec![0.0; strategy_count];
        let mut perfect_for_structure = 0.0;
        for row in plane {
            let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            perfect_for_structure += row_max;
            for (strategy, value) in row.iter().enumerate() {
                means[strategy] += value;
            }
        }
        for (strategy, mean) in means.iter_mut().enumerate() {
            *mean /= sample_count as f64;
            pooled_means[strategy] += probabilities[structure_index] * *mean;
        }
        perfect_information +=
            probabilities[structure_index] * perfect_for_structure / sample_count as f64;
    }
    let pooled_optimum = pooled_means.into_iter().fold(f64::NEG_INFINITY, f64::max);
    Ok((perfect_information - pooled_optimum).max(0.0))
}

#[cfg(test)]
mod tests {
    use voiage_domain::{SampleCube, SampleVector};

    use super::structural_evpi;

    #[test]
    fn structural_evpi_matches_python_aggregation_contract() {
        let cube = SampleCube::try_from(vec![
            vec![vec![10.0, 8.0], vec![11.0, 7.0]],
            vec![vec![6.0, 12.0], vec![5.0, 13.0]],
        ])
        .unwrap();
        let probabilities = SampleVector::try_from(vec![0.5, 0.5]).unwrap();
        assert!((structural_evpi(&cube, &probabilities).unwrap() - 1.5).abs() < 1.0e-12);
    }
}
