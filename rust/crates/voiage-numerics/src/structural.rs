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

/// Computes structural EVPPI for the structures treated as known.
pub fn structural_evppi(
    net_benefit_by_structure: &SampleCube,
    structure_probabilities: &SampleVector,
    structures_of_interest: &[usize],
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
    if structures_of_interest
        .iter()
        .any(|index| *index >= structure_count)
    {
        return Err(NumericalInputError::invalid(
            "structures_of_interest",
            "indices must be valid structure indices",
        ));
    }
    if structures_of_interest.is_empty() {
        return Ok(0.0);
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
    let known_probability: f64 = structures_of_interest
        .iter()
        .map(|i| probabilities[*i])
        .sum();
    if known_probability == 0.0 {
        return Ok(0.0);
    }
    let mut term1 = 0.0;
    let mut weighted_means = vec![0.0; strategy_count];
    for index in structures_of_interest {
        let plane = net_benefit_by_structure
            .planes()
            .nth(*index)
            .expect("validated structure index");
        let mut means = vec![0.0; strategy_count];
        let mut max_mean = 0.0;
        for row in plane {
            max_mean += row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for (strategy, value) in row.iter().enumerate() {
                means[strategy] += value;
            }
        }
        let weight = probabilities[*index] / known_probability;
        term1 += weight * max_mean / sample_count as f64;
        for strategy in 0..strategy_count {
            weighted_means[strategy] += weight * means[strategy] / sample_count as f64;
        }
    }
    Ok((term1 - weighted_means.into_iter().fold(f64::NEG_INFINITY, f64::max)).max(0.0))
}

#[cfg(test)]
mod tests {
    use voiage_domain::{SampleCube, SampleVector};

    use super::{structural_evpi, structural_evppi};

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

    #[test]
    fn structural_evppi_matches_known_structure_contract() {
        let cube = SampleCube::try_from(vec![
            vec![vec![10.0, 8.0], vec![11.0, 7.0]],
            vec![vec![6.0, 12.0], vec![5.0, 13.0]],
        ])
        .unwrap();
        let probabilities = SampleVector::try_from(vec![0.5, 0.5]).unwrap();
        assert!(structural_evppi(&cube, &probabilities, &[0]).unwrap().abs() < 1.0e-12);
    }
}
