use voiage_domain::{SampleCube, SampleVector};

use crate::NumericalInputError;

/// Structural VOI estimate with sample-average uncertainty metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralVoiKernelResult {
    /// Non-negative structural VOI estimate.
    pub value: f64,
    /// Number of structures.
    pub structure_count: usize,
    /// Number of uncertainty samples per structure.
    pub sample_count: usize,
    /// Number of strategies.
    pub strategy_count: usize,
    /// Unbiased variance of the sample-level informed value.
    pub informed_value_variance: Option<f64>,
    /// Monte Carlo standard error of the VOI estimate.
    pub monte_carlo_standard_error: Option<f64>,
}

/// Computes structural EVPI after model evaluators have produced one plane per structure.
///
/// # Errors
///
/// Returns [`NumericalInputError`] when probability dimensions or values are
/// invalid, or when the sample count exceeds the supported exact range.
pub fn structural_evpi(
    net_benefit_by_structure: &SampleCube,
    structure_probabilities: &SampleVector,
) -> Result<f64, NumericalInputError> {
    structural_evpi_with_assurance(net_benefit_by_structure, structure_probabilities)
        .map(|result| result.value)
}

/// Computes structural EVPI with sample-average uncertainty metadata.
///
/// # Errors
///
/// Returns the same probability, dimension, and range errors as
/// [`structural_evpi`].
pub fn structural_evpi_with_assurance(
    net_benefit_by_structure: &SampleCube,
    structure_probabilities: &SampleVector,
) -> Result<StructuralVoiKernelResult, NumericalInputError> {
    let [structure_count, sample_count, strategy_count] = net_benefit_by_structure.shape();
    let sample_divisor = exact_sample_count(sample_count)?;
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
    let mut informed_values = vec![0.0; sample_count];
    for (structure_index, plane) in net_benefit_by_structure.planes().enumerate() {
        let mut means = vec![0.0; strategy_count];
        let mut perfect_for_structure = 0.0;
        for (sample_index, row) in plane.iter().enumerate() {
            let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            perfect_for_structure += row_max;
            informed_values[sample_index] += probabilities[structure_index] * row_max;
            for (strategy, value) in row.iter().enumerate() {
                means[strategy] += value;
            }
        }
        for (strategy, mean) in means.iter_mut().enumerate() {
            *mean /= sample_divisor;
            pooled_means[strategy] += probabilities[structure_index] * *mean;
        }
        perfect_information +=
            probabilities[structure_index] * perfect_for_structure / sample_divisor;
    }
    let pooled_optimum = pooled_means.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let (informed_value_variance, monte_carlo_standard_error) = sampling_error(&informed_values)?;
    Ok(StructuralVoiKernelResult {
        value: (perfect_information - pooled_optimum).max(0.0),
        structure_count,
        sample_count,
        strategy_count,
        informed_value_variance,
        monte_carlo_standard_error,
    })
}

/// Computes structural EVPPI for the structures treated as known.
///
/// # Errors
///
/// Returns [`NumericalInputError`] when probability dimensions or values,
/// requested structure indices, or the sample count are invalid.
pub fn structural_evppi(
    net_benefit_by_structure: &SampleCube,
    structure_probabilities: &SampleVector,
    structures_of_interest: &[usize],
) -> Result<f64, NumericalInputError> {
    structural_evppi_with_assurance(
        net_benefit_by_structure,
        structure_probabilities,
        structures_of_interest,
    )
    .map(|result| result.value)
}

/// Computes structural EVPPI with sample-average uncertainty metadata.
///
/// # Errors
///
/// Returns the same probability, dimension, index, and range errors as
/// [`structural_evppi`].
pub fn structural_evppi_with_assurance(
    net_benefit_by_structure: &SampleCube,
    structure_probabilities: &SampleVector,
    structures_of_interest: &[usize],
) -> Result<StructuralVoiKernelResult, NumericalInputError> {
    let [structure_count, sample_count, strategy_count] = net_benefit_by_structure.shape();
    let sample_divisor = exact_sample_count(sample_count)?;
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
        return Ok(StructuralVoiKernelResult {
            value: 0.0,
            structure_count,
            sample_count,
            strategy_count,
            informed_value_variance: (sample_count >= 2).then_some(0.0),
            monte_carlo_standard_error: (sample_count >= 2).then_some(0.0),
        });
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
        return Ok(StructuralVoiKernelResult {
            value: 0.0,
            structure_count,
            sample_count,
            strategy_count,
            informed_value_variance: (sample_count >= 2).then_some(0.0),
            monte_carlo_standard_error: (sample_count >= 2).then_some(0.0),
        });
    }
    let mut term1 = 0.0;
    let mut weighted_means = vec![0.0; strategy_count];
    let mut informed_values = vec![0.0; sample_count];
    for index in structures_of_interest {
        let Some(plane) = net_benefit_by_structure.planes().nth(*index) else {
            return Err(NumericalInputError::invalid(
                "structures_of_interest",
                "indices must be valid structure indices",
            ));
        };
        let mut means = vec![0.0; strategy_count];
        let mut max_mean = 0.0;
        let weight = probabilities[*index] / known_probability;
        for (sample_index, row) in plane.iter().enumerate() {
            let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max_mean += row_max;
            informed_values[sample_index] += weight * row_max;
            for (strategy, value) in row.iter().enumerate() {
                means[strategy] += value;
            }
        }
        term1 += weight * max_mean / sample_divisor;
        for strategy in 0..strategy_count {
            weighted_means[strategy] += weight * means[strategy] / sample_divisor;
        }
    }
    let current_value = weighted_means.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let (informed_value_variance, monte_carlo_standard_error) = sampling_error(&informed_values)?;
    Ok(StructuralVoiKernelResult {
        value: (term1 - current_value).max(0.0),
        structure_count,
        sample_count,
        strategy_count,
        informed_value_variance,
        monte_carlo_standard_error,
    })
}

fn sampling_error(values: &[f64]) -> Result<(Option<f64>, Option<f64>), NumericalInputError> {
    if values.len() < 2 {
        return Ok((None, None));
    }
    let count = f64::from(u32::try_from(values.len()).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit_by_structure",
            "sample count exceeds the supported exact range",
        )
    })?);
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (index, value) in values.iter().copied().enumerate() {
        let observed = f64::from(u32::try_from(index + 1).map_err(|_| {
            NumericalInputError::invalid(
                "net_benefit_by_structure",
                "sample count exceeds the supported exact range",
            )
        })?);
        let delta = value - mean;
        mean += delta / observed;
        m2 += delta * (value - mean);
    }
    let variance = (m2 / (count - 1.0)).max(0.0);
    let standard_error = (variance / count).sqrt();
    if variance.is_finite() && standard_error.is_finite() {
        Ok((Some(variance), Some(standard_error)))
    } else {
        Err(NumericalInputError::invalid(
            "net_benefit_by_structure",
            "structural VOI sampling error is not finite",
        ))
    }
}

fn exact_sample_count(count: usize) -> Result<f64, NumericalInputError> {
    u32::try_from(count).map(f64::from).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit_by_structure",
            "sample count exceeds the supported exact range",
        )
    })
}

#[cfg(test)]
mod tests {
    use voiage_domain::{SampleCube, SampleVector};

    use super::{
        structural_evpi, structural_evpi_with_assurance, structural_evppi,
        structural_evppi_with_assurance,
    };

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

    #[test]
    fn structural_assurance_reports_sample_error_and_dimensions() {
        let cube = SampleCube::try_from(vec![
            vec![vec![10.0, 8.0], vec![11.0, 7.0]],
            vec![vec![6.0, 12.0], vec![5.0, 13.0]],
        ])
        .unwrap();
        let probabilities = SampleVector::try_from(vec![0.5, 0.5]).unwrap();

        let perfect = structural_evpi_with_assurance(&cube, &probabilities).unwrap();
        let partial = structural_evppi_with_assurance(&cube, &probabilities, &[0, 1]).unwrap();

        assert!((perfect.value - structural_evpi(&cube, &probabilities).unwrap()).abs() < 1.0e-12);
        assert!(
            (partial.value - structural_evppi(&cube, &probabilities, &[0, 1]).unwrap()).abs()
                < 1.0e-12
        );
        assert_eq!(perfect.sample_count, 2);
        assert_eq!(perfect.structure_count, 2);
        assert_eq!(perfect.strategy_count, 2);
        assert!(perfect.monte_carlo_standard_error.is_some());
        assert!(partial.monte_carlo_standard_error.is_some());
    }
}
