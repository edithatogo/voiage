//! Property contracts for CEAF ordering and probability invariants.

use proptest::prelude::*;
use voiage_domain::{SampleCube, SampleVector};
use voiage_numerics::ceaf;

fn cube(values: Vec<Vec<Vec<f64>>>) -> SampleCube {
    values.try_into().expect("generated cube is valid")
}

fn vector(values: Vec<f64>) -> SampleVector {
    values.try_into().expect("generated vector is valid")
}

proptest! {
    #[test]
    fn threshold_permutation_permutates_every_output(
        raw in prop::collection::vec(prop::array::uniform3(-100_i16..=100), 1..=12),
    ) {
        let values = raw.iter().map(|row| {
            row.iter().map(|value| {
                vec![f64::from(*value), f64::from(*value + 3), f64::from(*value - 2)]
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let permuted = values.iter().map(|row| {
            row.iter().map(|thresholds| vec![thresholds[2], thresholds[0], thresholds[1]]).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let thresholds = vector(vec![10.0, 20.0, 30.0]);
        let permuted_thresholds = vector(vec![30.0, 10.0, 20.0]);
        let baseline = ceaf(&cube(values), &thresholds, 0.95).expect("baseline CEAF");
        let shifted = ceaf(&cube(permuted), &permuted_thresholds, 0.95).expect("permuted CEAF");

        prop_assert_eq!(shifted.wtp_thresholds, vec![30.0, 10.0, 20.0]);
        prop_assert_eq!(shifted.optimal_strategy_indices, vec![baseline.optimal_strategy_indices[2], baseline.optimal_strategy_indices[0], baseline.optimal_strategy_indices[1]]);
        prop_assert_eq!(shifted.acceptability_probabilities, vec![baseline.acceptability_probabilities[2], baseline.acceptability_probabilities[0], baseline.acceptability_probabilities[1]]);
        prop_assert_eq!(shifted.expected_net_benefit, vec![baseline.expected_net_benefit[2], baseline.expected_net_benefit[0], baseline.expected_net_benefit[1]]);
    }

    #[test]
    fn probabilities_are_sample_fractions(
        raw in prop::collection::vec(prop::array::uniform2(-100_i16..=100), 1..=32),
    ) {
        let values = raw.iter().map(|row| {
            row.iter().map(|value| vec![f64::from(*value)]).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let samples = f64::from(u32::try_from(raw.len()).expect("bounded generated samples"));
        let result = ceaf(&cube(values), &vector(vec![0.0]), 0.95).expect("CEAF");
        let fraction = result.acceptability_probabilities[0] * samples;

        prop_assert!((fraction - fraction.round()).abs() <= 1.0e-12);
        prop_assert!(result.probability_lower[0] <= result.acceptability_probabilities[0]);
        prop_assert!(result.acceptability_probabilities[0] <= result.probability_upper[0]);
    }
}
