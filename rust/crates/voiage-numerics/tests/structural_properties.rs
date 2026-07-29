//! Metamorphic contracts for structural VOI aggregation.

use proptest::prelude::*;
use voiage_domain::{SampleCube, SampleVector};
use voiage_numerics::{structural_evpi, structural_evppi};

fn cube(values: Vec<Vec<Vec<f64>>>) -> SampleCube {
    values.try_into().expect("generated cube is valid")
}

fn probabilities() -> SampleVector {
    vec![0.25, 0.75]
        .try_into()
        .expect("fixed probabilities are valid")
}

proptest! {
    #[test]
    fn structural_values_are_translation_invariant_and_positive_homogeneous(
        raw in prop::collection::vec(
            prop::array::uniform2(prop::array::uniform2(-100_i16..=100)),
            1..=24,
        ),
        translation in -100_i16..=100,
        factor in 1_u8..=10,
    ) {
        let values = (0..2)
            .map(|structure| {
                raw.iter()
                    .map(|sample| {
                        sample[structure]
                            .iter()
                            .map(|value| f64::from(*value))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let translated = values
            .iter()
            .map(|plane| {
                plane
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|value| value + f64::from(translation))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scaled = values
            .iter()
            .map(|plane| {
                plane
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|value| value * f64::from(factor))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights = probabilities();

        let evpi = structural_evpi(&cube(values.clone()), &weights).expect("baseline EVPI");
        let evppi = structural_evppi(&cube(values), &weights, &[0, 1]).expect("baseline EVPPI");
        let translated_evpi =
            structural_evpi(&cube(translated.clone()), &weights).expect("translated EVPI");
        let translated_evppi =
            structural_evppi(&cube(translated), &weights, &[0, 1]).expect("translated EVPPI");
        let scaled_evpi = structural_evpi(&cube(scaled.clone()), &weights).expect("scaled EVPI");
        let scaled_evppi =
            structural_evppi(&cube(scaled), &weights, &[0, 1]).expect("scaled EVPPI");

        prop_assert!((translated_evpi - evpi).abs() <= 1.0e-10);
        prop_assert!((translated_evppi - evppi).abs() <= 1.0e-10);
        prop_assert!((scaled_evpi - evpi * f64::from(factor)).abs() <= 1.0e-8);
        prop_assert!((scaled_evppi - evppi * f64::from(factor)).abs() <= 1.0e-8);
    }
}
