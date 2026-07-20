//! Property contracts for deterministic foundational numerical kernels.

use proptest::prelude::*;
use voiage_domain::SampleMatrix;
use voiage_numerics::{enbs, evpi};

fn matrix(values: Vec<Vec<f64>>) -> SampleMatrix {
    values.try_into().expect("generated matrix is valid")
}

proptest! {
    #[test]
    fn evpi_is_non_negative_and_translation_invariant(
        raw in prop::collection::vec(prop::array::uniform3(-1_000_i16..=1_000), 1..=32),
        translation in -1_000_i16..=1_000,
    ) {
        let values = raw
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| f64::from(*value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let translated = values
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| value + f64::from(translation))
                    .collect()
            })
            .collect::<Vec<_>>();

        let baseline = evpi(&matrix(values)).expect("baseline EVPI");
        let shifted = evpi(&matrix(translated)).expect("translated EVPI");

        prop_assert!(baseline >= 0.0);
        prop_assert!((baseline - shifted).abs() <= 1.0e-10);
    }

    #[test]
    fn evpi_scales_with_positive_factors(
        raw in prop::collection::vec(prop::array::uniform3(-100_i16..=100), 1..=32),
        factor in 1_u8..=20,
    ) {
        let values = raw
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| f64::from(*value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scaled = values
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| value * f64::from(factor))
                    .collect()
            })
            .collect::<Vec<_>>();

        let baseline = evpi(&matrix(values)).expect("baseline EVPI");
        let scaled_result = evpi(&matrix(scaled)).expect("scaled EVPI");

        prop_assert!((scaled_result - baseline * f64::from(factor)).abs() <= 1.0e-10);
    }

    #[test]
    fn raw_enbs_is_exact_subtraction_for_finite_non_negative_costs(
        evsi in -1_000_i16..=1_000,
        cost in 0_u16..=1_000,
    ) {
        prop_assert_eq!(
            enbs(f64::from(evsi), f64::from(cost)),
            Ok(f64::from(evsi) - f64::from(cost)),
        );
    }
}
