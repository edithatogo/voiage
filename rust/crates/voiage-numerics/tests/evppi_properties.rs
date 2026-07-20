//! Property contracts for regression-based EVPPI.

use proptest::prelude::*;
use voiage_domain::SampleMatrix;
use voiage_numerics::evppi;

fn matrix(values: Vec<Vec<f64>>) -> SampleMatrix {
    values.try_into().expect("generated matrix is valid")
}

proptest! {
    #[test]
    fn evppi_is_non_negative_and_translation_invariant(
        raw in prop::collection::vec(prop::array::uniform3(-100_i16..=100), 3..=24),
        translation in -1_000_i16..=1_000,
    ) {
        let values = raw
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let translated = values
            .iter()
            .map(|row| row.iter().map(|value| value + f64::from(translation)).collect())
            .collect::<Vec<Vec<_>>>();
        let parameters = (0..values.len())
            .map(|index| vec![f64::from(u32::try_from(index).expect("small generated index"))])
            .collect::<Vec<_>>();

        let baseline = evppi(&matrix(values), &matrix(parameters.clone())).expect("baseline EVPPI");
        let shifted = evppi(&matrix(translated), &matrix(parameters)).expect("shifted EVPPI");

        prop_assert!(baseline >= 0.0);
        prop_assert!((baseline - shifted).abs() <= 1.0e-8);
    }

    #[test]
    fn evppi_scales_with_positive_factors(
        raw in prop::collection::vec(prop::array::uniform3(-100_i16..=100), 3..=24),
        factor in 1_u8..=20,
    ) {
        let values = raw
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let scaled = values
            .iter()
            .map(|row| row.iter().map(|value| value * f64::from(factor)).collect())
            .collect::<Vec<Vec<_>>>();
        let parameters = (0..values.len())
            .map(|index| vec![f64::from(u32::try_from(index).expect("small generated index"))])
            .collect::<Vec<_>>();

        let baseline = evppi(&matrix(values), &matrix(parameters.clone())).expect("baseline EVPPI");
        let scaled_result = evppi(&matrix(scaled), &matrix(parameters)).expect("scaled EVPPI");

        prop_assert!((scaled_result - baseline * f64::from(factor)).abs() <= 1.0e-8);
    }
}
