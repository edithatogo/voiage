//! Property contracts for expected opportunity loss.

use proptest::prelude::*;
use voiage_domain::SampleMatrix;
use voiage_numerics::{evpi, expected_loss};

fn matrix(values: Vec<Vec<f64>>) -> SampleMatrix {
    values.try_into().expect("generated matrix is valid")
}

proptest! {
    #[test]
    fn minimum_expected_loss_equals_evpi_and_is_translation_invariant(
        raw in prop::collection::vec(prop::array::uniform3(-1_000_i16..=1_000), 1..=32),
        translation in -1_000_i16..=1_000,
    ) {
        let values = raw
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let translated = values
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| value + f64::from(translation))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let baseline_matrix = matrix(values);
        let baseline = expected_loss(&baseline_matrix).expect("baseline expected loss");
        let shifted = expected_loss(&matrix(translated)).expect("translated expected loss");
        let perfect_information = evpi(&baseline_matrix).expect("EVPI");

        prop_assert!(
            (baseline.minimum_expected_opportunity_loss - perfect_information).abs()
                <= 1.0e-10
        );
        for (left, right) in baseline
            .expected_opportunity_loss_by_strategy
            .iter()
            .zip(&shifted.expected_opportunity_loss_by_strategy)
        {
            prop_assert!((left - right).abs() <= 1.0e-10);
        }
    }
}
