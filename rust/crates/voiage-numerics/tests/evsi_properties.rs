//! Property contracts for seeded-bootstrap EVSI.

use proptest::prelude::*;
use voiage_domain::SampleMatrix;
use voiage_numerics::evsi_stochastic;

fn matrix(values: Vec<Vec<f64>>) -> SampleMatrix {
    values.try_into().expect("generated matrix is valid")
}

proptest! {
    #[test]
    fn seeded_evsi_is_reproducible_and_non_negative(
        raw in prop::collection::vec(prop::array::uniform3(-100_i16..=100), 1..=24),
        trial_size in 1_usize..=32,
        resample_count in 1_usize..=16,
        seed in any::<u64>(),
    ) {
        let values = raw
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let input = matrix(values);

        let first = evsi_stochastic(&input, trial_size, resample_count, seed)
            .expect("generated EVSI input is valid");
        let second = evsi_stochastic(&input, trial_size, resample_count, seed)
            .expect("same EVSI input is valid");

        prop_assert_eq!(&first, &second);
        prop_assert!(first.evsi >= 0.0);
    }

    #[test]
    fn one_strategy_has_zero_evsi(
        raw in prop::collection::vec(-100_i16..=100, 1..=24),
    ) {
        let input = matrix(raw.iter().map(|value| vec![f64::from(*value)]).collect());

        let result = evsi_stochastic(&input, 4, 8, 42).expect("generated EVSI input");

        prop_assert!((result.evsi).abs() <= 1.0e-12);
    }
}
