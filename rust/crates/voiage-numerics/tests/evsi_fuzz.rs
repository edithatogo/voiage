//! Bounded generated-input fuzz coverage for the stable EVSI kernels.

use proptest::prelude::*;
use voiage_domain::SampleMatrix;
use voiage_numerics::{evsi_efficient_linear, evsi_moment_based, evsi_stochastic};

fn matrix(values: Vec<Vec<f64>>) -> SampleMatrix {
    values
        .try_into()
        .expect("generated EVSI matrix is rectangular and non-empty")
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    #[test]
    fn generated_finite_inputs_are_panic_free_finite_and_reproducible(
        rows in prop::collection::vec(prop::array::uniform2(-64_i16..=64), 3..=12),
        trial_sample_size in 1_usize..=12,
        resample_count in 1_usize..=8,
    ) {
        let net_benefit = matrix(rows
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect());
        let parameters = matrix(rows
            .iter()
            .enumerate()
            // Use a deterministic full-rank basis independent of the generated
            // net-benefit values.  The moment kernel deliberately rejects
            // rank-deficient designs, so deriving a column from `row[0]` would
            // make the fuzz assertion invalid for constant generated values.
            .map(|(index, _row)| {
                let index = f64::from(
                    u32::try_from(index).expect("bounded generated row index fits in u32"),
                );
                vec![index, index * index]
            })
            .collect());

        let seeded = evsi_stochastic(
            &net_benefit,
            trial_sample_size,
            resample_count,
            42,
        ).expect("generated seeded-bootstrap input must be accepted");
        prop_assert!(seeded.expected_current_value.is_finite());
        prop_assert!(seeded.expected_sample_value.is_finite());
        prop_assert!(seeded.expected_perfect_information.is_finite());
        prop_assert!(seeded.evsi.is_finite());
        prop_assert!(seeded.evsi >= 0.0);
        prop_assert_eq!(seeded, evsi_stochastic(
            &net_benefit,
            trial_sample_size,
            resample_count,
            42,
        ).expect("repeated generated seeded-bootstrap input must be accepted"));

        let deterministic_results = [
            evsi_efficient_linear(&net_benefit, &parameters, trial_sample_size),
            evsi_moment_based(&net_benefit, &parameters, trial_sample_size),
        ];
        // Rank-deficient designs are valid rejected inputs, especially when
        // the generated sample is smaller than the moment design basis.
        // Assert invariants for every accepted result without turning a
        // documented numerical input error into a fuzz-test panic.
        for result in deterministic_results.into_iter().flatten() {
            prop_assert!(result.expected_current_value.is_finite());
            prop_assert!(result.expected_sample_value.is_finite());
            prop_assert!(result.expected_perfect_information.is_finite());
            prop_assert!(result.evsi.is_finite());
            prop_assert!(result.evsi >= 0.0);
        }
    }
}
