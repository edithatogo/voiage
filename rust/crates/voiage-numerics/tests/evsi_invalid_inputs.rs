//! Panic-free and stable-error properties for malformed native EVSI inputs.

use std::panic::{catch_unwind, AssertUnwindSafe};

use proptest::prelude::*;
use voiage_diagnostics::ErrorCode;
use voiage_domain::SampleMatrix;
use voiage_numerics::{
    evsi_efficient_linear, evsi_moment_based, evsi_stochastic, NumericalInputError,
};

fn matrix(values: Vec<Vec<f64>>) -> SampleMatrix {
    values
        .try_into()
        .expect("generated matrix is finite and rectangular")
}

fn stable_error<T>(call: impl Fn() -> Result<T, NumericalInputError>) -> NumericalInputError {
    let Err(first) =
        catch_unwind(AssertUnwindSafe(&call)).expect("malformed EVSI input must not panic")
    else {
        panic!("malformed EVSI input unexpectedly succeeded")
    };
    let Err(second) =
        catch_unwind(AssertUnwindSafe(call)).expect("repeated malformed EVSI input must not panic")
    else {
        panic!("repeated malformed EVSI input unexpectedly succeeded")
    };

    assert_eq!(first, second, "malformed-input errors must be stable");
    first
}

proptest! {
    #[test]
    fn seeded_bootstrap_rejects_invalid_loop_sizes_without_panicking(
        raw in prop::collection::vec(prop::array::uniform2(-32_i16..=32), 1..=16),
        invalid_trial_size in Just(0_usize),
        invalid_resample_count in Just(0_usize),
    ) {
        let input = matrix(raw
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect());

        let trial_error = stable_error(|| {
            evsi_stochastic(&input, invalid_trial_size, 1, 42).map(|_| ())
        });
        prop_assert_eq!(trial_error.code(), ErrorCode::InvalidInput);
        prop_assert_eq!(
            trial_error.record().details().expect("invalid errors have details").field(),
            Some("trial_sample_size")
        );

        let resample_error = stable_error(|| {
            evsi_stochastic(&input, 1, invalid_resample_count, 42).map(|_| ())
        });
        prop_assert_eq!(resample_error.code(), ErrorCode::InvalidInput);
        prop_assert_eq!(
            resample_error.record().details().expect("invalid errors have details").field(),
            Some("resample_count")
        );
    }

    #[test]
    fn deterministic_estimators_reject_shape_mismatches_without_panicking(
        raw in prop::collection::vec(prop::array::uniform2(-16_i16..=16), 2..=12),
        parameter_rows in prop::collection::vec(prop::array::uniform1(-16_i16..=16), 1..=12),
    ) {
        prop_assume!(raw.len() != parameter_rows.len());
        let net_benefit = matrix(raw
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect());
        let parameters = matrix(parameter_rows
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect());

        let efficient_error = stable_error(|| {
            evsi_efficient_linear(&net_benefit, &parameters, 1).map(|_| ())
        });
        prop_assert_eq!(efficient_error.code(), ErrorCode::DimensionMismatch);

        let moment_error = stable_error(|| {
            evsi_moment_based(&net_benefit, &parameters, 1).map(|_| ())
        });
        prop_assert_eq!(moment_error.code(), ErrorCode::DimensionMismatch);
    }

    #[test]
    fn deterministic_estimators_handle_rank_deficiency_without_panicking(
        raw in prop::collection::vec(prop::array::uniform2(-16_i16..=16), 1..=12),
        constant_parameter in -16_i16..=16,
    ) {
        let net_benefit = matrix(raw
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect());
        let parameters = matrix((0..net_benefit.shape()[0])
            .map(|_| vec![f64::from(constant_parameter)])
            .collect());

        let efficient_result = evsi_efficient_linear(&net_benefit, &parameters, 1);
        prop_assert!(efficient_result.is_ok());
        prop_assert!(efficient_result.unwrap().expected_sample_value.is_finite());

        let moment_result = evsi_moment_based(&net_benefit, &parameters, 1);
        prop_assert!(moment_result.is_ok());
        prop_assert!(moment_result.unwrap().expected_sample_value.is_finite());
    }
}
