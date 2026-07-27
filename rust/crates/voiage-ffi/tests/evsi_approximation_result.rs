//! Contract tests for promoted Rust-native EVSI approximation results.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_evsi_moment_matching_result, voiage_v1_evsi_regression_result,
    VoiageEvsiApproximationResultV1, VoiageStatusV1, VOIAGE_ABI_EVSI_APPROXIMATION_RESULT,
    VOIAGE_EVSI_ASSURANCE_INCOMPLETE, VOIAGE_EVSI_ESTIMATOR_MOMENT_MATCHING,
    VOIAGE_EVSI_ESTIMATOR_REGRESSION,
};

fn empty_result() -> VoiageEvsiApproximationResultV1 {
    VoiageEvsiApproximationResultV1 {
        struct_size: 0,
        struct_version: 0,
        evsi: 0.0,
        expected_current_value: 0.0,
        expected_sample_value: 0.0,
        expected_perfect_information: 0.0,
        information_fraction: 0.0,
        sample_count: 0,
        strategy_count: 0,
        parameter_count: 0,
        trial_sample_size: 0,
        estimator_kind: u32::MAX,
        assurance_state: u32::MAX,
    }
}

#[test]
fn regression_and_moment_results_report_distinct_estimator_kinds() {
    let net_benefit = [5.0, 1.0, 4.0, 2.0, 1.0, 5.0, 2.0, 4.0];
    let parameters = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];

    for (function, expected_kind) in [
        (
            voiage_v1_evsi_regression_result
                as unsafe extern "C" fn(
                    *const f64,
                    u64,
                    u64,
                    *const f64,
                    u64,
                    u64,
                    u64,
                    *mut VoiageEvsiApproximationResultV1,
                ) -> VoiageStatusV1,
            VOIAGE_EVSI_ESTIMATOR_REGRESSION,
        ),
        (
            voiage_v1_evsi_moment_matching_result,
            VOIAGE_EVSI_ESTIMATOR_MOMENT_MATCHING,
        ),
    ] {
        let mut result = empty_result();
        let status = unsafe {
            function(
                net_benefit.as_ptr(),
                4,
                2,
                parameters.as_ptr(),
                4,
                2,
                3,
                &raw mut result,
            )
        };

        assert_eq!(status, VoiageStatusV1::Ok);
        assert_eq!(result.struct_size, 88);
        assert_eq!(result.struct_version, 1);
        assert!(result.evsi.is_finite());
        assert!(result.evsi >= 0.0);
        assert!(result.expected_sample_value >= result.expected_current_value);
        assert!(result.expected_sample_value <= result.expected_perfect_information);
        assert_eq!(result.sample_count, 4);
        assert_eq!(result.strategy_count, 2);
        assert_eq!(result.parameter_count, 2);
        assert_eq!(result.trial_sample_size, 3);
        assert_eq!(result.estimator_kind, expected_kind);
        assert_eq!(result.assurance_state, VOIAGE_EVSI_ASSURANCE_INCOMPLETE);
    }
}

#[test]
fn evsi_approximation_rejects_mismatched_rows_without_writing() {
    let net_benefit = [5.0, 1.0, 4.0, 2.0];
    let parameters = [0.0, 1.0, 2.0, 3.0];
    let sentinel = VoiageEvsiApproximationResultV1 {
        evsi: 73.0,
        ..empty_result()
    };
    let mut result = sentinel;

    let status = unsafe {
        voiage_v1_evsi_regression_result(
            net_benefit.as_ptr(),
            2,
            2,
            parameters.as_ptr(),
            4,
            1,
            3,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::DimensionMismatch);
    assert_eq!(result, sentinel);
}

#[test]
fn evsi_approximation_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_EVSI_APPROXIMATION_RESULT, 0);
}
