//! Contract tests for the stable regression EVPPI C ABI result.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_evppi_regression_result, VoiageEvppiRegressionResultV1, VoiageStatusV1,
    VOIAGE_ABI_EVPPI_REGRESSION_RESULT,
};

fn empty_result() -> VoiageEvppiRegressionResultV1 {
    VoiageEvppiRegressionResultV1 {
        struct_size: 0,
        struct_version: 0,
        value: 0.0,
        sample_count: 0,
        strategy_count: 0,
        parameter_count: 0,
        assurance_state: u32::MAX,
        reserved: u32::MAX,
    }
}

#[test]
fn regression_evppi_result_reports_value_fit_dimensions_and_assurance_state() {
    let net_benefit = [5.0, 1.0, 4.0, 2.0, 1.0, 5.0, 2.0, 4.0];
    let parameters = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_evppi_regression_result(
            net_benefit.as_ptr(),
            4,
            2,
            parameters.as_ptr(),
            4,
            2,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(result.struct_size, 48);
    assert_eq!(result.struct_version, 1);
    assert!(result.value.is_finite());
    assert!(result.value >= 0.0);
    assert_eq!(result.sample_count, 4);
    assert_eq!(result.strategy_count, 2);
    assert_eq!(result.parameter_count, 2);
    assert_eq!(result.assurance_state, 0);
    assert_eq!(result.reserved, 0);
}

#[test]
fn regression_evppi_rejects_invalid_inputs_without_writing() {
    let net_benefit = [5.0, 1.0, 4.0, 2.0];
    let parameters = [0.0, 1.0, 2.0, 3.0];
    let sentinel = VoiageEvppiRegressionResultV1 {
        value: 73.0,
        ..empty_result()
    };
    let mut result = sentinel;

    let status = unsafe {
        voiage_v1_evppi_regression_result(
            net_benefit.as_ptr(),
            2,
            2,
            parameters.as_ptr(),
            4,
            1,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::DimensionMismatch);
    assert_eq!(result, sentinel);
}

#[test]
fn regression_evppi_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_EVPPI_REGRESSION_RESULT, 0);
}
