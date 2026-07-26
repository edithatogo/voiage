//! Contract tests for the typed expected-loss C ABI result.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_expected_loss_result, VoiageExpectedLossResultV1, VoiageStatusV1,
    VOIAGE_ABI_EXPECTED_LOSS_RESULT,
};

fn empty_result() -> VoiageExpectedLossResultV1 {
    VoiageExpectedLossResultV1 {
        struct_size: 0,
        struct_version: 0,
        optimal_strategy_index: 0,
        sample_count: 0,
        strategy_count: 0,
        minimum_expected_opportunity_loss: 0.0,
        has_assurance: 0,
        reserved: 1,
        opportunity_loss_variance: 0.0,
        monte_carlo_standard_error: 0.0,
    }
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(left, right)| (left - right).abs() < f64::EPSILON));
}

#[test]
fn expected_loss_result_writes_summary_and_caller_owned_arrays() {
    let values = [10.0, 1.0, 2.0, 8.0];
    let mut benefits = [0.0; 2];
    let mut losses = [0.0; 2];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_expected_loss_result(
            values.as_ptr(),
            2,
            2,
            benefits.as_mut_ptr(),
            losses.as_mut_ptr(),
            2,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_close(&benefits, &[6.0, 4.5]);
    assert_close(&losses, &[3.0, 4.5]);
    assert_eq!(result.struct_size, 64);
    assert_eq!(result.struct_version, 1);
    assert_eq!(result.optimal_strategy_index, 0);
    assert_eq!(result.sample_count, 2);
    assert_eq!(result.strategy_count, 2);
    assert!((result.minimum_expected_opportunity_loss - 3.0).abs() < f64::EPSILON);
    assert_eq!(result.has_assurance, 1);
    assert_eq!(result.reserved, 0);
    assert!((result.opportunity_loss_variance - 18.0).abs() < f64::EPSILON);
    assert!((result.monte_carlo_standard_error - 3.0).abs() < f64::EPSILON);
}

#[test]
fn expected_loss_result_rejects_short_arrays_without_partial_writes() {
    let values = [10.0, 1.0, 2.0, 8.0];
    let mut benefits = [101.0; 2];
    let mut losses = [202.0; 2];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_expected_loss_result(
            values.as_ptr(),
            2,
            2,
            benefits.as_mut_ptr(),
            losses.as_mut_ptr(),
            1,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::BufferTooSmall);
    assert!(benefits
        .iter()
        .all(|value| value.to_bits() == 101.0_f64.to_bits()));
    assert!(losses
        .iter()
        .all(|value| value.to_bits() == 202.0_f64.to_bits()));
    assert_eq!(result, empty_result());
}

#[test]
fn expected_loss_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_EXPECTED_LOSS_RESULT, 0);
}

#[test]
#[allow(clippy::cast_ptr_alignment)]
fn expected_loss_result_rejects_misaligned_input_without_dereferencing() {
    let storage = [0_u8; 40];
    let values = unsafe { storage.as_ptr().add(1).cast::<f64>() };
    let mut benefits = [0.0; 2];
    let mut losses = [0.0; 2];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_expected_loss_result(
            values,
            2,
            2,
            benefits.as_mut_ptr(),
            losses.as_mut_ptr(),
            2,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::InvalidArgument);
}
