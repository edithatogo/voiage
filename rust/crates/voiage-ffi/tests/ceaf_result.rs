//! Contract tests for threshold-aligned CEAF C ABI results.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_ceaf_result, VoiageCeafResultV1, VoiageStatusV1, VOIAGE_ABI_CEAF_RESULT,
};

fn empty_result() -> VoiageCeafResultV1 {
    VoiageCeafResultV1 {
        struct_size: 0,
        struct_version: 0,
        sample_count: 0,
        strategy_count: 0,
        threshold_count: 0,
    }
}

#[test]
fn ceaf_result_writes_threshold_aligned_outputs_and_assurance() {
    let values = [10.0, 1.0, 5.0, 8.0, 2.0, 3.0, 7.0, 4.0];
    let thresholds = [0.0, 100.0];
    let mut optimal = [u64::MAX; 2];
    let mut probability = [f64::NAN; 2];
    let mut lower = [f64::NAN; 2];
    let mut upper = [f64::NAN; 2];
    let mut expected_net_benefit = [f64::NAN; 2];
    let mut has_assurance = [0_u32; 2];
    let mut variance = [f64::NAN; 2];
    let mut standard_error = [f64::NAN; 2];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_ceaf_result(
            values.as_ptr(),
            2,
            2,
            2,
            thresholds.as_ptr(),
            0.95,
            optimal.as_mut_ptr(),
            probability.as_mut_ptr(),
            lower.as_mut_ptr(),
            upper.as_mut_ptr(),
            expected_net_benefit.as_mut_ptr(),
            has_assurance.as_mut_ptr(),
            variance.as_mut_ptr(),
            standard_error.as_mut_ptr(),
            2,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(optimal, [0, 1]);
    assert!((probability[0] - 0.5).abs() < f64::EPSILON);
    assert!((probability[1] - 1.0).abs() < f64::EPSILON);
    assert_eq!(has_assurance, [1, 1]);
    assert!((variance[0] - 0.5).abs() < f64::EPSILON);
    assert!((standard_error[0] - 0.5).abs() < f64::EPSILON);
    assert!((expected_net_benefit[0] - 6.0).abs() < f64::EPSILON);
    assert!((expected_net_benefit[1] - 6.0).abs() < f64::EPSILON);
    assert!(lower.iter().all(|value| (0.0..=1.0).contains(value)));
    assert!(upper.iter().all(|value| (0.0..=1.0).contains(value)));
    assert_eq!(result.struct_size, 32);
    assert_eq!(result.struct_version, 1);
    assert_eq!(result.sample_count, 2);
    assert_eq!(result.strategy_count, 2);
    assert_eq!(result.threshold_count, 2);
}

#[test]
fn ceaf_result_rejects_short_capacity_without_partial_writes() {
    let values = [10.0, 1.0, 5.0, 8.0, 2.0, 3.0, 7.0, 4.0];
    let thresholds = [0.0, 100.0];
    let mut optimal = [u64::MAX; 2];
    let mut probability = [101.0_f64; 2];
    let mut lower = [102.0_f64; 2];
    let mut upper = [103.0_f64; 2];
    let mut expected_net_benefit = [104.0_f64; 2];
    let mut has_assurance = [9_u32; 2];
    let mut variance = [105.0_f64; 2];
    let mut standard_error = [106.0_f64; 2];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_ceaf_result(
            values.as_ptr(),
            2,
            2,
            2,
            thresholds.as_ptr(),
            0.95,
            optimal.as_mut_ptr(),
            probability.as_mut_ptr(),
            lower.as_mut_ptr(),
            upper.as_mut_ptr(),
            expected_net_benefit.as_mut_ptr(),
            has_assurance.as_mut_ptr(),
            variance.as_mut_ptr(),
            standard_error.as_mut_ptr(),
            1,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::BufferTooSmall);
    assert_eq!(optimal, [u64::MAX; 2]);
    assert_eq!(has_assurance, [9; 2]);
    assert_eq!(result, empty_result());
    assert_eq!(probability[0].to_bits(), 101.0_f64.to_bits());
    assert_eq!(lower[0].to_bits(), 102.0_f64.to_bits());
    assert_eq!(upper[0].to_bits(), 103.0_f64.to_bits());
    assert_eq!(expected_net_benefit[0].to_bits(), 104.0_f64.to_bits());
    assert_eq!(variance[0].to_bits(), 105.0_f64.to_bits());
    assert_eq!(standard_error[0].to_bits(), 106.0_f64.to_bits());
}

#[test]
fn ceaf_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_CEAF_RESULT, 0);
}
