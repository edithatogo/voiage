//! Contract tests for structural VOI C ABI results.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_structural_evpi_result, voiage_v1_structural_evppi_result, VoiageStatusV1,
    VoiageStructuralVoiResultV1, VOIAGE_ABI_STRUCTURAL_VOI_RESULT,
};

fn empty_result() -> VoiageStructuralVoiResultV1 {
    VoiageStructuralVoiResultV1 {
        struct_size: 0,
        struct_version: 0,
        value: 0.0,
        structure_count: 0,
        sample_count: 0,
        strategy_count: 0,
        has_assurance: 0,
        reserved: 0,
        informed_value_variance: 0.0,
        monte_carlo_standard_error: 0.0,
    }
}

#[test]
fn structural_evpi_result_reports_dimensions_value_and_assurance() {
    let values = [10.0, 8.0, 11.0, 7.0, 6.0, 12.0, 5.0, 13.0];
    let probabilities = [0.5, 0.5];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_structural_evpi_result(
            values.as_ptr(),
            2,
            2,
            2,
            probabilities.as_ptr(),
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(result.struct_size, 64);
    assert_eq!(result.struct_version, 1);
    assert!((result.value - 1.5).abs() < f64::EPSILON);
    assert_eq!(result.structure_count, 2);
    assert_eq!(result.sample_count, 2);
    assert_eq!(result.strategy_count, 2);
    assert_eq!(result.has_assurance, 1);
    assert!(result.informed_value_variance.is_finite());
    assert!(result.monte_carlo_standard_error.is_finite());
}

#[test]
fn structural_evppi_result_accepts_selected_structure_indices() {
    let values = [10.0, 8.0, 11.0, 7.0, 6.0, 12.0, 5.0, 13.0];
    let probabilities = [0.5, 0.5];
    let selected = [0_u64, 1_u64];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_structural_evppi_result(
            values.as_ptr(),
            2,
            2,
            2,
            probabilities.as_ptr(),
            selected.as_ptr(),
            selected.len() as u64,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert!((result.value - 1.5).abs() < f64::EPSILON);
    assert_eq!(result.has_assurance, 1);
}

#[test]
fn structural_result_failures_do_not_write_output() {
    let values = [10.0, 8.0, 11.0, 7.0, 6.0, 12.0, 5.0, 13.0];
    let invalid_probabilities = [0.8, 0.8];
    let sentinel = VoiageStructuralVoiResultV1 {
        value: 73.0,
        ..empty_result()
    };
    let mut result = sentinel;

    let status = unsafe {
        voiage_v1_structural_evpi_result(
            values.as_ptr(),
            2,
            2,
            2,
            invalid_probabilities.as_ptr(),
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::InvalidArgument);
    assert_eq!(result, sentinel);
}

#[test]
fn structural_evppi_supports_empty_selection_and_rejects_invalid_indices() {
    let values = [10.0, 8.0, 11.0, 7.0, 6.0, 12.0, 5.0, 13.0];
    let probabilities = [0.5, 0.5];
    let mut result = empty_result();

    let empty_status = unsafe {
        voiage_v1_structural_evppi_result(
            values.as_ptr(),
            2,
            2,
            2,
            probabilities.as_ptr(),
            std::ptr::null(),
            0,
            &raw mut result,
        )
    };
    assert_eq!(empty_status, VoiageStatusV1::Ok);
    assert!(result.value.abs() < f64::EPSILON);
    assert_eq!(result.has_assurance, 1);

    let invalid_index = [2_u64];
    let sentinel = VoiageStructuralVoiResultV1 {
        value: 73.0,
        ..empty_result()
    };
    result = sentinel;
    let invalid_status = unsafe {
        voiage_v1_structural_evppi_result(
            values.as_ptr(),
            2,
            2,
            2,
            probabilities.as_ptr(),
            invalid_index.as_ptr(),
            1,
            &raw mut result,
        )
    };
    assert_eq!(invalid_status, VoiageStatusV1::InvalidArgument);
    assert_eq!(result, sentinel);
}

#[test]
fn structural_result_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_STRUCTURAL_VOI_RESULT, 0);
}
