//! Contract tests for Rust-authoritative ENBS through the C ABI.

#![allow(unsafe_code)]

use voiage_ffi::{voiage_v1_enbs, voiage_v1_enbs_r, VoiageStatusV1, VOIAGE_ABI_ENBS};

#[test]
fn enbs_returns_raw_net_value_without_clipping() {
    let mut result = f64::NAN;
    let status = unsafe { voiage_v1_enbs(12.5, 3.0, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert!((result - 9.5).abs() < f64::EPSILON);

    let status = unsafe { voiage_v1_enbs(2.0, 3.0, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert!((result + 1.0).abs() < f64::EPSILON);
}

#[test]
fn enbs_r_adapter_executes_successfully() {
    let evsi = 12.5_f64;
    let cost = 3.0_f64;
    let mut out_value = 0.0_f64;
    let mut out_status = -1_i32;

    unsafe {
        voiage_v1_enbs_r(
            &raw const evsi,
            &raw const cost,
            &raw mut out_value,
            &raw mut out_status,
        );
    }
    assert_eq!(out_status, 0);
    assert!((out_value - 9.5).abs() < f64::EPSILON);
}

#[test]
fn enbs_r_adapter_handles_invalid_inputs_safely() {
    let evsi = f64::NAN;
    let cost = 3.0_f64;
    let mut out_value = 42.0_f64;
    let mut out_status = -1_i32;

    unsafe {
        voiage_v1_enbs_r(
            &raw const evsi,
            &raw const cost,
            &raw mut out_value,
            &raw mut out_status,
        );
    }
    assert_eq!(out_status, VoiageStatusV1::InvalidArgument.as_i32());
    assert_eq!(out_value, 42.0);

    // Null pointers should not crash
    unsafe {
        voiage_v1_enbs_r(
            std::ptr::null(),
            &raw const cost,
            &raw mut out_value,
            &raw mut out_status,
        );
    }
}

#[test]
fn enbs_rejects_invalid_inputs_without_writing() {
    let mut result = 101.0_f64;
    let status = unsafe { voiage_v1_enbs(f64::NAN, 3.0, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);
    assert_eq!(result.to_bits(), 101.0_f64.to_bits());

    let status = unsafe { voiage_v1_enbs(12.5, -1.0, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);
    assert_eq!(result.to_bits(), 101.0_f64.to_bits());

    let status = unsafe { voiage_v1_enbs(12.5, 3.0, std::ptr::null_mut()) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);
}

#[test]
fn enbs_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_ENBS, 0);
}
