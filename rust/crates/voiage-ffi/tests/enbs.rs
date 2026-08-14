//! Contract tests for Rust-authoritative ENBS through the C ABI.

#![allow(unsafe_code)]

use voiage_ffi::{voiage_v1_enbs, VoiageStatusV1, VOIAGE_ABI_ENBS};

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
