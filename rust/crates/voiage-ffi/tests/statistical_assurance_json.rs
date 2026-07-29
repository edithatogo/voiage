//! C ABI contract for statistical-assurance envelope JSON transport.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_statistical_assurance_json, VoiageStatusV1, VOIAGE_ABI_STATISTICAL_ASSURANCE_JSON,
};

#[test]
fn statistical_assurance_supports_caller_owned_size_queries() {
    let input =
        include_bytes!("../../../../specs/core-api/examples/v1/statistical-assurance.example.json");
    let mut required = 0_u64;
    let status = unsafe {
        voiage_v1_statistical_assurance_json(
            input.as_ptr(),
            input.len() as u64,
            std::ptr::null_mut(),
            0,
            &raw mut required,
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert!(required > 1);
}

#[test]
fn invalid_assurance_does_not_write_required_size() {
    let invalid = br#"{"reporting_class":"nested-monte-carlo"}"#;
    let mut required = 73_u64;
    let status = unsafe {
        voiage_v1_statistical_assurance_json(
            invalid.as_ptr(),
            invalid.len() as u64,
            std::ptr::null_mut(),
            0,
            &raw mut required,
        )
    };

    assert_eq!(status, VoiageStatusV1::SerializationFailure);
    assert_eq!(required, 73);
}

#[test]
fn statistical_assurance_json_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_STATISTICAL_ASSURANCE_JSON, 0);
}
