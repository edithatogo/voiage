//! C ABI contracts for structured result-envelope JSON transport.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_ceaf_result_json, voiage_v1_dominance_result_json,
    voiage_v1_expected_loss_result_json, VoiageStatusV1, VOIAGE_ABI_STRUCTURED_RESULT_JSON,
};

type JsonFunction = unsafe extern "C" fn(*const u8, u64, *mut u8, u64, *mut u64) -> VoiageStatusV1;

#[test]
fn each_structured_result_family_supports_caller_owned_size_queries() {
    for (input, function) in [
        (
            include_bytes!("../../../../specs/core-api/examples/v1/expected-loss.example.json")
                .as_slice(),
            voiage_v1_expected_loss_result_json as JsonFunction,
        ),
        (
            include_bytes!("../../../../specs/core-api/examples/v1/ceaf.example.json").as_slice(),
            voiage_v1_ceaf_result_json as JsonFunction,
        ),
        (
            include_bytes!("../../../../specs/core-api/examples/v1/dominance.example.json")
                .as_slice(),
            voiage_v1_dominance_result_json as JsonFunction,
        ),
    ] {
        let mut required = 0_u64;
        let status = unsafe {
            function(
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
}

#[test]
fn invalid_structured_result_does_not_write_required_size() {
    let invalid = br#"{"analysis_type":"dominance","strategy_names":[]}"#;
    let mut required = 73_u64;
    let status = unsafe {
        voiage_v1_dominance_result_json(
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
fn structured_result_json_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_STRUCTURED_RESULT_JSON, 0);
}
