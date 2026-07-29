//! Contract tests for registry-generated C ABI capability discovery.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_capabilities, voiage_v1_capabilities_json, VoiageStatusV1,
    VOIAGE_ABI_CAPABILITY_DOCUMENT,
};

#[test]
fn capability_document_supports_query_then_caller_owned_copy() {
    let mut required = 0_u64;
    let status = unsafe {
        voiage_v1_capabilities_json(std::ptr::null_mut(), 0, std::ptr::addr_of_mut!(required))
    };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert!(required > 1);

    let required_length = usize::try_from(required).expect("document fits address space");
    let mut buffer = vec![0_u8; required_length];
    let status = unsafe {
        voiage_v1_capabilities_json(
            buffer.as_mut_ptr(),
            required,
            std::ptr::addr_of_mut!(required),
        )
    };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(buffer.last(), Some(&0));

    let document = std::str::from_utf8(&buffer[..buffer.len() - 1]).expect("UTF-8 JSON");
    assert!(document.contains(r#""source":"specs/v1/stable-core-status.json""#));
    assert!(document.contains(r#""method_id":"net-benefit""#));
    assert!(document.contains(r#""method_id":"evpi""#));
    assert!(document.contains(r#""authority_boundary":"python-compatibility-path""#));
}

#[test]
fn capability_document_fails_closed_for_invalid_buffers() {
    let mut required = 0_u64;
    let mut undersized = [0xAA_u8; 1];

    let status = unsafe {
        voiage_v1_capabilities_json(undersized.as_mut_ptr(), 1, std::ptr::addr_of_mut!(required))
    };
    assert_eq!(status, VoiageStatusV1::BufferTooSmall);
    assert_eq!(undersized, [0xAA]);

    let status = unsafe { voiage_v1_capabilities_json(std::ptr::null_mut(), 1, &raw mut required) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);
}

#[test]
fn capability_bit_advertises_the_generated_document() {
    assert_ne!(
        voiage_v1_capabilities().capability_bits & VOIAGE_ABI_CAPABILITY_DOCUMENT,
        0
    );
}
