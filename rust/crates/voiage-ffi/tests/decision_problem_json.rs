//! Contract tests for caller-owned Decision Problem JSON transport.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_decision_problem_json, VoiageStatusV1, VOIAGE_ABI_DECISION_PROBLEM_JSON,
};

const VALID: &[u8] = br#"{
  "decision_problem_id":"screening-001",
  "title":"Screening programme",
  "analysis_type":"net-benefit-first",
  "currency":"AUD",
  "willingness_to_pay":50000.0,
  "interventions":[
    {"intervention_id":"usual-care","name":"Usual care","is_reference":true}
  ]
}"#;

#[test]
fn decision_problem_json_supports_query_then_caller_owned_copy() {
    let mut required = 0_u64;
    let status = unsafe {
        voiage_v1_decision_problem_json(
            VALID.as_ptr(),
            VALID.len() as u64,
            std::ptr::null_mut(),
            0,
            &raw mut required,
        )
    };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert!(required > 1);

    let mut output = vec![0_u8; usize::try_from(required).unwrap()];
    let status = unsafe {
        voiage_v1_decision_problem_json(
            VALID.as_ptr(),
            VALID.len() as u64,
            output.as_mut_ptr(),
            output.len() as u64,
            &raw mut required,
        )
    };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(output.last(), Some(&0));
    let decoded = std::str::from_utf8(&output[..output.len() - 1]).unwrap();
    assert!(decoded.contains(r#""decision_problem_id":"screening-001""#));
    assert!(decoded.contains(r#""is_reference":true"#));
    assert!(!decoded.contains('\n'));
}

#[test]
fn invalid_json_and_short_buffers_never_receive_partial_documents() {
    let invalid = br#"{"decision_problem_id":""}"#;
    let mut required = 73_u64;
    let mut output = [0xA5_u8; 16];
    let status = unsafe {
        voiage_v1_decision_problem_json(
            invalid.as_ptr(),
            invalid.len() as u64,
            output.as_mut_ptr(),
            output.len() as u64,
            &raw mut required,
        )
    };
    assert_eq!(status, VoiageStatusV1::SerializationFailure);
    assert_eq!(required, 73);
    assert_eq!(output, [0xA5; 16]);

    let status = unsafe {
        voiage_v1_decision_problem_json(
            VALID.as_ptr(),
            VALID.len() as u64,
            output.as_mut_ptr(),
            output.len() as u64,
            &raw mut required,
        )
    };
    assert_eq!(status, VoiageStatusV1::BufferTooSmall);
    assert!(required > output.len() as u64);
    assert_eq!(output, [0xA5; 16]);
}

#[test]
fn decision_problem_json_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_DECISION_PROBLEM_JSON, 0);
}
