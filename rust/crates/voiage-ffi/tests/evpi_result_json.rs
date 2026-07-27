//! Contract tests for caller-owned EVPI result JSON transport.

#![allow(unsafe_code)]

use voiage_ffi::{voiage_v1_evpi_result_json, VoiageStatusV1, VOIAGE_ABI_EVPI_RESULT_JSON};

const VALID: &[u8] = br#"{
  "analysis_id":"evpi-screening-001",
  "decision_problem_id":"screening-program-001",
  "analysis_type":"evpi",
  "willingness_to_pay":50000,
  "expected_current_value":1505.0,
  "expected_perfect_information":1630.0,
  "evpi":125.0,
  "strategy_names":["Usual care","Targeted screening"],
  "expected_net_benefit_by_strategy":[1500.0,1625.0],
  "method":"nested-monte-carlo"
}"#;

#[test]
fn evpi_result_json_supports_query_then_caller_owned_copy() {
    let mut required = 0_u64;
    let status = unsafe {
        voiage_v1_evpi_result_json(
            VALID.as_ptr(),
            VALID.len() as u64,
            std::ptr::null_mut(),
            0,
            &raw mut required,
        )
    };
    assert_eq!(status, VoiageStatusV1::Ok);

    let mut output = vec![0_u8; usize::try_from(required).unwrap()];
    let status = unsafe {
        voiage_v1_evpi_result_json(
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
    assert!(decoded.contains(r#""analysis_type":"evpi""#));
    assert!(decoded.contains(r#""evpi":125.0"#));
    assert!(!decoded.contains('\n'));
}

#[test]
fn invalid_evpi_envelopes_do_not_write_output() {
    let invalid = VALID
        .windows(b"125.0".len())
        .position(|window| window == b"125.0")
        .map(|offset| {
            let mut value = VALID.to_vec();
            value.splice(offset..offset + b"125.0".len(), b"-1.0".iter().copied());
            value
        })
        .unwrap();
    let mut required = 73_u64;
    let mut output = [0xA5_u8; 32];

    let status = unsafe {
        voiage_v1_evpi_result_json(
            invalid.as_ptr(),
            invalid.len() as u64,
            output.as_mut_ptr(),
            output.len() as u64,
            &raw mut required,
        )
    };

    assert_eq!(status, VoiageStatusV1::SerializationFailure);
    assert_eq!(required, 73);
    assert_eq!(output, [0xA5; 32]);
}

#[test]
fn evpi_result_json_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_EVPI_RESULT_JSON, 0);
}
