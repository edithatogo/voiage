//! End-to-end checks for the base-R `.C` JSON adapters.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_decision_problem_json_i32_r, voiage_v1_statistical_assurance_json_i32_r,
    VoiageStatusV1,
};

fn round_trip(
    operation: unsafe extern "C" fn(*const u8, *const i32, *mut u8, *const i32, *mut i32, *mut i32),
    input: &[u8],
) -> Vec<u8> {
    let input_length = i32::try_from(input.len()).expect("fixture length fits i32");
    let mut required = 0_i32;
    let mut status = -1_i32;
    let zero_capacity = 0_i32;
    // SAFETY: all scalar regions are aligned and valid; the zero-capacity
    // query does not access its dangling output pointer.
    unsafe {
        operation(
            input.as_ptr(),
            std::ptr::addr_of!(input_length),
            std::ptr::dangling_mut::<u8>(),
            std::ptr::addr_of!(zero_capacity),
            std::ptr::addr_of_mut!(required),
            std::ptr::addr_of_mut!(status),
        )
    };
    assert_eq!(status, VoiageStatusV1::Ok.as_i32());
    assert!(required > 1);

    let capacity = required;
    let mut output = vec![0_u8; required as usize];
    // SAFETY: input and output regions and all scalar pointers are valid,
    // aligned, and disjoint.
    unsafe {
        operation(
            input.as_ptr(),
            std::ptr::addr_of!(input_length),
            output.as_mut_ptr(),
            std::ptr::addr_of!(capacity),
            std::ptr::addr_of_mut!(required),
            std::ptr::addr_of_mut!(status),
        )
    };
    assert_eq!(status, VoiageStatusV1::Ok.as_i32());
    assert_eq!(output.last(), Some(&0));
    output
}

#[test]
fn normalizes_decision_problem_for_r() {
    let input =
        include_bytes!("../../../../specs/core-api/examples/v1/decision-problem.example.json");
    let output = round_trip(voiage_v1_decision_problem_json_i32_r, input);
    let document = std::str::from_utf8(&output[..output.len() - 1]).expect("normalized UTF-8 JSON");
    assert!(document.contains(r#""decision_problem_id":"screening-program-001""#));
}

#[test]
fn normalizes_statistical_assurance_for_r() {
    let input =
        include_bytes!("../../../../specs/core-api/examples/v1/statistical-assurance.example.json");
    let output = round_trip(voiage_v1_statistical_assurance_json_i32_r, input);
    let document = std::str::from_utf8(&output[..output.len() - 1]).expect("normalized UTF-8 JSON");
    assert!(document.contains(r#""reporting_class":"nested-monte-carlo""#));
}

#[test]
fn reports_invalid_signed_lengths_without_dereferencing_input() {
    let input_length = -1_i32;
    let capacity = 0_i32;
    let mut required = 9_i32;
    let mut status = -1_i32;
    // SAFETY: scalar pointers are valid; the invalid signed input length must
    // be rejected before the dangling input pointer is accessed.
    unsafe {
        voiage_v1_decision_problem_json_i32_r(
            std::ptr::dangling::<u8>(),
            std::ptr::addr_of!(input_length),
            std::ptr::dangling_mut::<u8>(),
            std::ptr::addr_of!(capacity),
            std::ptr::addr_of_mut!(required),
            std::ptr::addr_of_mut!(status),
        )
    };
    assert_eq!(required, 0);
    assert_eq!(status, VoiageStatusV1::InvalidArgument.as_i32());
}
