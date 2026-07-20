//! Contract tests for bounded caller-owned v1 ABI error transport.

#![allow(unsafe_code)]

use std::mem;

use voiage_ffi::{voiage_v1_abi_version, voiage_v1_error_message, VoiageStatusV1};

#[test]
fn exports_a_stable_namespaced_version() {
    let version = voiage_v1_abi_version();
    assert_eq!(version.abi_major, 1);
    assert_eq!(version.abi_minor, 0);
}

#[test]
fn reports_required_size_without_allocating() {
    let mut required = 0_u64;
    // SAFETY: the optional output pointer is aligned and writable for one u64.
    let status = unsafe {
        voiage_v1_error_message(std::ptr::null_mut(), 0, std::ptr::addr_of_mut!(required))
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(required, "invalid argument\0".len() as u64);
}

#[test]
fn copies_a_nul_terminated_message_into_caller_memory() {
    let mut output = [0xAA_u8; 32];
    let mut required = 0_u64;
    // SAFETY: both output regions are valid, aligned, writable, and disjoint.
    let status = unsafe {
        voiage_v1_error_message(
            output.as_mut_ptr(),
            output.len() as u64,
            std::ptr::addr_of_mut!(required),
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(required, "invalid argument\0".len() as u64);
    let required = usize::try_from(required).expect("test message length fits usize");
    assert_eq!(&output[..required], b"invalid argument\0");
    assert_eq!(output[required], 0xAA);
}

#[test]
fn reports_small_buffers_without_partial_output() {
    let mut output = [0xAA_u8; 4];
    let mut required = 0_u64;
    // SAFETY: both output regions are valid, aligned, writable, and disjoint.
    let status = unsafe {
        voiage_v1_error_message(
            output.as_mut_ptr(),
            output.len() as u64,
            std::ptr::addr_of_mut!(required),
        )
    };

    assert_eq!(status, VoiageStatusV1::BufferTooSmall);
    assert_eq!(required, "invalid argument\0".len() as u64);
    assert_eq!(output, [0xAA; 4]);
}

#[test]
fn rejects_null_and_impossible_lengths() {
    let mut required = 0_u64;
    // SAFETY: this intentionally violates the null-buffer contract; the
    // function must reject it before dereferencing the pointer.
    assert_eq!(
        unsafe {
            voiage_v1_error_message(std::ptr::null_mut(), 1, std::ptr::addr_of_mut!(required))
        },
        VoiageStatusV1::InvalidArgument
    );
    // SAFETY: this intentionally supplies an unrepresentable or impossible
    // capacity; validation must reject it before accessing the dangling pointer.
    assert_eq!(
        unsafe {
            voiage_v1_error_message(
                std::ptr::dangling_mut::<u8>(),
                (isize::MAX as u64).saturating_add(1),
                std::ptr::addr_of_mut!(required),
            )
        },
        VoiageStatusV1::InvalidArgument
    );
}

#[test]
fn rejects_misaligned_required_length_pointer() {
    #[repr(align(8))]
    struct Aligned([u8; mem::size_of::<u64>() + 1]);

    let mut storage = Aligned([0; mem::size_of::<u64>() + 1]);
    #[allow(clippy::cast_ptr_alignment)]
    let misaligned = storage.0.as_mut_ptr().wrapping_add(1).cast::<u64>();
    // SAFETY: this intentionally violates alignment; validation must reject it
    // before dereferencing the pointer.
    assert_eq!(
        unsafe { voiage_v1_error_message(std::ptr::null_mut(), 0, misaligned) },
        VoiageStatusV1::InvalidArgument
    );
}

#[test]
fn exposes_an_unsafe_fixed_width_c_function_type() {
    let _: unsafe extern "C" fn(*mut u8, u64, *mut u64) -> VoiageStatusV1 = voiage_v1_error_message;
}

#[test]
fn status_values_are_stable_and_c_sized() {
    assert_eq!(mem::size_of::<VoiageStatusV1>(), 4);
    assert_eq!(VoiageStatusV1::Ok.as_i32(), 0);
    assert_eq!(VoiageStatusV1::InvalidArgument.as_i32(), 1);
    assert_eq!(VoiageStatusV1::BufferTooSmall.as_i32(), 6);
    assert_eq!(VoiageStatusV1::Panic.as_i32(), 7);
    assert_eq!(VoiageStatusV1::InternalError.as_i32(), 255);
}
