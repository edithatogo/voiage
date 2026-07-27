//! Signed 32-bit JSON adapters for R's base `.C` interface.

use crate::{
    voiage_v1_decision_problem_json, voiage_v1_statistical_assurance_json, VoiageStatusV1,
};

type JsonTransport = unsafe extern "C" fn(*const u8, u64, *mut u8, u64, *mut u64) -> VoiageStatusV1;

unsafe fn call_for_r(
    operation: JsonTransport,
    input: *const u8,
    input_length: *const i32,
    buffer: *mut u8,
    capacity: *const i32,
    required_size: *mut i32,
    out_status: *mut i32,
) {
    if input_length.is_null()
        || capacity.is_null()
        || required_size.is_null()
        || out_status.is_null()
        || (input_length as usize) % std::mem::align_of::<i32>() != 0
        || (capacity as usize) % std::mem::align_of::<i32>() != 0
        || (required_size as usize) % std::mem::align_of::<i32>() != 0
        || (out_status as usize) % std::mem::align_of::<i32>() != 0
    {
        return;
    }

    // SAFETY: nullness and alignment were validated above.
    let (input_length, capacity) = unsafe { (input_length.read(), capacity.read()) };
    if input_length < 0 || capacity < 0 {
        // SAFETY: both output pointers were validated above.
        unsafe {
            required_size.write(0);
            out_status.write(VoiageStatusV1::InvalidArgument.as_i32());
        }
        return;
    }

    let mut required = 0_u64;
    // SAFETY: the adapter forwards the caller-owned regions and translated
    // non-negative lengths to the checked, panic-contained JSON transport.
    let status = unsafe {
        operation(
            input,
            input_length as u64,
            buffer,
            capacity as u64,
            std::ptr::addr_of_mut!(required),
        )
    };
    let required = i32::try_from(required);
    let status = if required.is_err() {
        VoiageStatusV1::InvalidArgument
    } else {
        status
    };
    // SAFETY: both output pointers were validated above.
    unsafe {
        required_size.write(required.unwrap_or(0));
        out_status.write(status.as_i32());
    }
}

/// R `.C` adapter for [`voiage_v1_decision_problem_json`].
///
/// # Safety
///
/// Input and output buffers follow the underlying JSON transport contract.
/// Scalar pointers must be non-null, aligned, and readable or writable for one
/// signed 32-bit integer as appropriate.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_decision_problem_json_i32_r(
    input: *const u8,
    input_length: *const i32,
    buffer: *mut u8,
    capacity: *const i32,
    required_size: *mut i32,
    out_status: *mut i32,
) {
    // SAFETY: this wrapper transfers its documented contract to the adapter.
    unsafe {
        call_for_r(
            voiage_v1_decision_problem_json,
            input,
            input_length,
            buffer,
            capacity,
            required_size,
            out_status,
        )
    }
}

/// R `.C` adapter for [`voiage_v1_statistical_assurance_json`].
///
/// # Safety
///
/// The pointer contract is identical to
/// [`voiage_v1_decision_problem_json_i32_r`].
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_statistical_assurance_json_i32_r(
    input: *const u8,
    input_length: *const i32,
    buffer: *mut u8,
    capacity: *const i32,
    required_size: *mut i32,
    out_status: *mut i32,
) {
    // SAFETY: this wrapper transfers its documented contract to the adapter.
    unsafe {
        call_for_r(
            voiage_v1_statistical_assurance_json,
            input,
            input_length,
            buffer,
            capacity,
            required_size,
            out_status,
        )
    }
}
