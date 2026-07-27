//! Caller-owned Decision Problem JSON validation and transport.

use std::{mem, panic};

use voiage_serialization::normalize_decision_problem_json;

use crate::VoiageStatusV1;

/// Validates and normalizes a Decision Problem into caller-owned JSON.
///
/// Pass a null `buffer` with zero `capacity` to query the required byte count.
/// The required count includes the trailing NUL. Invalid input does not modify
/// `required` or `buffer`; a short buffer updates only `required`.
///
/// # Safety
///
/// `input` must point to `input_length` readable bytes. `required` must be
/// non-null, aligned, and writable for one `u64`. When `capacity` is non-zero,
/// `buffer` must point to `capacity` writable bytes. The regions must not
/// overlap, and no pointer is retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_decision_problem_json(
    input: *const u8,
    input_length: u64,
    buffer: *mut u8,
    capacity: u64,
    required: *mut u64,
) -> VoiageStatusV1 {
    panic::catch_unwind(|| normalize_and_copy(input, input_length, buffer, capacity, required))
        .unwrap_or(VoiageStatusV1::Panic)
}

fn normalize_and_copy(
    input: *const u8,
    input_length: u64,
    buffer: *mut u8,
    capacity: u64,
    required: *mut u64,
) -> VoiageStatusV1 {
    if input.is_null() || required.is_null() || (required as usize) % mem::align_of::<u64>() != 0 {
        return VoiageStatusV1::InvalidArgument;
    }
    let (Ok(input_length), Ok(capacity)) =
        (usize::try_from(input_length), usize::try_from(capacity))
    else {
        return VoiageStatusV1::InvalidArgument;
    };
    if input_length == 0
        || input_length > isize::MAX as usize
        || capacity > isize::MAX as usize
        || (capacity != 0 && buffer.is_null())
    {
        return VoiageStatusV1::InvalidArgument;
    }

    // SAFETY: input nullness and addressable length were validated; the caller
    // guarantees this readable region for the duration of the call.
    let input = unsafe { std::slice::from_raw_parts(input, input_length) };
    let Ok(mut normalized) = normalize_decision_problem_json(input) else {
        return VoiageStatusV1::SerializationFailure;
    };
    normalized.push(0);

    // SAFETY: alignment and nullness were checked above; the caller contract
    // requires one writable u64.
    unsafe { required.write(normalized.len() as u64) };
    if capacity == 0 {
        return VoiageStatusV1::Ok;
    }
    if capacity < normalized.len() {
        return VoiageStatusV1::BufferTooSmall;
    }

    // SAFETY: the caller promises `capacity` writable bytes, the normalized
    // length is bounded by that capacity, and input/output must not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(normalized.as_ptr(), buffer, normalized.len());
    }
    VoiageStatusV1::Ok
}
