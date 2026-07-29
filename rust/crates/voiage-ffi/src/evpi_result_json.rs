//! Caller-owned EVPI result-envelope JSON validation and transport.

use std::panic;

use voiage_serialization::normalize_evpi_result_json;

use crate::{json_transport::normalize_and_copy, VoiageStatusV1};

/// Validates and normalizes an EVPI v1 result into caller-owned JSON.
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
pub unsafe extern "C" fn voiage_v1_evpi_result_json(
    input: *const u8,
    input_length: u64,
    buffer: *mut u8,
    capacity: u64,
    required: *mut u64,
) -> VoiageStatusV1 {
    panic::catch_unwind(|| {
        // SAFETY: the public wrapper transfers its caller contract to the
        // shared checked transport.
        unsafe {
            normalize_and_copy(
                input,
                input_length,
                buffer,
                capacity,
                required,
                normalize_evpi_result_json,
            )
        }
    })
    .unwrap_or(VoiageStatusV1::Panic)
}
