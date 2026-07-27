//! Caller-owned statistical-assurance envelope JSON transport.

use std::panic;

use voiage_serialization::normalize_statistical_assurance_json;

use crate::{json_transport::normalize_and_copy, VoiageStatusV1};

/// Validates and normalizes a statistical-assurance v1 envelope.
///
/// # Safety
///
/// `input` must point to `input_length` readable bytes. `required` must be
/// non-null, aligned, and writable for one `u64`. A non-zero `capacity`
/// requires that many writable bytes at `buffer`. Regions must not overlap,
/// and no pointer is retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_statistical_assurance_json(
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
                normalize_statistical_assurance_json,
            )
        }
    })
    .unwrap_or(VoiageStatusV1::Panic)
}
