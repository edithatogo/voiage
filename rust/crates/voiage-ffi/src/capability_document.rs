//! Caller-owned transport for the generated capability document.

use std::{mem, panic};

use crate::{generated_capabilities::CAPABILITY_DOCUMENT_JSON_NUL, VoiageStatusV1};

/// Copies the canonical stable-core capability document into caller memory.
///
/// Pass a null `buffer` with zero `capacity` to query the required byte count.
/// `required` must be non-null, aligned, and writable for one `u64`. The
/// required count includes the trailing NUL. No partial document is written.
///
/// # Safety
///
/// When `capacity` is non-zero, `buffer` must point to `capacity` writable
/// bytes. `buffer` and `required` must not overlap. Neither pointer is retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_capabilities_json(
    buffer: *mut u8,
    capacity: u64,
    required: *mut u64,
) -> VoiageStatusV1 {
    panic::catch_unwind(|| copy_document(buffer, capacity, required))
        .unwrap_or(VoiageStatusV1::Panic)
}

fn copy_document(buffer: *mut u8, capacity: u64, required: *mut u64) -> VoiageStatusV1 {
    if required.is_null() || (required as usize) % mem::align_of::<u64>() != 0 {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok(capacity) = usize::try_from(capacity) else {
        return VoiageStatusV1::InvalidArgument;
    };
    if capacity > isize::MAX as usize || (capacity != 0 && buffer.is_null()) {
        return VoiageStatusV1::InvalidArgument;
    }

    // SAFETY: alignment and nullness were checked above; the caller contract
    // requires one writable u64. The static document length fits in u64.
    unsafe { required.write(CAPABILITY_DOCUMENT_JSON_NUL.len() as u64) };
    if capacity == 0 {
        return VoiageStatusV1::Ok;
    }
    if capacity < CAPABILITY_DOCUMENT_JSON_NUL.len() {
        return VoiageStatusV1::BufferTooSmall;
    }

    // SAFETY: the caller promises `capacity` writable bytes and the document
    // length is bounded by that capacity. The source is a non-overlapping
    // immutable static byte slice.
    unsafe {
        std::ptr::copy_nonoverlapping(
            CAPABILITY_DOCUMENT_JSON_NUL.as_ptr(),
            buffer,
            CAPABILITY_DOCUMENT_JSON_NUL.len(),
        );
    };
    VoiageStatusV1::Ok
}
