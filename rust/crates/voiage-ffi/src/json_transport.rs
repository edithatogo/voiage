//! Shared caller-owned JSON query/copy transport.

use std::mem;

use crate::VoiageStatusV1;

/// Normalizes validated JSON and copies it to a caller-owned buffer.
///
/// # Safety
///
/// The caller must satisfy the pointer, length, non-overlap, and lifetime
/// requirements documented by the public wrapper.
pub(crate) unsafe fn normalize_and_copy<E>(
    input: *const u8,
    input_length: u64,
    buffer: *mut u8,
    capacity: u64,
    required: *mut u64,
    normalize: impl FnOnce(&[u8]) -> Result<Vec<u8>, E>,
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
    let Ok(mut normalized) = normalize(input) else {
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
