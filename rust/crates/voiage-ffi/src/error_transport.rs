//! Audited caller-owned error transport for the C ABI.

use std::{cell::RefCell, mem, panic};

use crate::VoiageStatusV1;

const INVALID_ARGUMENT: &[u8] = b"invalid argument\0";
const PANIC_CONTAINED: &[u8] = b"panic contained\0";

thread_local! {
    static LAST_ERROR: RefCell<&'static [u8]> = const { RefCell::new(INVALID_ARGUMENT) };
}

/// Copies the calling thread's last error into caller-owned memory.
///
/// Pass a null `buffer` with zero `capacity` to query the required byte count.
/// `required` may be null when the caller does not need that count. The output
/// is always NUL terminated. Rust retains no caller pointer and returns no
/// allocation requiring a Rust destructor.
///
/// # Pointer contract
///
/// When non-null, `required` must be aligned and writable for one `u64`.
/// When `capacity` is non-zero, `buffer` must be writable for `capacity`
/// bytes. The two regions must not overlap.
///
/// # Safety
///
/// Callers must uphold the pointer contract above for the duration of this
/// call. In particular, every non-null output pointer must refer to writable
/// memory of the documented size and alignment.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_error_message(
    buffer: *mut u8,
    capacity: u64,
    required: *mut u64,
) -> VoiageStatusV1 {
    if let Ok(status) = panic::catch_unwind(|| copy_last_error(buffer, capacity, required)) {
        status
    } else {
        set_last_error(PANIC_CONTAINED);
        VoiageStatusV1::Panic
    }
}

fn copy_last_error(buffer: *mut u8, capacity: u64, required: *mut u64) -> VoiageStatusV1 {
    if !required.is_null() && (required as usize) % mem::align_of::<u64>() != 0 {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok(capacity) = usize::try_from(capacity) else {
        return VoiageStatusV1::InvalidArgument;
    };
    if capacity > isize::MAX as usize || (capacity != 0 && buffer.is_null()) {
        return VoiageStatusV1::InvalidArgument;
    }

    LAST_ERROR.with_borrow(|message| {
        if !required.is_null() {
            // SAFETY: alignment was checked above and the caller contract
            // requires one writable u64 at this non-null address. Message
            // lengths originate from bounded static slices and fit in u64.
            unsafe { required.write(message.len() as u64) };
        }
        if capacity == 0 {
            return VoiageStatusV1::Ok;
        }
        if capacity < message.len() {
            return VoiageStatusV1::BufferTooSmall;
        }

        // SAFETY: null and maximum-slice-length checks were completed above.
        // The caller promises `capacity` writable bytes, non-overlapping with
        // `required`; the message length is bounded by that capacity.
        unsafe { std::ptr::copy_nonoverlapping(message.as_ptr(), buffer, message.len()) };
        VoiageStatusV1::Ok
    })
}

fn set_last_error(message: &'static [u8]) {
    LAST_ERROR.with_borrow_mut(|current| *current = message);
}

#[cfg(test)]
mod tests {
    use super::{set_last_error, VoiageStatusV1, PANIC_CONTAINED};
    use std::panic;

    fn guarded_for_test(action: impl FnOnce() + panic::UnwindSafe) -> VoiageStatusV1 {
        panic::catch_unwind(action).map_or_else(
            |_| {
                set_last_error(PANIC_CONTAINED);
                VoiageStatusV1::Panic
            },
            |()| VoiageStatusV1::Ok,
        )
    }

    #[test]
    fn panic_guard_contains_unwind() {
        assert_eq!(
            guarded_for_test(|| panic!("test-only panic")),
            VoiageStatusV1::Panic
        );
    }
}
