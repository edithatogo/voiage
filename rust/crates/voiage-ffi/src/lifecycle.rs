//! Thread-safe lifecycle primitives for opaque v1 ABI handles.
//!
//! Handles are non-pointer process-local tokens. They may be transferred
//! between threads, but callers must coordinate application-level ownership so
//! that exactly one caller performs the final free. Freeing the null handle is
//! always a successful no-op. Freeing an unknown or previously freed token is
//! safely rejected without dereferencing caller-controlled memory.

use std::collections::HashSet;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::VoiageStatusV1;

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
static LIVE_HANDLES: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();

fn live_handles() -> MutexGuard<'static, HashSet<u64>> {
    LIVE_HANDLES
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Creates a unique process-local opaque handle.
///
/// This infrastructure handle owns no numerical payload. Operation-specific
/// handles remain deferred until their Phase 5 kernels are stabilized.
fn create_handle() -> u64 {
    let mut handles = live_handles();
    loop {
        let candidate = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        if candidate != 0 && handles.insert(candidate) {
            return candidate;
        }
    }
}

/// Creates a unique process-local opaque handle in caller-owned storage.
///
/// # Safety
///
/// `out_handle` must be non-null, correctly aligned, and writable for one
/// `u64`. On failure it is not modified.
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_handle_create(out_handle: *mut u64) -> VoiageStatusV1 {
    if out_handle.is_null() || (out_handle as usize) % std::mem::align_of::<u64>() != 0 {
        return VoiageStatusV1::InvalidArgument;
    }

    match panic::catch_unwind(AssertUnwindSafe(create_handle)) {
        Ok(handle) => {
            // SAFETY: nullness and alignment were checked above; the C contract
            // requires writable caller-owned storage for exactly one u64.
            unsafe { out_handle.write(handle) };
            VoiageStatusV1::Ok
        }
        Err(_) => VoiageStatusV1::Panic,
    }
}

/// Releases an opaque handle created by [`voiage_v1_handle_create`].
///
/// Null-free is a successful no-op. Unknown and repeated frees return
/// [`VoiageStatusV1::InvalidArgument`] without touching external
/// memory. The registry lock makes create/free safe to call concurrently.
#[must_use]
#[allow(unsafe_code)]
#[no_mangle]
pub extern "C" fn voiage_v1_handle_free(handle: u64) -> VoiageStatusV1 {
    match panic::catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 || live_handles().remove(&handle) {
            VoiageStatusV1::Ok
        } else {
            VoiageStatusV1::InvalidArgument
        }
    })) {
        Ok(status) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}
