//! Contract tests for infrastructure-only opaque handle lifecycle exports.

#![allow(unsafe_code)]

use std::collections::HashSet;
use std::thread;

use voiage_ffi::{voiage_v1_handle_create, voiage_v1_handle_free, VoiageStatusV1};

fn create_handle() -> u64 {
    let mut handle = 0;
    // SAFETY: `handle` is aligned writable storage for exactly one u64.
    let status = unsafe { voiage_v1_handle_create(std::ptr::addr_of_mut!(handle)) };
    assert_eq!(status, VoiageStatusV1::Ok);
    handle
}

#[test]
fn creates_non_null_unique_opaque_handles_and_pairs_each_with_free() {
    let first = create_handle();
    let second = create_handle();

    assert_ne!(first, 0);
    assert_ne!(second, 0);
    assert_ne!(first, second);
    assert_eq!(voiage_v1_handle_free(first), VoiageStatusV1::Ok);
    assert_eq!(voiage_v1_handle_free(second), VoiageStatusV1::Ok);
}

#[test]
fn null_free_is_a_safe_no_op_and_repeated_free_is_rejected() {
    assert_eq!(voiage_v1_handle_free(0), VoiageStatusV1::Ok);

    let handle = create_handle();
    assert_eq!(voiage_v1_handle_free(handle), VoiageStatusV1::Ok);
    assert_eq!(
        voiage_v1_handle_free(handle),
        VoiageStatusV1::InvalidArgument
    );
}

#[test]
fn arbitrary_non_null_tokens_are_rejected_without_dereferencing() {
    assert_eq!(
        voiage_v1_handle_free(u64::MAX),
        VoiageStatusV1::InvalidArgument
    );
}

#[test]
fn lifecycle_registry_is_safe_for_concurrent_callers() {
    let workers = (0..8)
        .map(|_| {
            thread::spawn(|| {
                let handles = (0..128).map(|_| create_handle()).collect::<Vec<_>>();
                for handle in &handles {
                    assert_eq!(voiage_v1_handle_free(*handle), VoiageStatusV1::Ok);
                }
                handles
            })
        })
        .collect::<Vec<_>>();

    let handles = workers
        .into_iter()
        .flat_map(|worker| worker.join().expect("worker must not panic"))
        .collect::<Vec<_>>();
    let unique = handles.iter().copied().collect::<HashSet<_>>();

    assert_eq!(unique.len(), handles.len());
}

#[test]
fn exports_have_fixed_c_compatible_signatures_and_layouts() {
    let create: unsafe extern "C" fn(*mut u64) -> VoiageStatusV1 = voiage_v1_handle_create;
    let free: extern "C" fn(u64) -> VoiageStatusV1 = voiage_v1_handle_free;

    assert_eq!(std::mem::size_of::<u64>(), 8);
    assert_eq!(std::mem::size_of::<VoiageStatusV1>(), 4);
    let _ = (create, free);
}

#[test]
#[allow(clippy::cast_ptr_alignment)]
fn create_rejects_null_and_misaligned_outputs_without_writing() {
    // SAFETY: null is intentionally supplied to verify rejection before use.
    assert_eq!(
        unsafe { voiage_v1_handle_create(std::ptr::null_mut()) },
        VoiageStatusV1::InvalidArgument
    );

    let mut storage = [0_u8; 16];
    let base = storage.as_mut_ptr() as usize;
    let offset = (0..8)
        .find(|offset| (base + offset) % std::mem::align_of::<u64>() != 0)
        .expect("an eight-byte window contains a misaligned address");
    let pointer = storage.as_mut_ptr().wrapping_add(offset).cast::<u64>();
    // SAFETY: the deliberately misaligned pointer is rejected before writing.
    assert_eq!(
        unsafe { voiage_v1_handle_create(pointer) },
        VoiageStatusV1::InvalidArgument
    );
}
