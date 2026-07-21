//! Leaf C ABI adapter for the voiage Rust core.
//!
//! The versioned C ABI exposes portable discovery and Rust-owned scalar kernels.

#![deny(unsafe_op_in_unsafe_fn)]

// SAFETY: this is the sole pointer-dereferencing module. Every unsafe block is
// locally documented and guarded by the v1 pointer contract.
#[allow(unsafe_code)]
mod error_transport;
// SAFETY: lifecycle validates its sole caller-owned output pointer before the
// one documented write. Export wrappers contain panics before returning to C.
#[allow(unsafe_code)]
mod lifecycle;
mod status;

use std::panic::{self, AssertUnwindSafe};

use voiage_domain::SampleMatrix;
use voiage_numerics::evpi;

pub use error_transport::voiage_v1_error_message;
pub use lifecycle::{voiage_v1_handle_create, voiage_v1_handle_free};
pub use status::VoiageStatusV1;

/// Identifies this crate while the versioned C ABI is introduced.
pub const CRATE_NAME: &str = "voiage-ffi";

/// ABI major version implemented by the `voiage_v1_*` symbol namespace.
pub const VOIAGE_V1_ABI_MAJOR: u32 = 1;

/// Backwards-compatible ABI minor version implemented by this library.
pub const VOIAGE_V1_ABI_MINOR: u32 = 0;

/// Capability bit for ABI version negotiation.
pub const VOIAGE_ABI_VERSION_NEGOTIATION: u64 = 1 << 0;

/// Capability bit for infrastructure capability discovery.
pub const VOIAGE_ABI_CAPABILITY_QUERY: u64 = 1 << 1;

/// Capability bit for the stable scalar EVPI operation.
pub const VOIAGE_ABI_EVPI: u64 = 1 << 2;

const ABI_VERSION_STRUCT_SIZE: u32 = 12;
const ABI_CAPABILITIES_STRUCT_SIZE: u32 = 16;

/// Fixed-width, self-describing v1 ABI version response.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoiageAbiVersionV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// ABI major version. Breaking changes require a new symbol namespace.
    pub abi_major: u32,
    /// ABI minor version. Additive compatible changes increment this value.
    pub abi_minor: u32,
}

/// Fixed-width, self-describing v1 capability response.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoiageAbiCapabilitiesV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this capability response structure.
    pub struct_version: u32,
    /// Capability bitset for supported ABI operations.
    pub capability_bits: u64,
}

/// Returns the portable v1 ABI version contract.
// SAFETY: the export attribute is the only unsafe-code lint exception. The
// function has no pointers, mutable state, allocation, or numerical behavior.
#[allow(unsafe_code)]
#[no_mangle]
pub extern "C" fn voiage_v1_abi_version() -> VoiageAbiVersionV1 {
    VoiageAbiVersionV1 {
        struct_size: ABI_VERSION_STRUCT_SIZE,
        abi_major: VOIAGE_V1_ABI_MAJOR,
        abi_minor: VOIAGE_V1_ABI_MINOR,
    }
}

/// Returns the ABI capabilities implemented by this library.
// SAFETY: the export attribute is the only unsafe-code lint exception. The
// function returns a fixed-width value and does not access caller memory.
#[allow(unsafe_code)]
#[no_mangle]
pub extern "C" fn voiage_v1_capabilities() -> VoiageAbiCapabilitiesV1 {
    VoiageAbiCapabilitiesV1 {
        struct_size: ABI_CAPABILITIES_STRUCT_SIZE,
        struct_version: 1,
        capability_bits: VOIAGE_ABI_VERSION_NEGOTIATION
            | VOIAGE_ABI_CAPABILITY_QUERY
            | VOIAGE_ABI_EVPI,
    }
}

/// Computes EVPI from a row-major, finite net-benefit matrix.
///
/// # Safety
///
/// `values` must point to `rows * columns` readable `f64` values and `out`
/// must be non-null, aligned, and writable for one `f64`. Neither pointer is
/// retained after this call.
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evpi(
    values: *const f64,
    rows: u64,
    columns: u64,
    out: *mut f64,
) -> VoiageStatusV1 {
    if values.is_null()
        || out.is_null()
        || rows == 0
        || columns == 0
        || (out as usize) % std::mem::align_of::<f64>() != 0
    {
        return VoiageStatusV1::InvalidArgument;
    }
    let Some(length) = rows
        .checked_mul(columns)
        .and_then(|value| usize::try_from(value).ok())
    else {
        return VoiageStatusV1::InvalidArgument;
    };
    let (Ok(row_count), Ok(column_count)) = (usize::try_from(rows), usize::try_from(columns))
    else {
        return VoiageStatusV1::InvalidArgument;
    };
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller contract guarantees a readable row-major region.
        let slice = unsafe { std::slice::from_raw_parts(values, length) };
        let matrix = (0..row_count)
            .map(|row| {
                let start = row * column_count;
                slice[start..start + column_count].to_vec()
            })
            .collect::<Vec<_>>();
        let matrix = SampleMatrix::try_from(matrix).map_err(|_| VoiageStatusV1::InvalidArgument)?;
        evpi(&matrix).map_err(|_| VoiageStatusV1::NumericalFailure)
    }));
    match result {
        Ok(Ok(value)) => {
            // SAFETY: nullness and alignment were validated above.
            unsafe { out.write(value) };
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

/// Computes EVPI with signed 32-bit dimensions for runtimes such as base R
/// whose `.C` interface does not expose a portable unsigned 64-bit scalar.
///
/// # Safety
///
/// The pointer requirements are identical to [`voiage_v1_evpi`].
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evpi_i32(
    values: *const f64,
    rows: i32,
    columns: i32,
    out: *mut f64,
) -> VoiageStatusV1 {
    if rows <= 0 || columns <= 0 {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok(rows) = u64::try_from(rows) else {
        return VoiageStatusV1::InvalidArgument;
    };
    let Ok(columns) = u64::try_from(columns) else {
        return VoiageStatusV1::InvalidArgument;
    };
    // SAFETY: this adapter preserves the pointer contract of voiage_v1_evpi.
    unsafe { voiage_v1_evpi(values, rows, columns, out) }
}

/// Calls [`voiage_v1_evpi_i32`] and writes its status for `.C` runtimes that
/// cannot observe a C return value.
///
/// # Safety
///
/// `values` and `out_value` follow [`voiage_v1_evpi`] requirements, and
/// `out_status` must be non-null, aligned, and writable for one `i32`.
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evpi_i32_r(
    values: *const f64,
    rows: *const i32,
    columns: *const i32,
    out_value: *mut f64,
    out_status: *mut i32,
) {
    if rows.is_null()
        || columns.is_null()
        || out_status.is_null()
        || (rows as usize) % std::mem::align_of::<i32>() != 0
        || (columns as usize) % std::mem::align_of::<i32>() != 0
        || (out_status as usize) % std::mem::align_of::<i32>() != 0
    {
        return;
    }
    // SAFETY: nullness and alignment were validated above.
    let (rows, columns) = unsafe { (rows.read(), columns.read()) };
    // SAFETY: the caller contract validates out_status above; the delegated
    // operation validates the remaining pointers.
    let status = unsafe { voiage_v1_evpi_i32(values, rows, columns, out_value) };
    // SAFETY: nullness and alignment were validated above.
    unsafe { out_status.write(status.as_i32()) };
}
