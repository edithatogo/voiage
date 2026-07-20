//! Leaf C ABI adapter for the voiage Rust core.
//!
//! Phase 3 freezes only portable ABI discovery infrastructure. Numerical
//! operations are deliberately absent until their Phase 5 kernels pass parity
//! and profiling gates.

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

/// Fixed-width, self-describing v1 infrastructure capability response.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoiageAbiCapabilitiesV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this capability response structure.
    pub struct_version: u32,
    /// Infrastructure capability bitset; no numerical-operation bits exist.
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

/// Returns only the ABI infrastructure capabilities implemented in Phase 3.
// SAFETY: the export attribute is the only unsafe-code lint exception. The
// function returns a fixed-width value and does not access caller memory.
#[allow(unsafe_code)]
#[no_mangle]
pub extern "C" fn voiage_v1_capabilities() -> VoiageAbiCapabilitiesV1 {
    VoiageAbiCapabilitiesV1 {
        struct_size: ABI_CAPABILITIES_STRUCT_SIZE,
        struct_version: 1,
        capability_bits: VOIAGE_ABI_VERSION_NEGOTIATION | VOIAGE_ABI_CAPABILITY_QUERY,
    }
}
