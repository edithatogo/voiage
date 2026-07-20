//! Stable C ABI status values.

/// Stable status codes returned by v1 C ABI functions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum VoiageStatusV1 {
    /// The operation completed successfully.
    Ok = 0,
    /// An argument failed validation.
    InvalidArgument = 1,
    /// Input dimensions are incompatible.
    DimensionMismatch = 2,
    /// A requested backend is unavailable.
    BackendUnavailable = 3,
    /// A numerical operation failed.
    NumericalFailure = 4,
    /// Serialization or deserialization failed.
    SerializationFailure = 5,
    /// The caller-owned output buffer is too small.
    BufferTooSmall = 6,
    /// An unexpected panic was contained at the ABI boundary.
    Panic = 7,
    /// An internal invariant failed without unwinding into C.
    InternalError = 255,
}

impl VoiageStatusV1 {
    /// Returns the stable signed integer representation used by C callers.
    #[must_use]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }
}
