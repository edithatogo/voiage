//! Supported Rust facade for VOIAGE.
//!
//! The facade is intentionally module-qualified. This keeps decision
//! contracts, diagnostics, numerical kernels, and serialization boundaries
//! distinguishable while providing one stable package for Rust consumers.

#![forbid(unsafe_code)]

/// Structured diagnostic and error contracts.
pub mod diagnostics {
    pub use voiage_diagnostics::*;
}

/// Validated, binding-independent decision-analysis domain contracts.
pub mod domain {
    pub use voiage_domain::*;
}

/// Binding-independent Value of Information numerical kernels.
pub mod numerics {
    pub use voiage_numerics::*;
}

/// Canonical serialization contracts for validated VOIAGE values.
pub mod serialization {
    pub use voiage_serialization::*;
}

/// Facade package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
