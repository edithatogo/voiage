use core::fmt;

use voiage_diagnostics::{ErrorCategory, ErrorCode, ErrorDetails, ErrorRecord};

/// A validated numerical-kernel input error with stable machine identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumericalInputError(Box<ErrorRecord>);

impl NumericalInputError {
    pub(crate) fn invalid(field: &'static str, message: &'static str) -> Self {
        Self(Box::new(
            ErrorRecord::new(ErrorCode::InvalidInput, message)
                .with_details(ErrorDetails::new().with_field(field)),
        ))
    }

    pub(crate) fn dimension(
        field: &'static str,
        expected: usize,
        actual: usize,
        message: &'static str,
    ) -> Self {
        Self(Box::new(
            ErrorRecord::new(ErrorCode::DimensionMismatch, message).with_details(
                ErrorDetails::new()
                    .with_field(field)
                    .with_expected_dimensions([expected])
                    .with_actual_dimensions([actual]),
            ),
        ))
    }

    /// Returns the stable machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        self.0.code()
    }

    /// Returns the stable language-neutral error category.
    #[must_use]
    pub const fn category(&self) -> ErrorCategory {
        self.0.category()
    }

    /// Returns the structured diagnostic record.
    #[must_use]
    pub const fn record(&self) -> &ErrorRecord {
        &self.0
    }
}

impl fmt::Display for NumericalInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0.message())
    }
}

impl std::error::Error for NumericalInputError {}
