//! Validated, binding-independent domain contracts for voiage.

#![forbid(unsafe_code)]

use core::fmt;
use serde::{de, Deserialize, Deserializer, Serialize};

mod collections;
mod core_contracts;
mod decision_problem;

pub use collections::{DomainError, SampleCube, SampleMatrix, SampleVector, StrategyCollection};
pub use core_contracts::{
    CeacResult, CoreContractError, ParameterSet, TrialArm, TrialDesign, ValueArray,
};
pub use decision_problem::{
    AnalysisType, DecisionProblem, DecisionProblemError, ExecutionMode, Intervention, Provenance,
    ReproducibilityMetadata,
};

/// Identifies this crate while the production domain contracts are introduced.
pub const CRATE_NAME: &str = "voiage-domain";

/// A validation failure raised while constructing a domain primitive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationError {
    /// A textual identifier was empty after trimming surrounding whitespace.
    EmptyIdentifier,
    /// A floating-point value was NaN or infinite.
    NonFinite,
    /// A probability was outside the inclusive unit interval.
    ProbabilityOutOfRange,
    /// A threshold was zero or negative.
    NonPositiveThreshold,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyIdentifier => "identifier must not be empty",
            Self::NonFinite => "value must be finite",
            Self::ProbabilityOutOfRange => "probability must be between 0 and 1 inclusive",
            Self::NonPositiveThreshold => "threshold must be greater than zero",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ValidationError {}

macro_rules! identifier_type {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Creates an identifier after trimming surrounding whitespace.
            ///
            /// # Errors
            ///
            /// Returns [`ValidationError::EmptyIdentifier`] when the trimmed
            /// input is empty.
            pub fn new(value: impl AsRef<str>) -> Result<Self, ValidationError> {
                let value = value.as_ref().trim();
                if value.is_empty() {
                    return Err(ValidationError::EmptyIdentifier);
                }
                Ok(Self(value.to_owned()))
            }

            /// Returns the validated identifier text.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(de::Error::custom)
            }
        }
    };
}

identifier_type!(Identifier, "A validated, non-empty domain identifier.");
identifier_type!(StrategyId, "A validated, non-empty strategy identifier.");

macro_rules! finite_type {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Debug, PartialEq, Serialize)]
        #[serde(transparent)]
        pub struct $name(f64);

        impl $name {
            /// Creates a value when the input is finite.
            ///
            /// # Errors
            ///
            /// Returns [`ValidationError::NonFinite`] for NaN or infinity.
            pub fn new(value: f64) -> Result<Self, ValidationError> {
                if !value.is_finite() {
                    return Err(ValidationError::NonFinite);
                }
                Ok(Self(value))
            }

            /// Returns the validated floating-point value.
            #[must_use]
            pub const fn get(self) -> f64 {
                self.0
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = f64::deserialize(deserializer)?;
                Self::new(value).map_err(de::Error::custom)
            }
        }
    };
}

finite_type!(Cost, "A finite cost value; signed values are permitted.");
finite_type!(
    Effect,
    "A finite effect value; signed values are permitted."
);

/// A finite probability in the inclusive range from zero to one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Probability(f64);

impl Probability {
    /// Creates a probability from a finite value in the unit interval.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::NonFinite`] for NaN or infinity and
    /// [`ValidationError::ProbabilityOutOfRange`] outside `[0, 1]`.
    pub fn new(value: f64) -> Result<Self, ValidationError> {
        if !value.is_finite() {
            return Err(ValidationError::NonFinite);
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(ValidationError::ProbabilityOutOfRange);
        }
        Ok(Self(value))
    }

    /// Returns the validated probability.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// A finite threshold that is strictly greater than zero.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Threshold(f64);

impl Threshold {
    /// Creates a positive, finite threshold.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError::NonFinite`] for NaN or infinity and
    /// [`ValidationError::NonPositiveThreshold`] for zero or negative values.
    pub fn new(value: f64) -> Result<Self, ValidationError> {
        if !value.is_finite() {
            return Err(ValidationError::NonFinite);
        }
        if value <= 0.0 {
            return Err(ValidationError::NonPositiveThreshold);
        }
        Ok(Self(value))
    }

    /// Returns the validated threshold.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Threshold {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// An unsigned 64-bit seed used for deterministic stochastic execution.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Seed(u64);

impl Seed {
    /// Creates a seed. Every unsigned 64-bit value is valid.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the seed value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
