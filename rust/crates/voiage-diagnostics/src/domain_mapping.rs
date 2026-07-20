//! Deterministic translation from domain validation failures.

use voiage_domain::{CoreContractError, DecisionProblemError, DomainError, ValidationError};

use crate::{ErrorCode, ErrorDetails, ErrorRecord};

fn domain_record(
    code: ErrorCode,
    variant: &'static str,
    message: impl Into<String>,
    details: ErrorDetails,
) -> ErrorRecord {
    ErrorRecord::new(code, message).with_details(
        details
            .with_context("error_origin", "voiage-domain")
            .with_context("domain_variant", variant),
    )
}

impl From<ValidationError> for ErrorRecord {
    fn from(error: ValidationError) -> Self {
        let variant = match error {
            ValidationError::EmptyIdentifier => "empty_identifier",
            ValidationError::NonFinite => "non_finite",
            ValidationError::ProbabilityOutOfRange => "probability_out_of_range",
            ValidationError::NonPositiveThreshold => "non_positive_threshold",
        };
        domain_record(
            ErrorCode::InvalidInput,
            variant,
            error.to_string(),
            ErrorDetails::new(),
        )
    }
}

impl From<DomainError> for ErrorRecord {
    fn from(error: DomainError) -> Self {
        let message = error.to_string();
        match error {
            DomainError::Empty => domain_record(
                ErrorCode::InvalidInput,
                "empty",
                message,
                ErrorDetails::new(),
            ),
            DomainError::EmptyDimension { dimension } => domain_record(
                ErrorCode::DimensionMismatch,
                "empty_dimension",
                message,
                ErrorDetails::new().with_context("dimension", dimension.to_string()),
            ),
            DomainError::Ragged {
                dimension,
                expected,
                actual,
                index,
            } => domain_record(
                ErrorCode::DimensionMismatch,
                "ragged",
                message,
                ErrorDetails::new()
                    .with_expected_dimensions([expected])
                    .with_actual_dimensions([actual])
                    .with_context("dimension", dimension.to_string())
                    .with_context("index", index.to_string()),
            ),
            DomainError::NonFinite { index } => domain_record(
                ErrorCode::InvalidInput,
                "non_finite",
                message,
                ErrorDetails::new().with_context("index", index.to_string()),
            ),
            DomainError::BlankStrategy { index } => domain_record(
                ErrorCode::InvalidInput,
                "blank_strategy",
                message,
                ErrorDetails::new().with_context("index", index.to_string()),
            ),
            DomainError::DuplicateStrategy { index } => domain_record(
                ErrorCode::InvalidInput,
                "duplicate_strategy",
                message,
                ErrorDetails::new().with_context("index", index.to_string()),
            ),
        }
    }
}

impl From<DecisionProblemError> for ErrorRecord {
    fn from(error: DecisionProblemError) -> Self {
        let variant = match error {
            DecisionProblemError::CurrencyTooShort => "currency_too_short",
            DecisionProblemError::EmptyOutcomeNames => "empty_outcome_names",
            DecisionProblemError::DuplicateOutcomeName => "duplicate_outcome_name",
            DecisionProblemError::EmptyInterventions => "empty_interventions",
            DecisionProblemError::DuplicateInterventionId => "duplicate_intervention_id",
            DecisionProblemError::MissingFixtureIdentity => "missing_fixture_identity",
            DecisionProblemError::MissingStochasticSeed => "missing_stochastic_seed",
        };
        domain_record(
            ErrorCode::InvalidInput,
            variant,
            error.to_string(),
            ErrorDetails::new(),
        )
    }
}

impl From<CoreContractError> for ErrorRecord {
    fn from(error: CoreContractError) -> Self {
        let message = error.to_string();
        match error {
            CoreContractError::Empty(field) => domain_record(
                ErrorCode::InvalidInput,
                "empty",
                message,
                ErrorDetails::new().with_field(field),
            ),
            CoreContractError::Blank { field, index } => domain_record(
                ErrorCode::InvalidInput,
                "blank",
                message,
                ErrorDetails::new()
                    .with_field(field)
                    .with_context("index", index.to_string()),
            ),
            CoreContractError::Duplicate { field, index } => domain_record(
                ErrorCode::InvalidInput,
                "duplicate",
                message,
                ErrorDetails::new()
                    .with_field(field)
                    .with_context("index", index.to_string()),
            ),
            CoreContractError::Dimension {
                field,
                expected,
                actual,
            } => domain_record(
                ErrorCode::DimensionMismatch,
                "dimension",
                message,
                ErrorDetails::new()
                    .with_field(field)
                    .with_expected_dimensions([expected])
                    .with_actual_dimensions([actual]),
            ),
            CoreContractError::ZeroSampleSize => domain_record(
                ErrorCode::InvalidInput,
                "zero_sample_size",
                message,
                ErrorDetails::new().with_field("sample_size"),
            ),
            CoreContractError::Invalid(field) => domain_record(
                ErrorCode::InvalidInput,
                "invalid",
                message,
                ErrorDetails::new().with_field(field),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use voiage_domain::{CoreContractError, DecisionProblemError, DomainError, ValidationError};

    use crate::{ErrorCode, ErrorRecord};

    fn assert_code<E>(error: E, expected: ErrorCode)
    where
        E: Into<ErrorRecord>,
    {
        let record: ErrorRecord = error.into();
        assert_eq!(record.code(), expected);
        assert_eq!(
            record
                .details()
                .and_then(|details| details.context("error_origin")),
            Some("voiage-domain")
        );
    }

    #[test]
    fn every_primitive_validation_error_maps_to_invalid_input() {
        for error in [
            ValidationError::EmptyIdentifier,
            ValidationError::NonFinite,
            ValidationError::ProbabilityOutOfRange,
            ValidationError::NonPositiveThreshold,
        ] {
            assert_code(error, ErrorCode::InvalidInput);
        }
    }

    #[test]
    fn every_collection_error_has_a_stable_mapping() {
        let cases = [
            (DomainError::Empty, ErrorCode::InvalidInput),
            (
                DomainError::EmptyDimension { dimension: 1 },
                ErrorCode::DimensionMismatch,
            ),
            (
                DomainError::Ragged {
                    dimension: 1,
                    expected: 2,
                    actual: 1,
                    index: 3,
                },
                ErrorCode::DimensionMismatch,
            ),
            (DomainError::NonFinite { index: 4 }, ErrorCode::InvalidInput),
            (
                DomainError::BlankStrategy { index: 0 },
                ErrorCode::InvalidInput,
            ),
            (
                DomainError::DuplicateStrategy { index: 1 },
                ErrorCode::InvalidInput,
            ),
        ];

        for (error, expected) in cases {
            assert_code(error, expected);
        }
    }

    #[test]
    fn every_decision_problem_error_maps_to_invalid_input() {
        for error in [
            DecisionProblemError::CurrencyTooShort,
            DecisionProblemError::EmptyOutcomeNames,
            DecisionProblemError::DuplicateOutcomeName,
            DecisionProblemError::EmptyInterventions,
            DecisionProblemError::DuplicateInterventionId,
            DecisionProblemError::MissingFixtureIdentity,
            DecisionProblemError::MissingStochasticSeed,
        ] {
            assert_code(error, ErrorCode::InvalidInput);
        }
    }

    #[test]
    fn every_core_contract_error_has_a_stable_mapping() {
        let cases = [
            (
                CoreContractError::Empty("parameters"),
                ErrorCode::InvalidInput,
            ),
            (
                CoreContractError::Blank {
                    field: "parameters",
                    index: 2,
                },
                ErrorCode::InvalidInput,
            ),
            (
                CoreContractError::Duplicate {
                    field: "strategy_names",
                    index: 3,
                },
                ErrorCode::InvalidInput,
            ),
            (
                CoreContractError::Dimension {
                    field: "net_benefit",
                    expected: 4,
                    actual: 2,
                },
                ErrorCode::DimensionMismatch,
            ),
            (CoreContractError::ZeroSampleSize, ErrorCode::InvalidInput),
            (
                CoreContractError::Invalid("analysis_id"),
                ErrorCode::InvalidInput,
            ),
        ];

        for (error, expected) in cases {
            assert_code(error, expected);
        }
    }

    #[test]
    fn core_contract_mapping_preserves_deterministic_structured_context() {
        let record: ErrorRecord = CoreContractError::Dimension {
            field: "cost_effectiveness_probabilities",
            expected: 5,
            actual: 3,
        }
        .into();
        let details = record.details().expect("dimension details");

        assert_eq!(record.code(), ErrorCode::DimensionMismatch);
        assert_eq!(details.field(), Some("cost_effectiveness_probabilities"));
        assert_eq!(details.expected_dimensions(), Some([5].as_slice()));
        assert_eq!(details.actual_dimensions(), Some([3].as_slice()));
        assert_eq!(details.context("domain_variant"), Some("dimension"));

        for (error, variant, field, index) in [
            (
                CoreContractError::Blank {
                    field: "display_name",
                    index: 1,
                },
                "blank",
                "display_name",
                Some("1"),
            ),
            (
                CoreContractError::Duplicate {
                    field: "arm_id",
                    index: 2,
                },
                "duplicate",
                "arm_id",
                Some("2"),
            ),
            (CoreContractError::Empty("arms"), "empty", "arms", None),
            (
                CoreContractError::Invalid("analysis_id"),
                "invalid",
                "analysis_id",
                None,
            ),
        ] {
            let record: ErrorRecord = error.into();
            let details = record.details().expect("core contract details");
            assert_eq!(details.field(), Some(field));
            assert_eq!(details.context("domain_variant"), Some(variant));
            assert_eq!(details.context("index"), index);
        }

        let zero: ErrorRecord = CoreContractError::ZeroSampleSize.into();
        let details = zero.details().expect("zero sample-size details");
        assert_eq!(details.field(), Some("sample_size"));
        assert_eq!(details.context("domain_variant"), Some("zero_sample_size"));
    }
}
