//! Deterministic translation from wire-contract validation failures.

use voiage_diagnostics::{ErrorCode, ErrorDetails, ErrorRecord};

use crate::ValidationError;

impl From<ValidationError> for ErrorRecord {
    fn from(error: ValidationError) -> Self {
        ErrorRecord::new(ErrorCode::SerializationFailure, error.to_string()).with_details(
            ErrorDetails::new()
                .with_context("error_origin", "voiage-serialization")
                .with_context("serialization_variant", "invalid_result_payload"),
        )
    }
}

#[cfg(test)]
mod tests {
    use voiage_diagnostics::{ErrorCode, ErrorRecord};

    use crate::EvpiResultV1;

    #[test]
    fn result_validation_failure_maps_to_serialization_failure() {
        let error = serde_json::from_str::<EvpiResultV1>(
            r#"{"analysis_id":"","decision_problem_id":"d","analysis_type":"evpi","willingness_to_pay":1.0,"expected_current_value":0.0,"expected_perfect_information":0.0,"evpi":0.0}"#,
        )
        .expect_err("blank analysis ID must fail");

        assert!(!error.to_string().is_empty());
        let record = ErrorRecord::from(crate::ValidationError("invalid stable result payload"));
        assert_eq!(record.code(), ErrorCode::SerializationFailure);
        assert_eq!(
            record
                .details()
                .and_then(|details| details.context("error_origin")),
            Some("voiage-serialization")
        );
    }
}
