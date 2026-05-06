use voiage_core::{
    evppi_contract, validate_reporting_payload, AnalysisEnvelope, ApproximationStatus,
    DiagnosticStatus, DomainError, EvppiSummary, MethodMaturity,
};

#[test]
fn evppi_contract_builds_a_deterministic_analysis_envelope() {
    let parameter_names = vec!["effect".to_string(), "cost".to_string()];

    let envelope = evppi_contract("evppi-001", &parameter_names, 1505.0, 1530.0, 1630.0, None)
        .expect("valid EVPPI contract");

    assert_eq!(envelope.analysis_id, "evppi-001");
    assert_eq!(envelope.analysis_type, "evppi");
    assert_eq!(envelope.method.as_deref(), Some("deterministic-summary"));
    assert_eq!(
        envelope.method_metadata.method_maturity,
        MethodMaturity::Stable
    );
    assert_eq!(
        envelope.method_metadata.approximation_status,
        ApproximationStatus::Exact
    );
    assert_eq!(envelope.diagnostics.status, DiagnosticStatus::Ok);
    assert_eq!(
        envelope.result,
        EvppiSummary {
            evppi: 25.0,
            parameter_names,
            expected_current_value: 1505.0,
            expected_partial_information_value: 1530.0,
            expected_perfect_information: 1630.0,
            method: Some("deterministic-summary".to_string()),
        }
    );
    validate_reporting_payload(&envelope.reporting).expect("reporting should be valid");
    assert_eq!(envelope.reporting.reporting_standard, "CHEERS-VOI");
    assert_eq!(envelope.reporting.analysis_type, "evppi");
    assert_eq!(envelope.reporting.method_family, "partial-information");
    assert_eq!(
        envelope
            .reporting
            .provenance
            .get("parameter_count")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        envelope
            .reporting
            .reproducibility
            .get("value_rule")
            .map(String::as_str),
        Some("partial_minus_current")
    );
}

#[test]
fn evppi_contract_rejects_invalid_inputs() {
    let err = evppi_contract("evppi-001", &[], 1505.0, 1530.0, 1630.0, None)
        .expect_err("empty parameter names are rejected");
    assert_eq!(err, DomainError::EmptyCollection("parameter_names"));

    let err = evppi_contract(
        "evppi-001",
        &["effect".to_string(), "effect".to_string()],
        1505.0,
        1530.0,
        1630.0,
        None,
    )
    .expect_err("duplicate parameter names are rejected");
    assert_eq!(err, DomainError::DuplicateValue("parameter_names"));

    let err = evppi_contract(
        "evppi-001",
        &["effect".to_string()],
        f64::NAN,
        1530.0,
        1630.0,
        None,
    )
    .expect_err("non-finite scalars are rejected");
    assert_eq!(err, DomainError::NonFinite("expected_current_value"));
}

#[test]
fn evppi_contract_round_trips_through_serde() {
    let parameter_names = vec!["effect".to_string()];
    let envelope: AnalysisEnvelope<EvppiSummary> = evppi_contract(
        "evppi-002",
        &parameter_names,
        1505.0,
        1515.0,
        1630.0,
        Some("summary".to_string()),
    )
    .expect("valid EVPPI contract");

    let encoded = serde_json::to_string(&envelope).expect("serializes");
    let decoded: AnalysisEnvelope<EvppiSummary> =
        serde_json::from_str(&encoded).expect("deserializes");
    assert_eq!(decoded, envelope);
}
