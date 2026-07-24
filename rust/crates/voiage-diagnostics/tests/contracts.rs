//! Contract tests for diagnostics, warnings, and method metadata.

use voiage_diagnostics::{
    ApproximationStatus, DiagnosticStatus, Diagnostics, MethodMaturity, MethodMetadata,
    WarningRecord, WarningSeverity,
};

const DIAGNOSTICS_EXAMPLE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../specs/core-api/examples/v1/diagnostics.example.json"
));
const METHOD_METADATA_EXAMPLE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../specs/core-api/examples/v1/method-metadata.example.json"
));

fn warning(code: &str) -> WarningRecord {
    WarningRecord::new(WarningSeverity::Warning, code, "Plain-language explanation").unwrap()
}

#[test]
fn diagnostic_identifiers_and_labels_are_validated() {
    assert!(Diagnostics::new(" ", DiagnosticStatus::Ok, vec![], vec![], vec![], vec![]).is_err());
    assert!(Diagnostics::new(
        "analysis-1",
        DiagnosticStatus::Unsupported,
        vec![warning("unsupported_backend")],
        vec!["gpu".into(), "gpu".into()],
        vec![],
        vec![],
    )
    .is_err());
    assert!(WarningRecord::new(WarningSeverity::Warning, " ", "message").is_err());
    assert!(WarningRecord::new(WarningSeverity::Warning, "code", " ").is_err());
}

#[test]
fn degraded_and_approximate_statuses_require_explanatory_warnings() {
    assert!(Diagnostics::new(
        "analysis-1",
        DiagnosticStatus::Degraded,
        vec![],
        vec![],
        vec!["numpy-fallback".into()],
        vec![],
    )
    .is_err());
    assert!(Diagnostics::new(
        "analysis-1",
        DiagnosticStatus::Approximate,
        vec![],
        vec![],
        vec![],
        vec!["surrogate estimate".into()],
    )
    .is_err());
}

#[test]
fn statuses_are_consistent_with_diagnostic_details() {
    assert!(Diagnostics::new(
        "analysis-1",
        DiagnosticStatus::Ok,
        vec![warning("fallback")],
        vec![],
        vec!["numpy-fallback".into()],
        vec![],
    )
    .is_err());
    assert!(Diagnostics::new(
        "analysis-1",
        DiagnosticStatus::Unsupported,
        vec![warning("unsupported")],
        vec![],
        vec![],
        vec![],
    )
    .is_err());
}

#[test]
fn method_metadata_enforces_maturity_and_approximation_consistency() {
    assert!(MethodMetadata::new(
        "evsi",
        MethodMaturity::Experimental,
        ApproximationStatus::Exact,
        vec!["jax-acceleration".into()],
    )
    .is_ok());
    assert!(MethodMetadata::new(
        "evsi",
        MethodMaturity::Stable,
        ApproximationStatus::Exact,
        vec!["gpu".into(), "gpu".into()],
    )
    .is_err());
}

#[test]
fn valid_contracts_preserve_stable_values() {
    let diagnostics = Diagnostics::new(
        "analysis-1",
        DiagnosticStatus::Degraded,
        vec![warning("backend_fallback")],
        vec!["jax-acceleration".into()],
        vec!["numpy-fallback".into()],
        vec![],
    )
    .unwrap();
    assert_eq!(diagnostics.analysis_id(), "analysis-1");
    assert_eq!(diagnostics.status(), DiagnosticStatus::Degraded);

    let metadata = MethodMetadata::new(
        "evsi",
        MethodMaturity::FixtureBacked,
        ApproximationStatus::Surrogate,
        vec!["surrogate-regression".into()],
    )
    .unwrap()
    .with_analysis_id("analysis-1")
    .unwrap()
    .with_backend("numpy")
    .unwrap();
    assert_eq!(metadata.analysis_type(), "method_metadata");
    assert_eq!(metadata.method_family(), "evsi");
    assert_eq!(metadata.analysis_id(), Some("analysis-1"));
}

#[test]
fn canonical_diagnostics_round_trip_without_wire_drift() {
    let expected: serde_json::Value = serde_json::from_str(DIAGNOSTICS_EXAMPLE).unwrap();
    let diagnostics: Diagnostics = serde_json::from_value(expected.clone()).unwrap();
    assert_eq!(serde_json::to_value(diagnostics).unwrap(), expected);
}

#[test]
fn canonical_method_metadata_round_trip_without_wire_drift() {
    let expected: serde_json::Value = serde_json::from_str(METHOD_METADATA_EXAMPLE).unwrap();
    let metadata: MethodMetadata = serde_json::from_value(expected.clone()).unwrap();
    assert_eq!(serde_json::to_value(metadata).unwrap(), expected);
}

#[test]
fn wire_contracts_reject_unknown_and_invalid_fields() {
    let mut diagnostics: serde_json::Value = serde_json::from_str(DIAGNOSTICS_EXAMPLE).unwrap();
    diagnostics["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<Diagnostics>(diagnostics).is_err());

    let invalid_diagnostics = serde_json::json!({
        "analysis_id": "analysis-1",
        "status": "degraded",
        "warnings": [],
        "unsupported_capabilities": [],
        "degraded_paths": ["fallback"],
        "approximation_caveats": []
    });
    assert!(serde_json::from_value::<Diagnostics>(invalid_diagnostics).is_err());

    let mut metadata: serde_json::Value = serde_json::from_str(METHOD_METADATA_EXAMPLE).unwrap();
    metadata["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<MethodMetadata>(metadata).is_err());

    let invalid_metadata = serde_json::json!({
        "analysis_type": "method_metadata",
        "method_family": "evsi",
        "method_maturity": "approximate",
        "approximation_status": "exact",
        "capability_labels": ["surrogate-regression"]
    });
    assert!(serde_json::from_value::<MethodMetadata>(invalid_metadata).is_err());
}
