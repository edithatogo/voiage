use std::collections::BTreeMap;

use voiage_core::{
    validate_reporting_payload, AnalysisEnvelope, ApproximationStatus, DiagnosticSeverity,
    DiagnosticStatus, DiagnosticWarning, Diagnostics, EnbsSummary, EvpiSummary, EvppiSummary,
    EvsiSummary, MethodMaturity, MethodMetadata, ParameterSet, Reporting, TrialArm, TrialDesign,
    ValueArray,
};

fn round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de> + PartialEq + std::fmt::Debug,
{
    let encoded = serde_json::to_string(value).expect("serializes");
    let decoded = serde_json::from_str(&encoded).expect("deserializes");
    assert_eq!(decoded, *value);
    decoded
}

#[test]
fn value_array_round_trips_deterministically() {
    let value = ValueArray::new(
        "array-001",
        vec!["Strategy A".to_string(), "Strategy B".to_string()],
        vec![vec![10.0, 1.0], vec![2.0, 8.0]],
    )
    .expect("valid value array");

    let decoded = round_trip(&value);
    assert_eq!(decoded.sample_count, 2);
}

#[test]
fn parameter_set_round_trips_deterministically() {
    let mut parameters = BTreeMap::new();
    parameters.insert("alpha".to_string(), vec![0.1, 0.2]);
    parameters.insert("beta".to_string(), vec![0.3, 0.4]);

    let value = ParameterSet::new("parameters-001", parameters).expect("valid parameter set");
    let decoded = round_trip(&value);
    assert_eq!(decoded.sample_count, 2);
}

#[test]
fn trial_design_round_trips_deterministically() {
    let value = TrialDesign::new(
        "trial-001",
        vec![
            TrialArm::new("control", "Control", 50).expect("valid arm"),
            TrialArm::new("treatment", "Treatment", 50).expect("valid arm"),
        ],
    )
    .expect("valid trial design");

    let decoded = round_trip(&value);
    assert_eq!(decoded.arms.len(), 2);
}

#[test]
fn diagnostics_round_trip_preserves_status_and_warnings() {
    let value = Diagnostics {
        analysis_id: "evsi-001".to_string(),
        status: DiagnosticStatus::Degraded,
        backend: Some("numpy".to_string()),
        warnings: vec![DiagnosticWarning {
            severity: DiagnosticSeverity::Warning,
            code: "backend_fallback".to_string(),
            message: "JAX is unavailable; falling back to NumPy.".to_string(),
            capability: Some("jax-acceleration".to_string()),
            degraded_path: None,
            approximation: Some(false),
            backend: Some("numpy".to_string()),
            fallback: Some("numpy".to_string()),
        }],
        unsupported_capabilities: vec!["jax-acceleration".to_string()],
        degraded_paths: vec!["surrogate-regression".to_string()],
        approximation_caveats: vec!["Tolerance-bounded estimate.".to_string()],
    };

    let decoded = round_trip(&value);
    assert_eq!(decoded.status, DiagnosticStatus::Degraded);
}

#[test]
fn method_metadata_round_trips_with_explicit_approximation_status() {
    let mut metadata = MethodMetadata::new(
        "evsi",
        "sample-information",
        MethodMaturity::Approximate,
        ApproximationStatus::Surrogate,
    )
    .expect("valid method metadata");
    metadata.capability_labels = vec![
        "surrogate-regression".to_string(),
        "jax-acceleration".to_string(),
    ];
    metadata.analysis_id = Some("evsi-001".to_string());
    metadata.decision_problem_id = Some("screening-program-001".to_string());
    metadata.decision_context = Some("screening-program".to_string());
    metadata.backend = Some("jax".to_string());
    metadata.notes = vec!["Approximation status must be explicit.".to_string()];

    round_trip(&metadata);
}

#[test]
fn reporting_round_trips_with_deterministic_maps() {
    let mut reporting =
        Reporting::cheers_voi("evsi", "sample-information", MethodMaturity::Approximate)
            .expect("valid reporting payload");
    reporting.analysis_id = Some("evsi-001".to_string());
    reporting.decision_problem_id = Some("screening-program-001".to_string());
    reporting.decision_context = Some("screening-program".to_string());
    reporting.perspective_ids = vec!["payer".to_string(), "societal".to_string()];
    reporting.perspective_labels = vec!["Payer".to_string(), "Societal".to_string()];
    reporting.population = Some(100_000.0);
    reporting.estimator = Some("moment-matching".to_string());
    reporting.seed = Some(1234);
    reporting
        .provenance
        .insert("source".to_string(), "synthetic".to_string());
    reporting
        .reproducibility
        .insert("seed_policy".to_string(), "fixed".to_string());
    reporting
        .diagnostics
        .insert("status".to_string(), "degraded".to_string());

    validate_reporting_payload(&reporting).expect("valid reporting payload");
    round_trip(&reporting);
}

#[test]
fn analysis_envelope_round_trips() {
    let mut method_metadata = MethodMetadata::new(
        "evpi",
        "perfect-information",
        MethodMaturity::Stable,
        ApproximationStatus::Exact,
    )
    .expect("valid method metadata");
    method_metadata.analysis_id = Some("evpi-001".to_string());

    let diagnostics =
        Diagnostics::new("evpi-001", DiagnosticStatus::Ok).expect("valid diagnostics");
    let reporting = Reporting::cheers_voi("evpi", "perfect-information", MethodMaturity::Stable)
        .expect("valid reporting payload");
    let result = EvpiSummary {
        evpi: 125.0,
        expected_current_value: 1505.0,
        expected_perfect_information: 1630.0,
        strategy_names: vec!["Usual care".to_string(), "Targeted screening".to_string()],
        expected_net_benefit_by_strategy: vec![1500.0, 1625.0],
        method: Some("nested-monte-carlo".to_string()),
    };

    let envelope = AnalysisEnvelope::new(
        "evpi-001",
        "evpi",
        method_metadata,
        diagnostics,
        reporting,
        result,
    )
    .expect("valid envelope");

    let decoded: AnalysisEnvelope<EvpiSummary> = round_trip(&envelope);
    assert_eq!(decoded.analysis_id, "evpi-001");
}

#[test]
fn summary_payloads_round_trip() {
    round_trip(&EvppiSummary {
        evppi: 40.5,
        parameter_names: vec![
            "prob_screen_positive".to_string(),
            "incremental_cost".to_string(),
        ],
        expected_current_value: 1505.0,
        expected_partial_information_value: 1545.5,
        expected_perfect_information: 1630.0,
        method: Some("gam".to_string()),
    });
    round_trip(&EvsiSummary {
        evsi: 22.75,
        trial_design_id: "screening-trial-design-001".to_string(),
        sample_size: 240,
        expected_current_value: 1505.0,
        expected_sample_value: 1527.75,
        expected_perfect_information: 1630.0,
        method: Some("moment-matching".to_string()),
    });
    round_trip(&EnbsSummary {
        enbs: 12_500.0,
        trial_design_id: "screening-trial-design-001".to_string(),
        sample_size: 240,
        design_cost: 180_000.0,
        expected_sample_value: 1527.75,
        expected_perfect_information: 1630.0,
        method: Some("sample-size-optimization".to_string()),
    });
}
