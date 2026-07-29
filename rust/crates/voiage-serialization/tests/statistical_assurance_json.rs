//! Statistical-assurance envelope JSON normalization tests.

use voiage_serialization::normalize_statistical_assurance_json;

#[test]
fn canonical_statistical_assurance_example_normalizes_through_the_v1_dto() {
    let input =
        include_bytes!("../../../../specs/core-api/examples/v1/statistical-assurance.example.json");
    let normalized = normalize_statistical_assurance_json(input).unwrap();

    assert!(!normalized.contains(&b'\n'));
    let decoded: serde_json::Value = serde_json::from_slice(&normalized).unwrap();
    let canonical: serde_json::Value = serde_json::from_slice(input).unwrap();
    assert_eq!(decoded, canonical);
    assert_eq!(decoded["reporting_class"], "nested-monte-carlo");
    assert_eq!(decoded["replications"], 4);
    assert_eq!(decoded["convergence"]["converged"], true);
}

#[test]
fn statistical_assurance_rejects_reversed_confidence_intervals() {
    let invalid = br#"{
      "reporting_class":"sample-average",
      "bias_assessment":null,
      "variance_estimate":1.0,
      "monte_carlo_standard_error":1.0,
      "confidence_interval":{"level":0.95,"lower":2.0,"upper":1.0,"method":"normal"},
      "convergence":null,
      "effective_sample_size":null,
      "rng":null,
      "replications":1,
      "budget":{"draws":1,"evaluations":1,"elapsed_seconds":0.0},
      "stopping_reason":"fixed-budget",
      "numerical_error":{"absolute_bound":null,"relative_bound":null,"source":"declared"}
    }"#;

    assert!(normalize_statistical_assurance_json(invalid).is_err());
}
