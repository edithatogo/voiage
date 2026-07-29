//! EVPI result-envelope JSON validation and normalization tests.

use voiage_serialization::normalize_evpi_result_json;

const VALID: &[u8] = include_bytes!("../../../../specs/core-api/examples/v1/evpi.example.json");

#[test]
fn normalized_evpi_result_json_is_compact_and_schema_validated() {
    let normalized = normalize_evpi_result_json(VALID).unwrap();

    assert!(!normalized.contains(&b'\n'));
    let decoded: serde_json::Value = serde_json::from_slice(&normalized).unwrap();
    assert_eq!(decoded["analysis_type"], "evpi");
    assert_eq!(decoded["evpi"], 125.0);
}

#[test]
fn normalized_evpi_result_json_rejects_misaligned_optional_arrays() {
    let invalid = br#"{
      "analysis_id":"evpi-screening-001",
      "decision_problem_id":"screening-program-001",
      "analysis_type":"evpi",
      "willingness_to_pay":50000,
      "expected_current_value":1505.0,
      "expected_perfect_information":1630.0,
      "evpi":125.0,
      "strategy_names":["Usual care"]
    }"#;

    assert!(normalize_evpi_result_json(invalid).is_err());
}
