//! Structured result-envelope JSON normalization tests.

use voiage_serialization::{
    normalize_ceaf_result_json, normalize_dominance_result_json,
    normalize_expected_loss_result_json,
};

#[test]
fn structured_result_examples_normalize_through_their_versioned_dtos() {
    for (input, normalize, discriminator) in [
        (
            include_bytes!("../../../../specs/core-api/examples/v1/expected-loss.example.json")
                .as_slice(),
            normalize_expected_loss_result_json as fn(&[u8]) -> serde_json::Result<Vec<u8>>,
            "expected_loss",
        ),
        (
            include_bytes!("../../../../specs/core-api/examples/v1/ceaf.example.json").as_slice(),
            normalize_ceaf_result_json,
            "ceaf",
        ),
        (
            include_bytes!("../../../../specs/core-api/examples/v1/dominance.example.json")
                .as_slice(),
            normalize_dominance_result_json,
            "dominance",
        ),
    ] {
        let normalized = normalize(input).unwrap();
        assert!(!normalized.contains(&b'\n'));
        let decoded: serde_json::Value = serde_json::from_slice(&normalized).unwrap();
        assert_eq!(decoded["analysis_type"], discriminator);
    }
}

#[test]
fn structured_result_normalizers_enforce_method_specific_invariants() {
    let invalid_ceaf = br#"{
      "analysis_id":"a",
      "decision_problem_id":"d",
      "analysis_type":"ceaf",
      "wtp_thresholds":[50000],
      "optimal_strategy_indices":[0],
      "optimal_strategy_names":["usual care"],
      "acceptability_probabilities":[0.4],
      "probability_lower":[0.5],
      "probability_upper":[0.7],
      "expected_net_benefit":[10]
    }"#;

    assert!(normalize_ceaf_result_json(invalid_ceaf).is_err());
}
