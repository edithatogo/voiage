//! Scalar result-envelope JSON normalization tests.

use voiage_serialization::{
    normalize_enbs_result_json, normalize_evppi_result_json, normalize_evsi_result_json,
};

#[test]
fn scalar_result_examples_normalize_through_their_versioned_dtos() {
    for (input, normalize, discriminator) in [
        (
            include_bytes!("../../../../specs/core-api/examples/v1/evppi.example.json").as_slice(),
            normalize_evppi_result_json as fn(&[u8]) -> serde_json::Result<Vec<u8>>,
            "evppi",
        ),
        (
            include_bytes!("../../../../specs/core-api/examples/v1/evsi.example.json").as_slice(),
            normalize_evsi_result_json,
            "evsi",
        ),
        (
            include_bytes!("../../../../specs/core-api/examples/v1/enbs.example.json").as_slice(),
            normalize_enbs_result_json,
            "enbs",
        ),
    ] {
        let normalized = normalize(input).unwrap();
        assert!(!normalized.contains(&b'\n'));
        let decoded: serde_json::Value = serde_json::from_slice(&normalized).unwrap();
        assert_eq!(decoded["analysis_type"], discriminator);
    }
}

#[test]
fn scalar_result_normalizers_reject_wrong_discriminators() {
    let invalid = br#"{
      "analysis_id":"evsi-screening-001",
      "decision_problem_id":"screening-program-001",
      "analysis_type":"evppi",
      "trial_design_id":"screening-trial-design-001",
      "sample_size":240,
      "evsi":22.75
    }"#;

    assert!(normalize_evsi_result_json(invalid).is_err());
}
