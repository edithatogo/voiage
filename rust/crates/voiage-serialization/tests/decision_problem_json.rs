//! Decision Problem JSON validation and normalization tests.

use voiage_serialization::normalize_decision_problem_json;

#[test]
fn normalized_decision_problem_json_is_compact_and_validated() {
    let input = br#"{
      "decision_problem_id":"screening-001",
      "title":"Screening programme",
      "analysis_type":"net-benefit-first",
      "currency":"AUD",
      "willingness_to_pay":50000.0,
      "interventions":[
        {"intervention_id":"usual-care","name":"Usual care","is_reference":true}
      ]
    }"#;
    let normalized = normalize_decision_problem_json(input).unwrap();

    assert!(!normalized.contains(&b'\n'));
    let decoded: serde_json::Value = serde_json::from_slice(&normalized).unwrap();
    assert_eq!(decoded["decision_problem_id"], "screening-001");
}

#[test]
fn normalized_decision_problem_json_rejects_invalid_domain_values() {
    let invalid = br#"{
      "decision_problem_id":"screening-001",
      "title":"Screening programme",
      "analysis_type":"net-benefit-first",
      "currency":"AU",
      "willingness_to_pay":50000.0,
      "interventions":[
        {"intervention_id":"usual-care","name":"Usual care","is_reference":true}
      ]
    }"#;

    assert!(normalize_decision_problem_json(invalid).is_err());
}
