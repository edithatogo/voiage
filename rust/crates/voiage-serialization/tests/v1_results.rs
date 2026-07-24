//! Contract tests for the stable v1 result DTOs.

use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::PathBuf};
use voiage_serialization::{
    CeafResultV1, CeafResultV1Input, DominanceResultV1, DominanceResultV1Input, DominanceStatus,
    EnbsResultV1, EnbsResultV1Input, EvpiResultV1, EvpiResultV1Input, EvppiResultV1,
    EvppiResultV1Input, EvsiResultV1, EvsiResultV1Input, ExpectedLossResultV1,
    ExpectedLossResultV1Input,
};

#[test]
fn future_kernels_can_construct_all_results_without_json_round_trips() {
    assert!(EvpiResultV1::try_from(EvpiResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        willingness_to_pay: 1.0,
        expected_current_value: 1.0,
        expected_perfect_information: 2.0,
        evpi: 1.0,
        strategy_names: None,
        expected_net_benefit_by_strategy: None,
        method: None,
    })
    .is_ok());
    assert!(EvppiResultV1::try_from(EvppiResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        parameter_names: vec!["p".into()],
        evppi: 1.0,
        expected_current_value: None,
        expected_perfect_information: None,
        method: None,
        diagnostics: None,
    })
    .is_ok());
    assert!(EvsiResultV1::try_from(EvsiResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        trial_design_id: "t".into(),
        sample_size: 1,
        expected_sample_value: None,
        evsi: 1.0,
        expected_current_value: None,
        expected_perfect_information: None,
        method: None,
        diagnostics: None,
    })
    .is_ok());
    assert!(EnbsResultV1::try_from(EnbsResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        trial_design_id: "t".into(),
        sample_size: 1,
        design_cost: 1.0,
        enbs: 1.0,
        expected_sample_value: None,
        expected_perfect_information: None,
        method: None,
        diagnostics: None,
    })
    .is_ok());
    assert!(ExpectedLossResultV1::try_from(ExpectedLossResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        strategy_names: vec!["A".into(), "B".into()],
        expected_net_benefit_by_strategy: vec![1.0, 2.0],
        expected_opportunity_loss_by_strategy: vec![1.0, 0.5],
        optimal_strategy_index: 1,
        minimum_expected_opportunity_loss: 0.5,
        sample_count: 10,
        method: Some("expected-opportunity-loss".into()),
        reporting: None,
    })
    .is_ok());
    assert!(CeafResultV1::try_from(CeafResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        wtp_thresholds: vec![1.0],
        optimal_strategy_indices: vec![0],
        optimal_strategy_names: vec!["A".into()],
        acceptability_probabilities: vec![0.5],
        probability_lower: vec![0.4],
        probability_upper: vec![0.6],
        expected_net_benefit: vec![1.0],
        reporting: None,
    })
    .is_ok());
    assert!(DominanceResultV1::try_from(DominanceResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        strategy_names: vec!["A".into(), "B".into()],
        costs: vec![1.0, 2.0],
        effects: vec![1.0, 2.0],
        frontier_indices: vec![0, 1],
        strongly_dominated_indices: vec![],
        extended_dominated_indices: vec![],
        status: vec![DominanceStatus::Frontier, DominanceStatus::Frontier],
        incremental_costs: vec![1.0],
        incremental_effects: vec![1.0],
        icers: vec![1.0],
        reporting: None,
    })
    .is_ok());
}

#[test]
fn typed_kernel_inputs_remain_fail_closed() {
    assert!(EvsiResultV1::try_from(EvsiResultV1Input {
        analysis_id: "a".into(),
        decision_problem_id: "d".into(),
        trial_design_id: "t".into(),
        sample_size: 0,
        expected_sample_value: None,
        evsi: 1.0,
        expected_current_value: None,
        expected_perfect_information: None,
        method: None,
        diagnostics: None,
    })
    .is_err());
}

fn example(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../specs/core-api/examples/v1")
        .join(format!("{name}.example.json"));
    serde_json::from_str(&fs::read_to_string(path).expect("canonical example must exist"))
        .expect("canonical example must contain JSON")
}

fn round_trip<T>(name: &str)
where
    T: DeserializeOwned + serde::Serialize,
{
    let canonical = example(name);
    let result: T = serde_json::from_value(canonical.clone()).expect("example must validate");
    assert_json_equivalent(&serde_json::to_value(result).unwrap(), &canonical);
}

fn assert_json_equivalent(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(left), Value::Number(right)) => {
            assert_eq!(left.as_f64(), right.as_f64());
        }
        (Value::Array(left), Value::Array(right)) => {
            assert_eq!(left.len(), right.len());
            for (left, right) in left.iter().zip(right) {
                assert_json_equivalent(left, right);
            }
        }
        (Value::Object(left), Value::Object(right)) => {
            assert_eq!(left.len(), right.len());
            for (key, right) in right {
                assert_json_equivalent(&left[key], right);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn all_canonical_examples_round_trip_exactly() {
    round_trip::<EvpiResultV1>("evpi");
    round_trip::<EvppiResultV1>("evppi");
    round_trip::<EvsiResultV1>("evsi");
    round_trip::<EnbsResultV1>("enbs");
    round_trip::<ExpectedLossResultV1>("expected-loss");
    round_trip::<CeafResultV1>("ceaf");
    round_trip::<DominanceResultV1>("dominance");
}

#[test]
fn expected_loss_enforces_alignment_optimum_and_selected_loss() {
    let mut misaligned = example("expected-loss");
    misaligned["strategy_names"] = json!(["Current practice"]);
    assert!(serde_json::from_value::<ExpectedLossResultV1>(misaligned).is_err());

    let mut wrong_optimum = example("expected-loss");
    wrong_optimum["optimal_strategy_index"] = json!(0);
    assert!(serde_json::from_value::<ExpectedLossResultV1>(wrong_optimum).is_err());

    let mut wrong_loss = example("expected-loss");
    wrong_loss["minimum_expected_opportunity_loss"] = json!(1.0);
    assert!(serde_json::from_value::<ExpectedLossResultV1>(wrong_loss).is_err());
}

#[test]
fn unknown_fields_and_wrong_discriminators_fail_closed() {
    let mut unknown_field_payload = example("evpi");
    unknown_field_payload["unexpected"] = json!(true);
    assert!(serde_json::from_value::<EvpiResultV1>(unknown_field_payload).is_err());
    let mut wrong_discriminator_payload = example("evsi");
    wrong_discriminator_payload["analysis_type"] = json!("evpi");
    assert!(serde_json::from_value::<EvsiResultV1>(wrong_discriminator_payload).is_err());
}

#[test]
fn scalar_contracts_enforce_finite_nonnegative_and_signed_enbs_rules() {
    for (name, field) in [("evpi", "evpi"), ("evppi", "evppi"), ("evsi", "evsi")] {
        let mut value = example(name);
        value[field] = json!(-0.01);
        let rejected = match name {
            "evpi" => serde_json::from_value::<EvpiResultV1>(value).is_err(),
            "evppi" => serde_json::from_value::<EvppiResultV1>(value).is_err(),
            _ => serde_json::from_value::<EvsiResultV1>(value).is_err(),
        };
        assert!(rejected);
    }
    let mut enbs = example("enbs");
    enbs["enbs"] = json!(-10.0);
    assert!(serde_json::from_value::<EnbsResultV1>(enbs).is_ok());
    let mut invalid_cost = example("enbs");
    invalid_cost["design_cost"] = json!(-1.0);
    assert!(serde_json::from_value::<EnbsResultV1>(invalid_cost).is_err());
}

#[test]
fn enbs_uses_exactly_the_closed_schema_property_set() {
    let schema_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../specs/core-api/schemas/v1/results/enbs.schema.json");
    let schema: Value = serde_json::from_str(
        &fs::read_to_string(schema_path).expect("canonical ENBS schema must exist"),
    )
    .expect("canonical ENBS schema must contain JSON");
    let schema_properties: BTreeSet<_> = schema["properties"]
        .as_object()
        .expect("ENBS schema properties must be an object")
        .keys()
        .cloned()
        .collect();

    let canonical = example("enbs");
    let result: EnbsResultV1 = serde_json::from_value(canonical).unwrap();
    let serialized = serde_json::to_value(result).unwrap();
    let serialized_properties: BTreeSet<_> = serialized
        .as_object()
        .expect("serialized ENBS result must be an object")
        .keys()
        .cloned()
        .collect();
    assert!(serialized_properties.is_subset(&schema_properties));
    assert!(!serialized_properties.contains("expected_current_value"));

    let mut forbidden = example("enbs");
    forbidden["expected_current_value"] = json!(1.0);
    assert!(serde_json::from_value::<EnbsResultV1>(forbidden).is_err());
}

#[test]
fn evpi_optional_strategy_arrays_are_joint_and_aligned() {
    let mut missing = example("evpi");
    missing
        .as_object_mut()
        .unwrap()
        .remove("expected_net_benefit_by_strategy");
    assert!(serde_json::from_value::<EvpiResultV1>(missing).is_err());
    let mut mismatched = example("evpi");
    mismatched["expected_net_benefit_by_strategy"] = json!([1.0]);
    assert!(serde_json::from_value::<EvpiResultV1>(mismatched).is_err());
}

#[test]
fn ceaf_enforces_alignment_probability_range_and_ordered_bounds() {
    let mut misaligned = example("ceaf");
    misaligned["expected_net_benefit"] = json!([1.0]);
    assert!(serde_json::from_value::<CeafResultV1>(misaligned).is_err());
    let mut invalid_bounds = example("ceaf");
    invalid_bounds["probability_lower"] = json!([0.8, 0.82]);
    assert!(serde_json::from_value::<CeafResultV1>(invalid_bounds).is_err());
}

#[test]
fn dominance_enforces_partition_status_indices_and_transition_lengths() {
    let mut duplicate = example("dominance");
    duplicate["extended_dominated_indices"] = json!([4]);
    assert!(serde_json::from_value::<DominanceResultV1>(duplicate).is_err());
    let mut wrong_status = example("dominance");
    wrong_status["status"][2] = json!("frontier");
    assert!(serde_json::from_value::<DominanceResultV1>(wrong_status).is_err());
    let mut bad_transitions = example("dominance");
    bad_transitions["icers"] = json!([100.0]);
    assert!(serde_json::from_value::<DominanceResultV1>(bad_transitions).is_err());
}
