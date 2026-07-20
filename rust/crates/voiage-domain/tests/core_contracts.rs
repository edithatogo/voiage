//! Contract tests for stable aggregate domain types.

use std::collections::BTreeMap;

use voiage_domain::{CeacResult, ParameterSet, TrialArm, TrialDesign, ValueArray};

fn assert_stable_count(_: u64) {}

#[test]
fn value_array_is_dimension_checked_and_fail_closed() {
    let value = ValueArray::new(
        " values ",
        vec!["care".into(), "new".into()],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap();
    assert_eq!(value.id().as_str(), "values");
    assert_eq!(value.sample_count(), 2);
    assert_stable_count(value.sample_count());
    assert_eq!(value.net_benefit().shape(), [2, 2]);
    assert!(ValueArray::new("x", vec!["only".into()], vec![vec![1.0, 2.0]]).is_err());
    assert!(ValueArray::new("x", vec!["a".into()], vec![vec![f64::NAN]]).is_err());
    assert!(serde_json::from_str::<ValueArray>(
        r#"{"value_array_id":"x","sample_count":2,"strategy_names":["a"],"net_benefit":[[1.0]]}"#
    )
    .is_err());
    assert!(serde_json::from_str::<ValueArray>(r#"{"value_array_id":"x","sample_count":1,"strategy_names":["a"],"net_benefit":[[1.0]],"extra":true}"#).is_err());
}

#[test]
fn parameter_set_normalizes_names_and_aligns_samples() {
    let parameters = BTreeMap::from([
        (" age ".into(), vec![1.0, 2.0]),
        ("risk".into(), vec![0.1, 0.2]),
    ]);
    let set = ParameterSet::new("params", parameters).unwrap();
    assert_eq!(set.sample_count(), 2);
    assert_stable_count(set.sample_count());
    assert_eq!(set.parameter("age").unwrap().as_slice(), &[1.0, 2.0]);
    assert!(ParameterSet::new(
        "params",
        BTreeMap::from([("a".into(), vec![1.0]), ("b".into(), vec![1.0, 2.0])])
    )
    .is_err());
    assert!(ParameterSet::new(
        "params",
        BTreeMap::from([("a".into(), vec![f64::INFINITY])])
    )
    .is_err());
    assert!(ParameterSet::new(
        "params",
        BTreeMap::from([("a".into(), vec![1.0]), (" a ".into(), vec![2.0])])
    )
    .is_err());
    assert!(serde_json::from_str::<ParameterSet>(
        r#"{"parameter_set_id":"p","sample_count":2,"parameters":{"a":[1.0]}}"#
    )
    .is_err());
}

#[test]
fn trial_design_rejects_invalid_and_duplicate_arms() {
    let control = TrialArm::new("control", " Control ", 50).unwrap();
    assert_stable_count(control.sample_size());
    let active = TrialArm::new("active", "Active", 50).unwrap();
    let design = TrialDesign::new("trial", vec![control.clone(), active]).unwrap();
    assert_eq!(design.arms().len(), 2);
    assert_eq!(design.arms()[0].name(), "Control");
    assert!(TrialArm::new("x", "x", 0).is_err());
    assert!(TrialDesign::new("trial", vec![control.clone(), control]).is_err());
    assert!(serde_json::from_str::<TrialDesign>(
        r#"{"trial_design_id":"t","arms":[{"arm_id":"a","name":"A","sample_size":1,"extra":1}]}"#
    )
    .is_err());
}

#[test]
fn ceac_aligns_thresholds_strategies_and_probabilities() {
    let ceac = CeacResult::new(
        "analysis",
        "decision",
        vec!["care".into(), "new".into()],
        vec![0.0, 100.0],
        vec![vec![0.6, 0.4], vec![0.2, 0.8]],
        Some(" empirical ".into()),
    )
    .unwrap();
    assert_eq!(ceac.threshold_count(), 2);
    assert_eq!(ceac.strategy_count(), 2);
    let json = serde_json::to_value(&ceac).unwrap();
    assert_eq!(json["analysis_type"], "ceac");
    assert_eq!(json["method"], "empirical");
    assert!(CeacResult::new("a", "d", vec!["s".into()], vec![1.0], vec![vec![1.1]], None).is_err());
    assert!(CeacResult::new(
        "a",
        "d",
        vec!["s".into()],
        vec![f64::NAN],
        vec![vec![1.0]],
        None
    )
    .is_err());
    assert!(CeacResult::new(
        "a",
        "d",
        vec!["s".into()],
        vec![1.0, 2.0],
        vec![vec![1.0]],
        None
    )
    .is_err());
    assert!(serde_json::from_value::<CeacResult>(serde_json::json!({"analysis_id":"a","decision_problem_id":"d","analysis_type":"evpi","strategy_names":["s"],"willingness_to_pay_values":[1.0],"cost_effectiveness_probabilities":[[1.0]]})).is_err());
}

#[test]
fn ceac_round_trips_the_canonical_strategy_major_example() {
    let source = include_str!("../../../../specs/core-api/examples/v1/ceac.example.json");
    let canonical: serde_json::Value = serde_json::from_str(source).unwrap();
    let result: CeacResult = serde_json::from_str(source).unwrap();

    assert_eq!(result.strategy_count(), 2);
    assert_eq!(result.threshold_count(), 4);
    assert_eq!(serde_json::to_value(result).unwrap(), canonical);

    let mut unknown = canonical;
    unknown["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<CeacResult>(unknown).is_err());
}
