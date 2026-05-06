use serde::Deserialize;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use voiage_core::{evsi_stochastic_contract, TrialArm, TrialDesign, ValueArray};

fn sample_trial_design() -> TrialDesign {
    TrialDesign::new(
        "screening-trial-design-001",
        vec![
            TrialArm::new("control", "Control", 1).expect("valid arm"),
            TrialArm::new("treatment", "Treatment", 2).expect("valid arm"),
        ],
    )
    .expect("valid trial design")
}

fn sample_value_array() -> ValueArray {
    ValueArray::new(
        "current-net-benefit-001",
        vec!["Strategy A".to_string(), "Strategy B".to_string()],
        vec![
            vec![8.0, 3.0],
            vec![7.0, 4.0],
            vec![1.0, 9.0],
            vec![2.0, 10.0],
        ],
    )
    .expect("valid sample array")
}

fn run_evsi_stochastic_baseline() -> f64 {
    evsi_stochastic_contract(
        "evsi-001",
        &sample_trial_design(),
        &sample_value_array(),
        42,
        None,
    )
    .expect("evsi contract should be valid")
    .result
    .evsi
}

#[test]
fn evsi_stochastic_baseline_returns_expected_value() {
    let result = run_evsi_stochastic_baseline();
    assert_eq!(result, 0.75);
}

#[test]
fn evsi_stochastic_baseline_runs_a_repeatable_workload_shape() {
    let start = Instant::now();
    let mut total = 0.0;

    for _ in 0..10_000 {
        total += black_box(run_evsi_stochastic_baseline());
    }

    let elapsed = start.elapsed();

    assert!(elapsed.as_nanos() > 0);
    assert!(total > 0.0);
}

#[derive(Debug, Deserialize)]
struct EvsiStochasticBaselineArtifact {
    benchmark_name: String,
    metric_type: String,
    workload: EvsiStochasticBaselineWorkload,
    expected: EvsiStochasticBaselineExpected,
    metadata: EvsiStochasticBaselineMetadata,
}

#[derive(Debug, Deserialize)]
struct EvsiStochasticBaselineWorkload {
    analysis_id: String,
    repeats: u64,
    trial_design: TrialDesign,
    current_net_benefit: ValueArray,
    seed: u64,
    method: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EvsiStochasticBaselineExpected {
    evsi: f64,
    comparison_rule: String,
    regression_policy: String,
}

#[derive(Debug, Deserialize)]
struct EvsiStochasticBaselineMetadata {
    phase: String,
    notes: Vec<String>,
    provenance: BTreeMap<String, String>,
}

#[test]
fn evsi_stochastic_baseline_artifact_matches_contract() {
    let artifact: EvsiStochasticBaselineArtifact =
        serde_json::from_str(include_str!("evsi_stochastic_baseline.json"))
            .expect("baseline artifact should be valid JSON");

    assert_eq!(artifact.benchmark_name, "evsi_stochastic_baseline");
    assert_eq!(artifact.metric_type, "evsi_stochastic_kernel");
    assert_eq!(artifact.workload.analysis_id, "evsi-001");
    assert_eq!(artifact.workload.repeats, 10_000);
    assert_eq!(artifact.workload.trial_design, sample_trial_design());
    assert_eq!(artifact.workload.current_net_benefit, sample_value_array());
    assert_eq!(artifact.workload.seed, 42);
    assert_eq!(artifact.workload.method, None);
    assert_eq!(artifact.expected.evsi, 0.75);
    assert_eq!(artifact.expected.comparison_rule, "exact");
    assert_eq!(artifact.expected.regression_policy, "ci-contract-only");
    assert_eq!(artifact.metadata.phase, "phase-1-evsi-kernel-baseline");
    assert!(!artifact.metadata.notes.is_empty());
    assert_eq!(
        artifact
            .metadata
            .provenance
            .get("benchmark_kind")
            .map(String::as_str),
        Some("deterministic-contract-baseline")
    );
}
