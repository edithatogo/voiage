use serde::Deserialize;
use std::hint::black_box;

use voiage_core::evpi;

fn baseline_workload() -> Vec<Vec<f64>> {
    vec![vec![10.0, 1.0], vec![2.0, 8.0]]
}

fn run_scalar_cpu_baseline() -> f64 {
    evpi(&baseline_workload()).expect("scalar baseline workload should be valid")
}

#[test]
fn scalar_cpu_baseline_returns_expected_value() {
    let result = run_scalar_cpu_baseline();
    assert_eq!(result, 3.0);
}

#[test]
fn scalar_cpu_baseline_runs_a_repeatable_workload_shape() {
    let mut total = 0.0;

    for _ in 0..10_000 {
        total += black_box(run_scalar_cpu_baseline());
    }
    assert!(total > 0.0);
}

#[derive(Debug, Deserialize)]
struct ScalarCpuBaselineArtifact {
    benchmark_name: String,
    metric_type: String,
    workload: ScalarCpuBaselineWorkload,
    expected: ScalarCpuBaselineExpected,
    metadata: ScalarCpuBaselineMetadata,
}

#[derive(Debug, Deserialize)]
struct ScalarCpuBaselineWorkload {
    seed: u64,
    repeats: u64,
    net_benefits: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct ScalarCpuBaselineExpected {
    evpi: f64,
    comparison_rule: String,
    regression_policy: String,
}

#[derive(Debug, Deserialize)]
struct ScalarCpuBaselineMetadata {
    phase: String,
    notes: Vec<String>,
}

#[test]
fn scalar_cpu_baseline_artifact_matches_contract() {
    let artifact: ScalarCpuBaselineArtifact =
        serde_json::from_str(include_str!("scalar_cpu_baseline.json"))
            .expect("baseline artifact should be valid JSON");

    assert_eq!(artifact.benchmark_name, "scalar_cpu_baseline");
    assert_eq!(artifact.metric_type, "scalar_cpu");
    assert_eq!(artifact.workload.seed, 42);
    assert_eq!(artifact.workload.repeats, 10_000);
    assert_eq!(artifact.workload.net_benefits, baseline_workload());
    assert_eq!(artifact.expected.evpi, 3.0);
    assert_eq!(artifact.expected.comparison_rule, "exact");
    assert_eq!(artifact.expected.regression_policy, "ci-contract-only");
    assert_eq!(artifact.metadata.phase, "phase-1-scalar-cpu-baseline");
    assert!(!artifact.metadata.notes.is_empty());
}
