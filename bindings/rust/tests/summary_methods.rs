use voiage_core::{
    calculate_ceaf, calculate_dominance, cost_effectiveness_frontier, value_of_heterogeneity,
    CeafDiagnostics, CeafResult, DominanceDiagnostics, DominanceResult, HeterogeneityDiagnostics,
    HeterogeneityResult, PartialError,
};

fn round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de> + PartialEq + core::fmt::Debug,
{
    let encoded = serde_json::to_string(value).expect("serializes");
    let decoded = serde_json::from_str(&encoded).expect("deserializes");
    assert_eq!(decoded, *value);
    decoded
}

#[test]
fn dominance_contract_round_trips_deterministically() {
    let result = calculate_dominance(
        &[10.0, 12.0, 13.0],
        &[1.0, 1.1, 1.3],
        Some(&["A".to_string(), "B".to_string(), "C".to_string()]),
    )
    .expect("valid dominance fixture");

    assert_eq!(result.frontier_indices, vec![0, 2]);
    assert_eq!(
        result.status,
        vec!["frontier", "extended_dominated", "frontier"]
    );
    assert_eq!(
        result.diagnostics,
        DominanceDiagnostics {
            strategy_count: 3,
            frontier_size: 2,
            strong_dominated_count: 0,
            extended_dominated_count: 1,
            finite: true,
        }
    );
    assert_eq!(result.reporting.contract_version, "rust-core-summary-v1");
    assert_eq!(result.reporting.method, "calculate_dominance");
    assert_eq!(result.reporting.policy, "deterministic_frontier");

    let decoded: DominanceResult = round_trip(&result);
    assert_eq!(decoded.strategy_names, vec!["A", "B", "C"]);
}

#[test]
fn dominance_frontier_helpers_are_consistent() {
    let frontier =
        cost_effectiveness_frontier(&[10.0, 12.0, 13.0], &[1.0, 1.1, 1.3]).expect("valid inputs");
    assert_eq!(frontier, vec![0, 2]);
}

#[test]
fn heterogeneity_contract_round_trips_deterministically() {
    let result = value_of_heterogeneity(
        &[
            vec![10.0, 8.0],
            vec![11.0, 7.0],
            vec![6.0, 12.0],
            vec![5.0, 13.0],
        ],
        &[
            "low".to_string(),
            "low".to_string(),
            "high".to_string(),
            "high".to_string(),
        ],
        Some(&["A".to_string(), "B".to_string()]),
    )
    .expect("valid heterogeneity fixture");

    assert_eq!(result.value, 1.5);
    assert_eq!(result.subgroup_labels, vec!["high", "low"]);
    assert_eq!(
        result.diagnostics,
        HeterogeneityDiagnostics {
            sample_count: 4,
            strategy_count: 2,
            subgroup_count: 2,
            finite: true,
        }
    );
    assert_eq!(result.reporting.method, "value_of_heterogeneity");
    assert_eq!(result.reporting.policy, "subgroup_optimal_expected_value");

    let decoded: HeterogeneityResult = round_trip(&result);
    assert_eq!(decoded.overall_optimal_strategy_name, "B");
}

#[test]
fn ceaf_contract_round_trips_deterministically() {
    let result = calculate_ceaf(
        &[
            vec![vec![10.0, 11.0], vec![12.0, 9.0]],
            vec![vec![9.0, 10.5], vec![11.0, 10.0]],
        ],
        &[10000.0, 20000.0],
        Some(&["A".to_string(), "B".to_string()]),
    )
    .expect("valid ceaf fixture");

    assert_eq!(result.wtp_thresholds, vec![10000.0, 20000.0]);
    assert_eq!(result.optimal_strategy_indices, vec![1, 0]);
    assert_eq!(result.acceptability_probabilities, vec![1.0, 1.0]);
    assert_eq!(
        result.diagnostics,
        CeafDiagnostics {
            sample_count: 2,
            strategy_count: 2,
            threshold_count: 2,
            finite: true,
        }
    );
    assert_eq!(result.reporting.method, "calculate_ceaf");
    assert_eq!(result.reporting.policy, "frontier_probability");

    let decoded: CeafResult = round_trip(&result);
    assert_eq!(decoded.optimal_strategy_names, vec!["B", "A"]);
}

#[test]
fn partial_summary_rejects_mismatched_lengths() {
    let err = calculate_dominance(&[10.0, 12.0], &[1.0], None).expect_err("length mismatch");
    assert_eq!(
        err,
        PartialError::LengthMismatch {
            expected: 2,
            actual: 1,
            field: "effects",
        }
    );

    let err = value_of_heterogeneity(
        &[vec![10.0, 8.0]],
        &["low".to_string(), "high".to_string()],
        None,
    )
    .expect_err("sample mismatch");
    assert_eq!(
        err,
        PartialError::LengthMismatch {
            expected: 1,
            actual: 2,
            field: "subgroups",
        }
    );
}
