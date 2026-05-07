use voiage_core::{
    enbs, enbs_contract, evpi, evpi_contract, EnbsDiagnostics, EvpiDiagnostics, ScalarError,
};

#[test]
fn evpi_contract_returns_expected_fixture_result() {
    let fixture = vec![vec![10.0, 1.0], vec![2.0, 8.0]];
    let result = evpi_contract(&fixture).expect("fixture should be valid");

    assert_eq!(result.value, 3.0);
    assert_eq!(
        result.diagnostics,
        EvpiDiagnostics {
            sample_count: 2,
            strategy_count: 2,
            min_row_width: 2,
            max_row_width: 2,
            finite: true,
        }
    );
    assert_eq!(result.reporting.contract_version, "rust-core-scalar-v1");
    assert_eq!(result.reporting.method, "evpi");
    assert!(result.reporting.deterministic);
    assert_eq!(result.reporting.status, "complete");
    assert_eq!(result.reporting.policy, "sample_mean_vs_row_max");
}

#[test]
fn evpi_wrapper_preserves_fixture_value() {
    let fixture = vec![vec![10.0, 1.0], vec![2.0, 8.0]];
    let got = evpi(&fixture).expect("fixture should be valid");

    assert_eq!(got, 3.0);
}

#[test]
fn evpi_handles_single_strategy_and_empty_inputs() {
    let empty = evpi_contract(&[]).expect("empty input is a valid deterministic baseline");
    assert_eq!(empty.value, 0.0);
    assert_eq!(empty.diagnostics.sample_count, 0);
    assert_eq!(empty.diagnostics.strategy_count, 0);

    let single = evpi_contract(&[vec![4.0], vec![7.0]]).expect("single strategy is valid");
    assert_eq!(single.value, 0.0);
    assert_eq!(single.diagnostics.strategy_count, 1);
}

#[test]
fn evpi_rejects_ragged_rows() {
    let err = evpi_contract(&[vec![1.0], vec![1.0, 2.0]]).expect_err("ragged rows");
    assert_eq!(
        err,
        ScalarError::RaggedRows {
            expected: 1,
            actual: 2,
            row_index: 1,
        }
    );
    assert_eq!(
        err.message(),
        "net_benefits rows must have a consistent width"
    );
}

#[test]
fn enbs_contract_returns_expected_fixture_result() {
    let result = enbs_contract(12.5, 5.0).expect("fixture should be valid");

    assert_eq!(result.value, 7.5);
    assert_eq!(
        result.diagnostics,
        EnbsDiagnostics {
            evsi_result: 12.5,
            research_cost: 5.0,
            finite: true,
            non_negative_research_cost: true,
        }
    );
    assert_eq!(result.reporting.contract_version, "rust-core-scalar-v1");
    assert_eq!(result.reporting.method, "enbs");
    assert!(result.reporting.deterministic);
    assert_eq!(result.reporting.status, "complete");
    assert_eq!(result.reporting.policy, "raw_subtraction");
}

#[test]
fn enbs_wrapper_preserves_raw_difference() {
    let got = enbs(4.0, 7.0).expect("valid scalar inputs");
    assert_eq!(got, -3.0);
}

#[test]
fn enbs_rejects_negative_or_non_finite_costs() {
    let negative = enbs_contract(10.0, -1.0).expect_err("negative cost");
    assert_eq!(negative, ScalarError::NegativeResearchCost);
    assert_eq!(negative.message(), "research_cost cannot be negative");

    let non_finite = enbs_contract(f64::INFINITY, 1.0).expect_err("non-finite");
    assert_eq!(non_finite, ScalarError::NonFiniteScalarInput);
    assert_eq!(non_finite.message(), "inputs must be finite numbers");
}
