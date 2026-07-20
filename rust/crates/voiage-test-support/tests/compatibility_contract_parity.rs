//! Executable Phase 3 compatibility contract parity tests.

use std::collections::{BTreeMap, BTreeSet};

use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_test_support::{
    classify_compatibility_contracts, execute_deterministic_compatibility_contracts,
    execute_foundational_compatibility_contracts, load_compatibility_cases,
    CompatibilityClassification, CompatibilityMethod, ContractOutcome,
};

#[test]
fn classifies_all_25_fixtures_through_rust_contracts_without_running_kernels() {
    let cases = load_compatibility_cases().expect("canonical cases must load");
    let reports = classify_compatibility_contracts(&cases)
        .expect("all canonical fixture shapes and mappings must validate");

    assert_eq!(reports.len(), 25);
    assert!(reports
        .iter()
        .all(|report| !report.numerical_kernel_executed));

    let methods: BTreeSet<_> = reports.iter().map(|report| report.method).collect();
    assert_eq!(
        methods,
        BTreeSet::from([
            CompatibilityMethod::Evpi,
            CompatibilityMethod::Evppi,
            CompatibilityMethod::Evsi,
            CompatibilityMethod::Enbs,
            CompatibilityMethod::Ceaf,
            CompatibilityMethod::Dominance,
        ])
    );

    let classifications: BTreeSet<_> = reports.iter().map(|report| report.classification).collect();
    assert_eq!(
        classifications,
        BTreeSet::from([
            CompatibilityClassification::Normal,
            CompatibilityClassification::Edge,
            CompatibilityClassification::Invalid,
        ])
    );
}

#[test]
fn preserves_exact_method_classification_and_error_code_mappings() {
    let cases = load_compatibility_cases().expect("canonical cases must load");
    let reports = classify_compatibility_contracts(&cases).expect("contracts must classify");
    let by_id: BTreeMap<_, _> = reports
        .iter()
        .map(|report| (report.case_id.as_str(), report))
        .collect();

    let invalid = [
        (
            "evpi-invalid-shape-001",
            "shape_mismatch",
            ErrorCode::DimensionMismatch,
        ),
        (
            "evppi-invalid-parameter-001",
            "unknown_parameter",
            ErrorCode::InvalidInput,
        ),
        (
            "evsi-invalid-estimator-001",
            "unsupported_estimator",
            ErrorCode::BackendUnavailable,
        ),
        (
            "enbs-invalid-cost-001",
            "negative_research_cost",
            ErrorCode::InvalidInput,
        ),
        (
            "ceaf-invalid-thresholds-001",
            "threshold_count_mismatch",
            ErrorCode::DimensionMismatch,
        ),
        (
            "dominance-invalid-length-001",
            "strategy_count_mismatch",
            ErrorCode::DimensionMismatch,
        ),
        (
            "dominance-nan-invalid-001",
            "non_finite_value",
            ErrorCode::InvalidInput,
        ),
        (
            "dominance-infinity-invalid-001",
            "non_finite_value",
            ErrorCode::InvalidInput,
        ),
        (
            "evpi-nan-invalid-001",
            "non_finite_value",
            ErrorCode::InvalidInput,
        ),
        (
            "evppi-nan-invalid-001",
            "non_finite_value",
            ErrorCode::InvalidInput,
        ),
        (
            "evsi-nan-invalid-001",
            "non_finite_value",
            ErrorCode::InvalidInput,
        ),
        (
            "enbs-nan-invalid-001",
            "non_finite_value",
            ErrorCode::InvalidInput,
        ),
        (
            "ceaf-nan-invalid-001",
            "non_finite_value",
            ErrorCode::InvalidInput,
        ),
    ];

    for (case_id, exact_code, stable_code) in invalid {
        let report = by_id[case_id];
        assert_eq!(report.classification, CompatibilityClassification::Invalid);
        match &report.outcome {
            ContractOutcome::Error(error) => {
                assert_eq!(error.fixture_code, exact_code);
                assert_eq!(error.stable_code, stable_code);
                assert_eq!(error.stable_category, stable_code.category());
            }
            ContractOutcome::Result => panic!("{case_id} must classify as an error"),
        }
    }

    assert_eq!(
        by_id["evsi-invalid-estimator-001"]
            .outcome
            .error()
            .expect("error report")
            .stable_category,
        ErrorCategory::BackendUnavailable
    );
}

#[test]
fn validates_every_success_result_shape_through_canonical_serialization_dtos() {
    let cases = load_compatibility_cases().expect("canonical cases must load");
    let reports = classify_compatibility_contracts(&cases).expect("contracts must classify");

    assert_eq!(
        reports
            .iter()
            .filter(|report| matches!(report.outcome, ContractOutcome::Result))
            .count(),
        12
    );
    assert!(reports.iter().all(|report| report.input_contract_validated));
    assert!(reports
        .iter()
        .all(|report| report.expected_contract_validated));
}

#[test]
fn executes_all_foundational_fixtures_and_preserves_invalid_error_codes() {
    let cases = load_compatibility_cases().expect("canonical cases must load");
    let reports = execute_foundational_compatibility_contracts(&cases)
        .expect("foundational kernels must match their canonical fixtures");

    assert_eq!(reports.len(), 8);
    assert_eq!(
        reports
            .iter()
            .filter(|report| report.numerical_kernel_executed)
            .count(),
        4
    );
    let invalid_codes = reports
        .iter()
        .filter_map(|report| report.outcome.error())
        .map(|error| error.fixture_code.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        invalid_codes,
        BTreeSet::from([
            "shape_mismatch",
            "non_finite_value",
            "negative_research_cost",
        ])
    );
}

#[test]
fn executes_all_deterministic_kernel_fixtures_and_preserves_invalid_mappings() {
    let cases = load_compatibility_cases().expect("canonical cases must load");
    let reports = execute_deterministic_compatibility_contracts(&cases)
        .expect("deterministic kernels must match canonical fixtures");

    assert_eq!(reports.len(), 21);
    assert_eq!(
        reports
            .iter()
            .filter(|report| report.numerical_kernel_executed)
            .count(),
        10
    );
    let expected_invalid = BTreeMap::from([
        ("ceaf-invalid-thresholds-001", "threshold_count_mismatch"),
        ("ceaf-nan-invalid-001", "non_finite_value"),
        ("dominance-invalid-length-001", "strategy_count_mismatch"),
        ("dominance-nan-invalid-001", "non_finite_value"),
        ("dominance-infinity-invalid-001", "non_finite_value"),
        ("evppi-invalid-parameter-001", "unknown_parameter"),
        ("evppi-nan-invalid-001", "non_finite_value"),
        ("evpi-invalid-shape-001", "shape_mismatch"),
        ("evpi-nan-invalid-001", "non_finite_value"),
        ("enbs-invalid-cost-001", "negative_research_cost"),
        ("enbs-nan-invalid-001", "non_finite_value"),
    ]);
    for (case_id, fixture_code) in expected_invalid {
        let report = reports
            .iter()
            .find(|report| report.case_id == case_id)
            .expect("expected invalid deterministic fixture");
        assert!(!report.numerical_kernel_executed);
        assert_eq!(
            report
                .outcome
                .error()
                .expect("invalid outcome")
                .fixture_code,
            fixture_code
        );
    }
}
