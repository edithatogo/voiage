//! Behavioral contracts for deterministic dominance and frontier kernels.

use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_domain::SampleVector;
use voiage_numerics::{dominance, DominanceStatus};

fn vector(values: &[f64]) -> SampleVector {
    values.to_vec().try_into().expect("valid sample vector")
}

#[test]
fn dominance_matches_the_canonical_fixture() {
    let result = dominance(
        &vector(&[100.0, 200.0, 500.0, 800.0, 900.0]),
        &vector(&[1.0, 2.0, 2.5, 4.0, 3.5]),
    )
    .expect("dominance should be computable");

    assert_eq!(result.frontier_indices, [0, 1, 3]);
    assert_eq!(result.strongly_dominated_indices, [4]);
    assert_eq!(result.extended_dominated_indices, [2]);
    assert_eq!(
        result.status,
        [
            DominanceStatus::Frontier,
            DominanceStatus::Frontier,
            DominanceStatus::ExtendedDominated,
            DominanceStatus::Frontier,
            DominanceStatus::StronglyDominated,
        ]
    );
    assert_eq!(result.incremental_costs, [100.0, 600.0]);
    assert_eq!(result.incremental_effects, [1.0, 2.0]);
    assert_eq!(result.icers, [100.0, 300.0]);
}

#[test]
fn two_strategy_frontier_is_preserved() {
    let result = dominance(&vector(&[100.0, 200.0]), &vector(&[1.0, 2.0]))
        .expect("dominance should be computable");

    assert_eq!(result.frontier_indices, [0, 1]);
    assert!(result.strongly_dominated_indices.is_empty());
    assert!(result.extended_dominated_indices.is_empty());
    assert_eq!(result.icers, [100.0]);
}

#[test]
fn cheaper_and_more_effective_strategies_strongly_dominate() {
    let result = dominance(&vector(&[10.0, 12.0, 9.0]), &vector(&[1.0, 0.9, 1.2]))
        .expect("dominance should be computable");

    assert_eq!(result.frontier_indices, [2]);
    assert_eq!(result.strongly_dominated_indices, [0, 1]);
}

#[test]
fn mismatched_lengths_use_dimension_error_identity() {
    let error =
        dominance(&vector(&[1.0, 2.0]), &vector(&[1.0])).expect_err("mismatched lengths must fail");

    assert_eq!(error.code(), ErrorCode::DimensionMismatch);
    assert_eq!(error.category(), ErrorCategory::DimensionMismatch);
}

#[test]
fn duplicate_strategy_retains_the_first_and_extends_the_second() {
    let result =
        dominance(&vector(&[10.0, 10.0]), &vector(&[1.0, 1.0])).expect("duplicates are valid");

    assert_eq!(result.frontier_indices, [0]);
    assert!(result.strongly_dominated_indices.is_empty());
    assert_eq!(result.extended_dominated_indices, [1]);
    assert!(result.icers.is_empty());
}

#[test]
fn frontier_uses_the_stable_numpy_effect_tolerance() {
    let close = dominance(&vector(&[10.0, 11.0]), &vector(&[1.0, 1.0 + 5.0e-6]))
        .expect("close effects are valid");
    let distinct = dominance(&vector(&[10.0, 11.0]), &vector(&[1.0, 1.0 + 2.0e-5]))
        .expect("distinct effects are valid");

    assert_eq!(close.frontier_indices, [0]);
    assert_eq!(close.extended_dominated_indices, [1]);
    assert_eq!(distinct.frontier_indices, [0, 1]);
}

#[test]
fn equal_adjacent_icers_remove_the_middle_strategy() {
    let result = dominance(&vector(&[100.0, 200.0, 300.0]), &vector(&[1.0, 2.0, 3.0]))
        .expect("dominance should be computable");

    assert_eq!(result.frontier_indices, [0, 2]);
    assert_eq!(result.extended_dominated_indices, [1]);
    assert_eq!(result.icers, [100.0]);
}

#[test]
fn fewer_than_two_strategies_is_an_input_error() {
    let error = dominance(&vector(&[1.0]), &vector(&[1.0]))
        .expect_err("one strategy must fail the dominance contract");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.category(), ErrorCategory::Input);
}
