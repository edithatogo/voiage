//! Behavioral contracts for expected opportunity loss.

use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_domain::SampleMatrix;
use voiage_numerics::expected_loss;

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

#[test]
fn expected_loss_matches_the_analytical_fixture() {
    let samples = matrix(&[&[10.0, 12.0], &[11.0, 9.0], &[13.0, 14.0]]);

    let result = expected_loss(&samples).expect("expected loss should be computable");

    assert_eq!(result.optimal_strategy_index, 1);
    assert_eq!(result.sample_count, 3);
    assert_eq!(result.strategy_count, 2);
    assert!((result.expected_net_benefit_by_strategy[0] - 34.0 / 3.0).abs() <= 1.0e-12);
    assert!((result.expected_net_benefit_by_strategy[1] - 35.0 / 3.0).abs() <= 1.0e-12);
    assert!((result.expected_opportunity_loss_by_strategy[0] - 1.0).abs() <= 1.0e-12);
    assert!((result.expected_opportunity_loss_by_strategy[1] - 2.0 / 3.0).abs() <= 1.0e-12);
    assert!((result.minimum_expected_opportunity_loss - 2.0 / 3.0).abs() <= 1.0e-12);
}

#[test]
fn expected_loss_uses_the_lowest_index_for_exact_ties() {
    let samples = matrix(&[&[0.0, 2.0], &[2.0, 0.0]]);

    let result = expected_loss(&samples).expect("expected loss should be computable");

    assert_eq!(result.optimal_strategy_index, 0);
    assert_eq!(result.expected_opportunity_loss_by_strategy, vec![1.0, 1.0]);
    assert!((result.minimum_expected_opportunity_loss - 1.0).abs() <= 1.0e-12);
}

#[test]
fn expected_loss_is_zero_for_a_single_strategy() {
    let samples = matrix(&[&[-2.0], &[4.0], &[1.0]]);

    let result = expected_loss(&samples).expect("expected loss should be computable");

    assert_eq!(result.optimal_strategy_index, 0);
    assert_eq!(result.expected_opportunity_loss_by_strategy, vec![0.0]);
    assert!(result.minimum_expected_opportunity_loss.abs() <= 1.0e-12);
}

#[test]
fn expected_loss_rejects_finite_inputs_that_overflow() {
    let difference_overflow = matrix(&[&[f64::MAX, -f64::MAX]]);
    let accumulation_overflow = matrix(&[&[f64::MAX], &[f64::MAX]]);

    for samples in [&difference_overflow, &accumulation_overflow] {
        let error = expected_loss(samples).expect_err("non-finite result must fail");
        assert_eq!(error.code(), ErrorCode::InvalidInput);
        assert_eq!(error.category(), ErrorCategory::Input);
    }
}
