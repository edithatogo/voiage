//! Behavioral contracts for the retained regression-based EVPPI kernel.

use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_domain::SampleMatrix;
use voiage_numerics::evppi;

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

#[test]
fn evppi_matches_the_canonical_linear_fixture() {
    let net_benefit = matrix(&[&[0.0, 2.0], &[1.0, 0.0], &[2.0, 1.0], &[3.0, 4.0]]);
    let parameters = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);

    let result = evppi(&net_benefit, &parameters).expect("EVPPI should be computable");

    assert!((result - 0.05).abs() <= 1.0e-10);
}

#[test]
fn evppi_is_zero_for_identical_or_single_strategy_samples() {
    let parameters = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let identical = matrix(&[&[2.0, 2.0], &[3.0, 3.0], &[-1.0, -1.0], &[4.0, 4.0]]);
    let single = matrix(&[&[2.0], &[3.0], &[-1.0], &[4.0]]);

    assert_eq!(evppi(&identical, &parameters), Ok(0.0));
    assert_eq!(evppi(&single, &parameters), Ok(0.0));
}

#[test]
fn evppi_rejects_parameter_sample_dimension_mismatch() {
    let net_benefit = matrix(&[&[0.0, 1.0], &[1.0, 0.0], &[2.0, 1.0]]);
    let parameters = matrix(&[&[0.0], &[1.0]]);

    let error = evppi(&net_benefit, &parameters).expect_err("sample counts must match");

    assert_eq!(error.code(), ErrorCode::DimensionMismatch);
    assert_eq!(error.category(), ErrorCategory::DimensionMismatch);
}

#[test]
fn evppi_handles_small_scale_predictors_without_rank_loss() {
    let net_benefit = matrix(&[&[0.0, 1.0], &[1.0, 0.0]]);
    let parameters = matrix(&[&[0.0], &[1.0e-7]]);

    let result = evppi(&net_benefit, &parameters).expect("small-scale regression");

    assert!((result - 0.5).abs() <= 1.0e-10);
}

#[test]
fn evppi_rejects_finite_inputs_that_overflow_the_result() {
    let net_benefit = matrix(&[&[f64::MAX, -f64::MAX], &[f64::MAX, -f64::MAX]]);
    let parameters = matrix(&[&[0.0], &[1.0]]);

    let error = evppi(&net_benefit, &parameters).expect_err("overflow must fail closed");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.category(), ErrorCategory::Input);
}
