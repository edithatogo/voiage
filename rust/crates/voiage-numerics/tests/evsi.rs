//! Behavioral contracts for the retained seeded-bootstrap EVSI kernel.

use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_domain::SampleMatrix;
use voiage_numerics::evsi_stochastic;

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

#[test]
fn seeded_bootstrap_evsi_matches_the_committed_fixture() {
    let net_benefit = matrix(&[&[10.0, 4.0], &[8.0, 6.0], &[6.0, 8.0], &[4.0, 10.0]]);

    let result = evsi_stochastic(&net_benefit, 2, 4, 42).expect("EVSI should be computable");

    assert_eq!(result.estimator, "seeded_bootstrap");
    assert_eq!(result.contract_version, 1);
    assert_eq!(result.draw_count, 2);
    assert_eq!(result.resample_count, 4);
    assert!((result.expected_current_value - 7.0).abs() <= 1.0e-12);
    assert!((result.expected_sample_value - 7.75).abs() <= 1.0e-12);
    assert!((result.expected_perfect_information - 9.0).abs() <= 1.0e-12);
    assert!((result.evsi - 0.75).abs() <= 1.0e-12);
}

#[test]
fn seeded_bootstrap_uses_scaled_accumulation_for_large_finite_values() {
    let net_benefit = matrix(&[
        &[f64::MAX, f64::MAX],
        &[f64::MAX, f64::MAX],
        &[f64::MAX, f64::MAX],
    ]);

    let result = evsi_stochastic(&net_benefit, 2, 4, 42)
        .expect("finite repeated maxima must not overflow their means");

    assert!(result.expected_current_value > f64::MAX * 0.999);
    assert!(result.expected_sample_value > f64::MAX * 0.999);
    assert!(result.expected_perfect_information > f64::MAX * 0.999);
    assert!(result.evsi.abs() <= 1.0e-12);
}

#[test]
fn seeded_bootstrap_evsi_rejects_invalid_loop_sizes() {
    let net_benefit = matrix(&[&[1.0, 0.0], &[0.0, 1.0]]);

    for (trial_sample_size, resample_count) in [(0, 4), (2, 0)] {
        let error = evsi_stochastic(&net_benefit, trial_sample_size, resample_count, 42)
            .expect_err("invalid loop sizes must fail");
        assert_eq!(error.code(), ErrorCode::InvalidInput);
        assert_eq!(error.category(), ErrorCategory::Input);
    }
}

#[test]
fn seeded_bootstrap_evsi_rejects_perfect_information_overflow() {
    let net_benefit = matrix(&[&[f64::MAX, -f64::MAX], &[-f64::MAX, f64::MAX]]);

    let error = evsi_stochastic(&net_benefit, 1, 1, 42)
        .expect_err("finite inputs with an overflowing perfect-information sum must fail");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.category(), ErrorCategory::Input);
}

#[test]
fn seeded_bootstrap_evsi_handles_zero_state_seed_edge() {
    let net_benefit = matrix(&[&[10.0, 4.0], &[8.0, 6.0], &[6.0, 8.0], &[4.0, 10.0]]);

    let first = evsi_stochastic(&net_benefit, 2, 4, 0x9E37_79B9_7F4A_7C15)
        .expect("the zero-state seed edge must remain valid");
    let second = evsi_stochastic(&net_benefit, 2, 4, 0x9E37_79B9_7F4A_7C15)
        .expect("the zero-state seed edge must be reproducible");

    assert_eq!(first, second);
    assert!(first.evsi.is_finite());
}
