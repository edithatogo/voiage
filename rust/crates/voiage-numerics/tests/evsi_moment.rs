//! Contract tests for deterministic moment-based EVSI.

use voiage_diagnostics::ErrorCode;
use voiage_domain::SampleMatrix;
use voiage_numerics::evsi_moment_based;

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

#[test]
fn moment_based_matches_the_normative_fixture() {
    let parameters = matrix(&[&[-1.0], &[0.0], &[1.0], &[2.0]]);
    let net_benefit = matrix(&[&[1.0, 2.0], &[0.0, 3.0], &[1.0, 2.0], &[4.0, -1.0]]);

    let result = evsi_moment_based(&net_benefit, &parameters, 2)
        .expect("moment-based EVSI should be computable");

    assert_eq!(result.estimator, "moment_based");
    assert_eq!(result.contract_version, 1);
    assert!((result.information_fraction - (1.0 / 3.0)).abs() <= 1.0e-12);
    assert!((result.expected_current_value - 1.5).abs() <= 1.0e-12);
    assert!((result.expected_perfect_information - 2.75).abs() <= 1.0e-12);
    assert!((result.expected_sample_value - (23.0 / 12.0)).abs() <= 1.0e-12);
    assert!((result.evsi - (5.0 / 12.0)).abs() <= 1.0e-12);
}

#[test]
fn moment_based_rejects_rank_deficient_designs() {
    let parameters = matrix(&[&[1.0], &[1.0]]);
    let net_benefit = matrix(&[&[0.0, 1.0], &[1.0, 0.0]]);

    let error = evsi_moment_based(&net_benefit, &parameters, 2)
        .expect_err("rank-deficient moment designs must fail closed");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
}
