//! Contract tests for deterministic efficient-linear EVSI.

use voiage_domain::SampleMatrix;
use voiage_numerics::evsi_efficient_linear;

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

#[test]
fn efficient_linear_matches_the_normative_fixture() {
    let parameters = matrix(&[&[-1.0], &[0.0], &[1.0], &[2.0]]);
    let net_benefit = matrix(&[&[0.0, 6.0], &[2.0, 4.0], &[4.0, 2.0], &[6.0, 0.0]]);

    let result = evsi_efficient_linear(&net_benefit, &parameters, 2)
        .expect("efficient-linear EVSI should be computable");

    assert_eq!(result.estimator, "efficient_linear");
    assert_eq!(result.contract_version, 1);
    assert!((result.information_fraction - (1.0 / 3.0)).abs() <= 1.0e-12);
    assert!((result.expected_current_value - 3.0).abs() <= 1.0e-12);
    assert!((result.expected_perfect_information - 5.0).abs() <= 1.0e-12);
    assert!((result.expected_sample_value - (11.0 / 3.0)).abs() <= 1.0e-12);
    assert!((result.evsi - (2.0 / 3.0)).abs() <= 1.0e-12);
}

#[test]
fn efficient_linear_uses_scaled_accumulation_for_large_finite_values() {
    let parameters = matrix(&[&[0.0], &[1.0], &[2.0]]);
    let net_benefit = matrix(&[&[f64::MAX], &[f64::MAX], &[f64::MAX]]);

    let result = evsi_efficient_linear(&net_benefit, &parameters, 1)
        .expect("finite repeated maxima must not overflow their mean");

    assert!(result.expected_current_value > f64::MAX * 0.999);
    assert!(result.expected_perfect_information > f64::MAX * 0.999);
    assert!(result.evsi.abs() <= 1.0e-12);
}

#[test]
fn efficient_linear_handles_rank_deficient_designs() {
    let parameters = matrix(&[&[1.0], &[1.0]]);
    let net_benefit = matrix(&[&[0.0, 1.0], &[1.0, 0.0]]);

    let result = evsi_efficient_linear(&net_benefit, &parameters, 2)
        .expect("rank-aware native contract should remain finite");

    assert!(result.expected_sample_value.is_finite());
    assert!(result.evsi.is_finite());
}
