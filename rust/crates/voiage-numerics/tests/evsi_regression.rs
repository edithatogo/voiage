#![allow(missing_docs)]

use voiage_domain::SampleMatrix;
use voiage_numerics::evsi_regression;

#[test]
fn regression_matches_linear_fixture_and_reports_envelope() {
    let targets = SampleMatrix::try_from(vec![vec![1.0], vec![3.0], vec![5.0]]).unwrap();
    let parameters = SampleMatrix::try_from(vec![vec![0.0], vec![1.0], vec![2.0]]).unwrap();
    let predictions = SampleMatrix::try_from(vec![vec![-1.0], vec![3.0]]).unwrap();

    let result = evsi_regression(&targets, &parameters, &predictions).unwrap();

    assert_eq!(result.estimator, "regression");
    assert_eq!(result.contract_version, 1);
    assert_eq!(result.sample_count, 3);
    assert_eq!(result.prediction_count, 2);
    assert_eq!(result.parameter_count, 1);
    assert!((result.expected_sample_value - 3.0).abs() < 1e-12);
}

#[test]
fn regression_handles_rank_deficient_design() {
    let targets = SampleMatrix::try_from(vec![vec![1.0], vec![2.0]]).unwrap();
    let parameters = SampleMatrix::try_from(vec![vec![1.0], vec![1.0]]).unwrap();
    let predictions = SampleMatrix::try_from(vec![vec![1.0]]).unwrap();

    let result = evsi_regression(&targets, &parameters, &predictions).unwrap();

    assert!(result.expected_sample_value.is_finite());
}
