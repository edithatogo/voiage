//! Behavioral contracts for the deterministic CEAF kernel.

use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_domain::{SampleCube, SampleVector};
use voiage_numerics::ceaf;

fn cube(values: Vec<Vec<Vec<f64>>>) -> SampleCube {
    values.try_into().expect("valid sample cube")
}

fn vector(values: &[f64]) -> SampleVector {
    values.to_vec().try_into().expect("valid sample vector")
}

#[test]
fn ceaf_matches_the_canonical_fixture() {
    let result = ceaf(
        &cube(vec![
            vec![vec![5.0, 1.0], vec![0.0, 2.0]],
            vec![vec![5.0, 1.0], vec![0.0, 2.0]],
        ]),
        &vector(&[0.0, 1.0]),
        0.95,
    )
    .expect("CEAF should be computable");

    assert_eq!(result.wtp_thresholds, [0.0, 1.0]);
    assert_eq!(result.optimal_strategy_indices, [0, 1]);
    assert_eq!(result.acceptability_probabilities, [1.0, 1.0]);
    assert_eq!(result.probability_lower, [1.0, 1.0]);
    assert_eq!(result.probability_upper, [1.0, 1.0]);
    assert_eq!(result.expected_net_benefit, [5.0, 2.0]);
}

#[test]
fn exact_ties_select_the_lowest_original_strategy_index() {
    let result = ceaf(
        &cube(vec![vec![vec![5.0], vec![5.0]], vec![vec![5.0], vec![5.0]]]),
        &vector(&[0.0]),
        0.95,
    )
    .expect("ties are valid");

    assert_eq!(result.optimal_strategy_indices, [0]);
    assert_eq!(result.acceptability_probabilities, [1.0]);
}

#[test]
fn threshold_count_mismatch_has_dimension_identity() {
    let error = ceaf(
        &cube(vec![vec![vec![5.0, 1.0], vec![0.0, 2.0]]]),
        &vector(&[0.0]),
        0.95,
    )
    .expect_err("threshold mismatch must fail");

    assert_eq!(error.code(), ErrorCode::DimensionMismatch);
    assert_eq!(error.category(), ErrorCategory::DimensionMismatch);
}

#[test]
fn confidence_level_must_be_finite_and_strictly_between_zero_and_one() {
    let values = cube(vec![vec![vec![5.0]]]);
    let thresholds = vector(&[0.0]);
    for confidence in [0.0, 1.0, f64::NAN, f64::INFINITY] {
        let error =
            ceaf(&values, &thresholds, confidence).expect_err("invalid confidence level must fail");
        assert_eq!(error.code(), ErrorCode::InvalidInput);
    }
}
