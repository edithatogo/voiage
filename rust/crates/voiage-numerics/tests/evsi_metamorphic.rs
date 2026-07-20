//! Metamorphic invariants for the retained native EVSI estimators.

use voiage_domain::SampleMatrix;
use voiage_numerics::{
    evsi_efficient_linear, evsi_moment_based, evsi_stochastic, EvsiApproximationResult,
    NumericalInputError,
};

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

fn shifted(rows: &[&[f64]], offset: f64, factor: f64) -> SampleMatrix {
    rows.iter()
        .map(|row| row.iter().map(|value| value * factor + offset).collect())
        .collect::<Vec<Vec<f64>>>()
        .try_into()
        .expect("valid transformed matrix")
}

fn assert_close(left: f64, right: f64) {
    assert!((left - right).abs() <= 1.0e-10);
}

fn assert_deterministic_invariants(
    estimator: fn(
        &SampleMatrix,
        &SampleMatrix,
        usize,
    ) -> Result<EvsiApproximationResult, NumericalInputError>,
    parameters: &SampleMatrix,
    raw: &[&[f64]],
) {
    let base = estimator(&matrix(raw), parameters, 2).expect("base deterministic EVSI");
    let shifted_result =
        estimator(&shifted(raw, 7.0, 1.0), parameters, 2).expect("shifted deterministic EVSI");
    let scaled_result =
        estimator(&shifted(raw, 0.0, 2.5), parameters, 2).expect("scaled deterministic EVSI");

    assert_close(
        shifted_result.expected_current_value,
        base.expected_current_value + 7.0,
    );
    assert_close(
        shifted_result.expected_sample_value,
        base.expected_sample_value + 7.0,
    );
    assert_close(
        shifted_result.expected_perfect_information,
        base.expected_perfect_information + 7.0,
    );
    assert_close(shifted_result.evsi, base.evsi);
    assert_close(
        scaled_result.expected_current_value,
        base.expected_current_value * 2.5,
    );
    assert_close(
        scaled_result.expected_sample_value,
        base.expected_sample_value * 2.5,
    );
    assert_close(
        scaled_result.expected_perfect_information,
        base.expected_perfect_information * 2.5,
    );
    assert_close(scaled_result.evsi, base.evsi * 2.5);
}

#[test]
fn seeded_evsi_preserves_value_shift_and_positive_scale_invariants() {
    let raw = [&[10.0, 4.0][..], &[8.0, 6.0], &[6.0, 8.0], &[4.0, 10.0]];
    let base = evsi_stochastic(&matrix(&raw), 2, 4, 42).expect("base EVSI");
    let shifted_result = evsi_stochastic(&shifted(&raw, 7.0, 1.0), 2, 4, 42).expect("shifted EVSI");
    let scaled_result = evsi_stochastic(&shifted(&raw, 0.0, 2.5), 2, 4, 42).expect("scaled EVSI");

    assert_close(
        shifted_result.expected_current_value,
        base.expected_current_value + 7.0,
    );
    assert_close(
        shifted_result.expected_sample_value,
        base.expected_sample_value + 7.0,
    );
    assert_close(
        shifted_result.expected_perfect_information,
        base.expected_perfect_information + 7.0,
    );
    assert_close(shifted_result.evsi, base.evsi);
    assert_close(
        scaled_result.expected_current_value,
        base.expected_current_value * 2.5,
    );
    assert_close(
        scaled_result.expected_sample_value,
        base.expected_sample_value * 2.5,
    );
    assert_close(
        scaled_result.expected_perfect_information,
        base.expected_perfect_information * 2.5,
    );
    assert_close(scaled_result.evsi, base.evsi * 2.5);
}

#[test]
fn deterministic_evsi_estimators_preserve_value_shift_and_scale_invariants() {
    let parameters = matrix(&[&[-1.0], &[0.0], &[1.0], &[2.0]]);
    let raw = [&[1.0, 2.0][..], &[0.0, 3.0], &[1.0, 2.0], &[4.0, -1.0]];

    assert_deterministic_invariants(evsi_efficient_linear, &parameters, &raw);
    assert_deterministic_invariants(evsi_moment_based, &parameters, &raw);
}
