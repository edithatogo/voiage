//! Contract tests for the centered moment-based interaction design.

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
fn moment_based_uses_centered_interaction_order_and_is_deterministic() {
    // The means are zero. The documented two-parameter order is:
    // intercept, x, y, x^2, y^2, and x*y.
    let parameters = matrix(&[
        &[-1.0, -1.0],
        &[-1.0, 0.0],
        &[-1.0, 1.0],
        &[0.0, -1.0],
        &[0.0, 1.0],
        &[1.0, -1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ]);
    let centered = [-1.0, 1.0];
    let documented_row = [
        1.0,
        centered[0],
        centered[1],
        centered[0] * centered[0],
        centered[1] * centered[1],
        centered[0] * centered[1],
    ];
    let expected_row: [f64; 6] = [1.0, -1.0, 1.0, 1.0, 1.0, -1.0];
    for (actual, expected) in documented_row.iter().zip(expected_row) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }

    // Strategy zero is exactly the final x*y interaction column; strategy one
    // is the zero baseline. This makes the interaction contribution observable
    // in the normative EVSI result.
    let net_benefit = matrix(&[
        &[1.0, 0.0],
        &[0.0, 0.0],
        &[-1.0, 0.0],
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[-1.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 0.0],
    ]);

    let first = evsi_moment_based(&net_benefit, &parameters, 2)
        .expect("full-rank interaction design should be computable");
    let second = evsi_moment_based(&net_benefit, &parameters, 2)
        .expect("repeated interaction calculation should be computable");

    assert_eq!(first, second);
    assert_eq!(first.estimator, "moment_based");
    assert_eq!(first.contract_version, 1);
    assert!((first.expected_current_value - 0.0).abs() <= 1.0e-12);
    assert!((first.expected_perfect_information - 0.25).abs() <= 1.0e-12);
    assert!((first.expected_sample_value - 0.05).abs() <= 1.0e-12);
    assert!((first.evsi - 0.05).abs() <= 1.0e-12);
}
