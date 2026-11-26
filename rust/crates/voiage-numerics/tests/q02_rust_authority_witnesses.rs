//! Independent EVPI, weighted-EVPI, and ENBS authority witnesses for Q02.

use voiage_domain::{SampleCube, SampleMatrix, SampleVector};
use voiage_numerics::{enbs, evpi, structural_evpi};

#[test]
fn evpi_and_weighted_evpi_match_independent_q02_witnesses() {
    // Payoff rows [[0, 100], [100, 0]] -> equal weights give 100 - 50 = 50.
    let equal_matrix =
        SampleMatrix::try_from(vec![vec![0.0, 100.0], vec![100.0, 0.0]]).expect("valid matrix");
    let equal_evpi = evpi(&equal_matrix).expect("equal evpi");
    assert!((equal_evpi - 50.0).abs() <= 1.0e-12);

    // Weights [0.9, 0.1] give E[max] = 100, max E = max(10, 90) = 90 -> EVPI = 10.
    let weighted_cube = SampleCube::try_from(vec![vec![vec![0.0, 100.0]], vec![vec![100.0, 0.0]]])
        .expect("valid cube");
    let probs = SampleVector::try_from(vec![0.9, 0.1]).expect("valid probabilities");
    let weighted_evpi = structural_evpi(&weighted_cube, &probs).expect("weighted evpi");
    assert!((weighted_evpi - 10.0).abs() <= 1.0e-12);
}

#[test]
fn unsupported_weights_and_malformed_inputs_fail_explicitly() {
    let cube =
        SampleCube::try_from(vec![vec![vec![0.0, 100.0]], vec![vec![100.0, 0.0]]]).expect("cube");

    let negative = SampleVector::try_from(vec![-0.1, 1.1]).expect("finite vector");
    assert!(structural_evpi(&cube, &negative).is_err());

    let bad_sum = SampleVector::try_from(vec![0.5, 0.4]).expect("finite vector");
    assert!(structural_evpi(&cube, &bad_sum).is_err());

    assert!(SampleMatrix::try_from(vec![vec![f64::NAN, 1.0]]).is_err());
    assert!(SampleMatrix::try_from(vec![vec![f64::INFINITY, 1.0]]).is_err());
    assert!(SampleMatrix::try_from(vec![vec![1.0, 2.0], vec![3.0]]).is_err());
    assert!(SampleMatrix::try_from(Vec::<Vec<f64>>::new()).is_err());
    assert!(enbs(10.0, -1.0).is_err());
    assert!(enbs(f64::NAN, 1.0).is_err());
}

#[test]
fn translation_scaling_ties_extreme_finite_and_enbs_identities_hold() {
    let base = SampleMatrix::try_from(vec![vec![0.0, 100.0], vec![100.0, 0.0]]).expect("base");
    let shifted =
        SampleMatrix::try_from(vec![vec![1.0e9, 1.0e9 + 100.0], vec![1.0e9 + 100.0, 1.0e9]])
            .expect("shifted");
    let scaled = SampleMatrix::try_from(vec![vec![0.0, 250.0], vec![250.0, 0.0]]).expect("scaled");
    let ties = SampleMatrix::try_from(vec![vec![5.0, 5.0], vec![5.0, 5.0]]).expect("ties");

    assert!((evpi(&base).expect("base") - 50.0).abs() <= 1.0e-12);
    assert!((evpi(&shifted).expect("shifted") - 50.0).abs() <= 1.0e-9);
    assert!((evpi(&scaled).expect("scaled") - 125.0).abs() <= 1.0e-12);
    assert_eq!(evpi(&ties).expect("ties"), 0.0);

    let big = 0.75 * f64::MAX;
    let extreme =
        SampleMatrix::try_from(vec![vec![big, 0.0], vec![0.5 * big, big]]).expect("extreme");
    let expected = 0.25 * big;
    let actual = evpi(&extreme).expect("extreme evpi");
    assert!((actual - expected).abs() / expected <= 1.0e-12);

    assert!((enbs(50.0 * 100.0, 6000.0).expect("enbs") - (-1000.0)).abs() <= 1.0e-12);
}

#[test]
fn catastrophic_cancellation_and_pre_clamp_overflow_are_handled_consistently() {
    let cancelling = SampleMatrix::try_from(vec![
        vec![1.0e20, 1.0e20],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0e20, -1.0e20],
    ])
    .expect("cancelling");
    let evpi_val = evpi(&cancelling).expect("cancelling evpi");
    assert!((evpi_val - 0.25).abs() <= 1.0e-12);

    // Three rows of f64::MAX cause M/3 * 3 to round above f64::MAX -> must fail rather than clamp NaN to 0.
    let overflow = SampleMatrix::try_from(vec![
        vec![f64::MAX, 0.0],
        vec![0.0, f64::MAX],
        vec![0.0, f64::MAX],
    ])
    .expect("finite matrix");
    assert!(evpi(&overflow).is_err());
}
