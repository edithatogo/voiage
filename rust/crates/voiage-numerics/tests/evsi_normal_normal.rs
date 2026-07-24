//! Scientific-reference tests for the normal-normal EVSI kernel.

use voiage_numerics::normal_normal_two_arm_evsi;

#[test]
fn normal_normal_two_arm_evsi_matches_the_health_reference() {
    let value = normal_normal_two_arm_evsi(0.060, 0.030, 1.0, 200, 50_000.0, -3_000.0)
        .expect("declared study is valid");

    assert!((value - 124.179_365_520_623_8).abs() < 1e-5);
}

#[test]
fn normal_normal_two_arm_evsi_is_bounded_and_responds_to_information() {
    let small = normal_normal_two_arm_evsi(0.060, 0.030, 1.0, 50, 50_000.0, -3_000.0)
        .expect("declared study is valid");
    let large = normal_normal_two_arm_evsi(0.060, 0.030, 1.0, 1_200, 50_000.0, -3_000.0)
        .expect("declared study is valid");

    assert!(small >= 0.0);
    assert!(large > small);
    assert!(large < 598.5);
}

#[test]
fn normal_normal_two_arm_evsi_matches_off_centre_references() {
    let positive = normal_normal_two_arm_evsi(0.05, 0.02, 1.0, 200, 50_000.0, -2_430.0)
        .expect("positive-slope study is valid");
    let negative = normal_normal_two_arm_evsi(0.05, 0.02, 1.0, 200, -50_000.0, 2_430.0)
        .expect("negative-slope study is valid");

    assert!((positive - 27.701_379_070_215_58).abs() < 1e-4);
    assert!((negative - 27.701_379_070_215_58).abs() < 1e-4);
}

#[test]
fn normal_normal_two_arm_evsi_rejects_undefined_study_models() {
    assert!(normal_normal_two_arm_evsi(0.0, 0.0, 1.0, 200, 1.0, 0.0).is_err());
    assert!(normal_normal_two_arm_evsi(0.0, 1.0, 0.0, 200, 1.0, 0.0).is_err());
    assert!(normal_normal_two_arm_evsi(0.0, 1.0, 1.0, 0, 1.0, 0.0).is_err());
    assert!(normal_normal_two_arm_evsi(0.0, 1.0, 1.0, 51, 1.0, 0.0).is_err());
    assert!(normal_normal_two_arm_evsi(f64::NAN, 1.0, 1.0, 200, 1.0, 0.0).is_err());
}
