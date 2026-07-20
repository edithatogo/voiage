//! Behavioral contracts for deterministic foundational numerical kernels.

use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_domain::SampleMatrix;
use voiage_numerics::{enbs, evpi};

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

#[test]
fn evpi_matches_the_canonical_fixture() {
    let samples = matrix(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

    let result = evpi(&samples).expect("EVPI should be computable");

    assert!((result - (2.0 / 3.0)).abs() <= 1.0e-12);
}

#[test]
fn evpi_is_exactly_zero_for_identical_or_single_strategies() {
    let identical = matrix(&[&[2.0, 2.0], &[3.0, 3.0], &[-1.0, -1.0]]);
    let single = matrix(&[&[2.0], &[3.0], &[-1.0]]);

    assert_eq!(evpi(&identical), Ok(0.0));
    assert_eq!(evpi(&single), Ok(0.0));
}

#[test]
fn evpi_is_invariant_to_order_and_translation() {
    let baseline = matrix(&[&[3.0, 0.0], &[0.0, 2.0], &[1.0, 1.0]]);
    let row_permuted = matrix(&[&[1.0, 1.0], &[3.0, 0.0], &[0.0, 2.0]]);
    let column_permuted = matrix(&[&[0.0, 3.0], &[2.0, 0.0], &[1.0, 1.0]]);
    let translated = matrix(&[&[10.0, 7.0], &[7.0, 9.0], &[8.0, 8.0]]);
    let expected = evpi(&baseline).expect("baseline EVPI");

    assert_eq!(evpi(&row_permuted), Ok(expected));
    assert_eq!(evpi(&column_permuted), Ok(expected));
    assert!((evpi(&translated).expect("translated EVPI") - expected).abs() <= 1.0e-12);
}

#[test]
fn enbs_preserves_raw_negative_results_and_break_even() {
    assert_eq!(enbs(12.5, 5.0), Ok(7.5));
    assert_eq!(enbs(5.0, 5.0), Ok(0.0));
    assert_eq!(enbs(2.0, 5.0), Ok(-3.0));
}

#[test]
fn enbs_rejects_negative_cost_with_stable_error_identity() {
    let error = enbs(12.5, -1.0).expect_err("negative cost must fail");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.category(), ErrorCategory::Input);
}

#[test]
fn enbs_rejects_non_finite_inputs_with_stable_error_identity() {
    for (evsi, cost) in [
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (1.0, f64::NEG_INFINITY),
    ] {
        let error = enbs(evsi, cost).expect_err("non-finite input must fail");
        assert_eq!(error.code(), ErrorCode::InvalidInput);
        assert_eq!(error.category(), ErrorCategory::Input);
    }
}
