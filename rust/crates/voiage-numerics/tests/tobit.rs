//! Analytical controls for left-censored Gaussian observations and finite EVSI.

use std::num::NonZeroUsize;

use voiage_domain::{BigRational, Identifier, SampleMatrix};
use voiage_numerics::{sequential_evsi, SequentialEvsi, TobitModel, TobitObservation, TobitState};

fn id(name: &str) -> Identifier {
    Identifier::new(name).unwrap()
}

fn state(mean: f64, sigma: f64, observed: f64) -> TobitState {
    TobitState::new(mean, sigma, observed).unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected}"
    );
}

#[test]
fn censored_mass_density_and_unavailability_have_distinct_likelihoods() {
    let model = TobitModel::new(0.0, vec![state(0.0, 2.0, 0.5), state(0.0, 1.0, 1.0)]).unwrap();
    let censored = model.log_likelihoods(TobitObservation::BelowLimit).unwrap();
    let observed = model
        .log_likelihoods(TobitObservation::Reading(0.0))
        .unwrap();
    let unavailable = model
        .log_likelihoods(TobitObservation::Unavailable)
        .unwrap();
    close(censored[0].exp(), 0.25, 1e-15);
    close(censored[1].exp(), 0.5, 1e-15);
    close(
        observed[0].exp(),
        0.25 / (2.0 * std::f64::consts::PI).sqrt(),
        1e-15,
    );
    close(
        observed[1].exp(),
        1.0 / (2.0 * std::f64::consts::PI).sqrt(),
        1e-15,
    );
    close(unavailable[0], 0.5_f64.ln(), 1e-15);
    assert!(unavailable[1].is_infinite() && unavailable[1].is_sign_negative());
    assert!(model
        .log_likelihoods(TobitObservation::Reading(-0.5))
        .is_err());
    assert!(model
        .log_likelihoods(TobitObservation::Reading(f64::INFINITY))
        .is_err());
}

#[test]
fn log_likelihood_retains_a_far_left_tail_and_handles_large_standardization() {
    let model = TobitModel::new(0.0, vec![state(40.0, 1.0, 1.0)]).unwrap();
    let tail = model.log_likelihoods(TobitObservation::BelowLimit).unwrap();
    close(tail[0], -804.608_442_013_753_8, 2e-12);

    let scaled = TobitModel::new(f64::MAX, vec![state(-f64::MAX, f64::MAX, 1.0)]).unwrap();
    let log_likelihood = scaled
        .log_likelihoods(TobitObservation::BelowLimit)
        .unwrap();
    close(log_likelihood[0].exp(), 0.977_249_868_051_820_8, 2e-15);
}

#[test]
fn generated_finite_channels_measure_the_difference_between_acquisition_actions() {
    let coarse = TobitModel::new(0.0, vec![state(-1.0, 2.0, 1.0), state(1.0, 2.0, 1.0)]).unwrap();
    let sharper = TobitModel::new(0.0, vec![state(-1.0, 1.0, 1.0), state(1.0, 1.0, 1.0)]).unwrap();
    let labels = vec![id("unavailable"), id("below"), id("above")];
    let coarse = coarse
        .finite_channel(id("coarse"), &[], labels.clone())
        .unwrap();
    let sharper = sharper
        .finite_channel(id("sharper"), &[], labels.clone())
        .unwrap();
    assert_eq!(sharper.outcomes(), labels);
    assert_eq!(sharper.likelihoods()[0][0], BigRational::default());
    let payoffs = SampleMatrix::try_from(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    let value = |choices| {
        let SequentialEvsi::Complete { evsi, .. } = sequential_evsi(
            &[1.0, 1.0],
            &payoffs,
            &[choices],
            NonZeroUsize::new(32).unwrap(),
        )
        .unwrap() else {
            panic!("finite channels fit the analytical budget")
        };
        evsi
    };
    let sharp_value = value(vec![sharper.clone()]);
    let expected = BigRational::from_float(0.341_344_746_068_542_9).unwrap();
    let error = &sharp_value - expected;
    let tolerance = BigRational::from_float(2e-15).unwrap();
    assert!(error > -&tolerance && error < tolerance);
    assert!(sharp_value > value(vec![coarse.clone()]));
    let pooling = SampleMatrix::try_from(vec![vec![1.0]; 3]).unwrap();
    let pooled = sharper.garble(vec![id("pooled")], &pooling).unwrap();
    assert_eq!(value(vec![pooled]), BigRational::default());
    assert_eq!(value(vec![coarse, sharper]), sharp_value);
}

#[test]
fn identical_censored_likelihoods_have_exactly_zero_sequential_information() {
    let model = TobitModel::new(0.0, vec![state(0.4, 0.8, 0.7); 2]).unwrap();
    let labels = ["unavailable", "below", "middle", "upper"].map(id);
    let channel = model
        .finite_channel(id("assay"), &[1.0], labels.to_vec())
        .unwrap();
    let payoffs = SampleMatrix::try_from(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    let outcome = sequential_evsi(
        &[1.0, 1.0],
        &payoffs,
        &[vec![channel.clone()], vec![channel]],
        NonZeroUsize::new(20).unwrap(),
    )
    .unwrap();
    assert!(
        matches!(outcome, SequentialEvsi::Complete { evsi, .. } if evsi == BigRational::default())
    );
}

#[test]
fn ordered_bins_partition_the_detected_probability_on_both_sides_of_the_mean() {
    let model = TobitModel::new(-1.0, vec![state(0.0, 1.0, 0.8)]).unwrap();
    let labels = ["unavailable", "below", "negative", "positive", "upper"].map(id);
    let channel = model
        .finite_channel(id("assay"), &[0.0, 1.0], labels.to_vec())
        .unwrap();
    let tail = 0.8 * 0.158_655_253_931_457_07;
    let bin = 0.4 - tail;
    let tolerance = BigRational::from_float(2e-15).unwrap();
    for (actual, expected) in channel.likelihoods()[0]
        .iter()
        .zip([0.2, tail, bin, bin, tail])
    {
        let error = actual - BigRational::from_float(expected).unwrap();
        assert!(error > -&tolerance && error < tolerance);
    }
}

#[test]
fn narrow_measured_bins_keep_probability_when_both_cumulative_values_round_together() {
    let cases = [
        (state(0.0, 1.0, 1.0), 1e-20, 3.989_422_804_014_327e-21),
        (state(1e16, 1e16, 1.0), 1.0, 2.419_707_245_191_433_7e-17),
    ];
    for (signal, cutpoint, mass) in cases {
        let model = TobitModel::new(0.0, vec![signal]).unwrap();
        let labels = ["unavailable", "below", "narrow", "upper"].map(id);
        let channel = model
            .finite_channel(id("assay"), &[cutpoint], labels.to_vec())
            .unwrap();
        let error = &channel.likelihoods()[0][2] - BigRational::from_float(mass).unwrap();
        let tolerance = BigRational::from_float(mass * 1e-13).unwrap();
        assert!(error > -&tolerance && error < tolerance);
    }
}

#[test]
fn invalid_models_empty_bins_and_unrepresentable_weights_fail_closed() {
    for (mean, sigma, observed) in [(f64::NAN, 1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.1)] {
        assert!(TobitState::new(mean, sigma, observed).is_err());
    }
    assert!(TobitModel::new(f64::INFINITY, vec![state(0.0, 1.0, 1.0)]).is_err());
    assert!(TobitModel::new(0.0, vec![]).is_err());
    let model = TobitModel::new(0.0, vec![state(0.0, 1.0, 0.0)]).unwrap();
    let labels = vec![id("unavailable"), id("below"), id("first"), id("last")];
    assert!(model
        .finite_channel(id("assay"), &[0.0], labels.clone())
        .is_err());
    let result = model.finite_channel(id("assay"), &[1.0], labels).unwrap();
    assert_eq!(
        result.likelihoods()[0][0],
        BigRational::from_integer(1.into())
    );
    assert!(result.likelihoods()[0][1..]
        .iter()
        .all(|weight| weight == &BigRational::default()));
    let tail = TobitModel::new(0.0, vec![state(40.0, 1.0, 1.0)]).unwrap();
    assert!(tail
        .finite_channel(
            id("tail"),
            &[],
            vec![id("unavailable"), id("below"), id("above")]
        )
        .is_err());
}
