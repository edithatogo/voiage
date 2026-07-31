//! Property contracts for estimation-focused variance-reduction kernels.

use proptest::prelude::*;
use voiage_domain::SampleVector;
use voiage_numerics::{evppi_variance, evsi_variance_with_assurance};

fn samples(values: &[i16]) -> SampleVector {
    values
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>()
        .try_into()
        .expect("generated samples are finite and non-empty")
}

proptest! {
    #[test]
    fn evppi_variance_is_translation_invariant_and_bounded(
        raw in prop::collection::vec(-1_000_i16..=1_000, 2..=64),
        translation in -10_000_i16..=10_000,
    ) {
        let groups = (0..raw.len())
            .map(|index| format!("group-{}", index % 4))
            .collect::<Vec<_>>();
        let translated = raw
            .iter()
            .map(|value| f64::from(*value) + f64::from(translation))
            .collect::<Vec<_>>()
            .try_into()
            .expect("translated samples remain finite");
        let baseline = evppi_variance(&samples(&raw), &groups).expect("valid EVPPI-var");
        let shifted = evppi_variance(&translated, &groups).expect("valid shifted EVPPI-var");

        prop_assert!((baseline.raw_reduction - shifted.raw_reduction).abs() <= 1.0e-8);
        prop_assert!(baseline.absolute_reduction >= 0.0);
        prop_assert!(baseline.absolute_reduction <= baseline.prior_variance + 1.0e-8);
    }

    #[test]
    fn evppi_variance_scales_quadratically(
        raw in prop::collection::vec(-100_i16..=100, 2..=64),
        factor in 1_u8..=20,
    ) {
        let groups = (0..raw.len())
            .map(|index| format!("group-{}", index % 3))
            .collect::<Vec<_>>();
        let scaled = raw
            .iter()
            .map(|value| f64::from(*value) * f64::from(factor))
            .collect::<Vec<_>>()
            .try_into()
            .expect("scaled samples remain finite");
        let baseline = evppi_variance(&samples(&raw), &groups).expect("valid EVPPI-var");
        let scaled_result = evppi_variance(&scaled, &groups).expect("valid scaled EVPPI-var");
        let expected = baseline.raw_reduction * f64::from(factor).powi(2);

        prop_assert!((scaled_result.raw_reduction - expected).abs() <= 1.0e-7);
    }

    #[test]
    fn evsi_assurance_is_seed_reproducible(
        raw in prop::collection::vec(-100_i16..=100, 2..=32),
        posterior in prop::collection::vec(0_u16..=10_000, 2..=16),
        seed in any::<u64>(),
    ) {
        let prior = samples(&raw);
        let posterior: SampleVector = posterior
            .iter()
            .map(|value| f64::from(*value) / 10.0)
            .collect::<Vec<_>>()
            .try_into()
            .expect("posterior variances are finite");
        let first = evsi_variance_with_assurance(&prior, &posterior, 16, seed, 0.1)
            .expect("valid assured EVSI-var");
        let second = evsi_variance_with_assurance(&prior, &posterior, 16, seed, 0.1)
            .expect("same assured EVSI-var");

        prop_assert_eq!(first, second);
    }
}
