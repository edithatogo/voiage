//! Analytical and property evidence for Rust-owned net-benefit construction.

use proptest::prelude::*;
use voiage_numerics::{net_benefit, WtpMode};

#[test]
fn elementwise_compatibility_is_explicit_and_rust_owned() {
    let result = net_benefit(
        &[100.0, 150.0, 90.0, 140.0],
        &[0.5, 0.6, 0.45, 0.55],
        &[15_000.0, 20_000.0, 16_000.0, 18_000.0],
        WtpMode::LegacyElementwise,
    )
    .expect("matching finite arrays");
    assert_eq!(result.values, vec![7_400.0, 11_850.0, 7_110.0, 9_760.0]);
}

proptest! {
    #[test]
    fn scalar_kernel_matches_the_analytical_definition(
        cost in -1.0e6_f64..1.0e6,
        effect in -1.0e3_f64..1.0e3,
        threshold in -1.0e5_f64..1.0e5,
    ) {
        let result = net_benefit(&[cost], &[effect], &[threshold], WtpMode::Scalar)
            .expect("bounded finite values");
        prop_assert_eq!(result.values, vec![effect * threshold - cost]);
    }

    #[test]
    fn adding_a_cost_constant_translates_net_benefit(
        cost in -1.0e5_f64..1.0e5,
        effect in -1.0e2_f64..1.0e2,
        threshold in -1.0e4_f64..1.0e4,
        offset in -1.0e4_f64..1.0e4,
    ) {
        let baseline = net_benefit(&[cost], &[effect], &[threshold], WtpMode::Scalar)
            .expect("bounded finite values").values[0];
        let shifted = net_benefit(&[cost + offset], &[effect], &[threshold], WtpMode::Scalar)
            .expect("bounded finite values").values[0];
        prop_assert!(((shifted - baseline) + offset).abs() < 1e-8);
    }
}
