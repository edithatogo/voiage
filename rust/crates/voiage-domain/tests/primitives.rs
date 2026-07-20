//! Contract tests for validated scalar and identity primitives.

use proptest::prelude::*;
use voiage_domain::{Cost, Effect, Identifier, Probability, Seed, StrategyId, Threshold};

#[test]
fn identifiers_require_non_empty_trimmed_values() {
    assert!(Identifier::new("").is_err());
    assert!(Identifier::new("  \n").is_err());
    assert_eq!(
        Identifier::new(" analysis-1 ").unwrap().as_str(),
        "analysis-1"
    );
    assert!(StrategyId::new("\t").is_err());
    assert_eq!(
        StrategyId::new(" strategy-a ").unwrap().as_str(),
        "strategy-a"
    );
}

#[test]
fn numeric_primitives_enforce_their_domains() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Cost::new(value).is_err());
        assert!(Effect::new(value).is_err());
        assert!(Probability::new(value).is_err());
        assert!(Threshold::new(value).is_err());
    }
    assert_eq!(
        Cost::new(-12.5).unwrap().get().to_bits(),
        (-12.5_f64).to_bits()
    );
    assert_eq!(
        Effect::new(-0.25).unwrap().get().to_bits(),
        (-0.25_f64).to_bits()
    );
    assert_eq!(
        Probability::new(0.0).unwrap().get().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        Probability::new(1.0).unwrap().get().to_bits(),
        1.0_f64.to_bits()
    );
    assert!(Probability::new(-f64::EPSILON).is_err());
    assert!(Probability::new(1.0 + f64::EPSILON).is_err());
    assert!(Threshold::new(0.0).is_err());
    assert!(Threshold::new(-1.0).is_err());
    assert_eq!(
        Threshold::new(30_000.0).unwrap().get().to_bits(),
        30_000.0_f64.to_bits()
    );
}

#[test]
fn serde_deserialization_is_fail_closed() {
    assert!(serde_json::from_str::<Identifier>(r#"" ""#).is_err());
    assert!(serde_json::from_str::<StrategyId>(r#""""#).is_err());
    assert!(serde_json::from_str::<Probability>("-0.1").is_err());
    assert!(serde_json::from_str::<Probability>("1.1").is_err());
    assert!(serde_json::from_str::<Threshold>("0.0").is_err());
    assert!(serde_json::from_str::<Cost>(r#"{"value":1.0}"#).is_err());
    assert!(serde_json::from_str::<Seed>(r#"{"value":1}"#).is_err());

    let probability = Probability::new(0.25).unwrap();
    assert_eq!(serde_json::to_string(&probability).unwrap(), "0.25");
    assert_eq!(
        serde_json::from_str::<Probability>("0.25").unwrap(),
        probability
    );
    let seed = Seed::new(u64::MAX);
    assert_eq!(seed.get(), u64::MAX);
    assert_eq!(
        serde_json::from_str::<Seed>(&serde_json::to_string(&seed).unwrap()).unwrap(),
        seed
    );
}

proptest! {
    #[test]
    fn finite_costs_and_effects_round_trip(value in any::<f64>().prop_filter("finite", |v| v.is_finite())) {
        let cost = Cost::new(value).unwrap();
        let effect = Effect::new(value).unwrap();
        let decoded_cost = serde_json::from_str::<Cost>(&serde_json::to_string(&cost).unwrap()).unwrap();
        let decoded_effect = serde_json::from_str::<Effect>(&serde_json::to_string(&effect).unwrap()).unwrap();
        let tolerance = 1.0e-12 * value.abs().max(1.0);
        prop_assert!((decoded_cost.get() - cost.get()).abs() <= tolerance);
        prop_assert!((decoded_effect.get() - effect.get()).abs() <= tolerance);
    }

    #[test]
    fn probabilities_accept_exactly_finite_unit_interval(value in any::<f64>()) {
        prop_assert_eq!(Probability::new(value).is_ok(), value.is_finite() && (0.0..=1.0).contains(&value));
    }

    #[test]
    fn thresholds_accept_exactly_positive_finite_values(value in any::<f64>()) {
        prop_assert_eq!(Threshold::new(value).is_ok(), value.is_finite() && value > 0.0);
    }
}
