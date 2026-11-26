//! O02 distinguishing tests for paired strategy/draw identity and weight consistency.

use serde::Deserialize;
use voiage_domain::{
    DrawIdentity, SimulationOutcomeError, SimulationOutcomeProfileInput,
    ValidatedSimulationOutcomeProfile, WeightSemantics,
};

#[derive(Deserialize)]
struct PairingFixtures {
    base_profile: SimulationOutcomeProfileInput,
    expected_draw_keys: Vec<String>,
    expected_strategy_ids: Vec<String>,
    expected_payoff_matrix: Vec<Vec<f64>>,
    expected_draw_weights: Vec<f64>,
}

fn load_fixtures() -> PairingFixtures {
    let raw =
        include_str!("../../../../specs/ecosystem/simulation-outcome/v1/pairing-fixtures.json");
    serde_json::from_str(raw).expect("valid pairing-fixtures.json")
}

#[test]
fn reordered_complete_keyed_records_preserve_canonical_mapping() {
    let fixtures = load_fixtures();
    let baseline = ValidatedSimulationOutcomeProfile::validate(&fixtures.base_profile)
        .expect("baseline pairing fixture validates");

    let keys: Vec<&str> = baseline
        .draw_keys()
        .iter()
        .map(voiage_domain::Identifier::as_str)
        .collect();
    let strategies: Vec<&str> = baseline
        .strategy_ids()
        .iter()
        .map(voiage_domain::StrategyId::as_str)
        .collect();
    assert_eq!(keys, fixtures.expected_draw_keys);
    assert_eq!(strategies, fixtures.expected_strategy_ids);
    assert_eq!(baseline.payoff_matrix(), fixtures.expected_payoff_matrix);
    assert_eq!(baseline.draw_weights(), fixtures.expected_draw_weights);
    assert_eq!(
        baseline.draw_identities(),
        vec![
            DrawIdentity::new("ns.paired.v1", "draw-001").unwrap(),
            DrawIdentity::new("ns.paired.v1", "draw-002").unwrap(),
        ]
    );

    let mut reversed = fixtures.base_profile.clone();
    reversed.outcomes.reverse();
    let reordered = ValidatedSimulationOutcomeProfile::validate(&reversed)
        .expect("reordered complete keyed records preserve pairing");
    assert_eq!(reordered, baseline);
}

#[test]
fn duplicate_missing_unknown_untrimmed_and_conflicting_pairings_fail_explicitly() {
    let fixtures = load_fixtures();

    let mut duplicate = fixtures.base_profile.clone();
    duplicate.outcomes.push(duplicate.outcomes[0].clone());
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&duplicate),
        Err(SimulationOutcomeError::DuplicateOutcomeRecord {
            draw_key: "draw-001".to_owned(),
            strategy_id: "alpha".to_owned(),
        })
    );

    let mut incomplete = fixtures.base_profile.clone();
    incomplete.outcomes.pop();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&incomplete),
        Err(SimulationOutcomeError::IncompleteDraw {
            draw_key: "draw-002".to_owned(),
            missing_strategy: "beta".to_owned(),
        })
    );

    let mut unknown_strategy = fixtures.base_profile.clone();
    unknown_strategy.outcomes[0].strategy_id = "gamma".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&unknown_strategy),
        Err(SimulationOutcomeError::UnknownStrategy("gamma".to_owned()))
    );

    let mut blank_strategy = fixtures.base_profile.clone();
    blank_strategy.outcomes[0].strategy_id = "   ".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&blank_strategy),
        Err(SimulationOutcomeError::EmptyIdentifier("strategy_id"))
    );

    let mut untrimmed_strategy = fixtures.base_profile.clone();
    untrimmed_strategy.outcomes[0].strategy_id = " alpha".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&untrimmed_strategy),
        Err(SimulationOutcomeError::UntrimmedIdentifier("strategy_id"))
    );

    let mut blank_draw_key = fixtures.base_profile.clone();
    blank_draw_key.outcomes[0].draw_key = "   ".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&blank_draw_key),
        Err(SimulationOutcomeError::EmptyIdentifier("draw_key"))
    );

    let mut untrimmed_draw_key = fixtures.base_profile.clone();
    untrimmed_draw_key.outcomes[0].draw_key = " draw-001".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&untrimmed_draw_key),
        Err(SimulationOutcomeError::UntrimmedIdentifier("draw_key"))
    );

    let mut mismatched_ns = fixtures.base_profile.clone();
    mismatched_ns.outcomes[0].draw_namespace = "ns.other.v1".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&mismatched_ns),
        Err(SimulationOutcomeError::MismatchedDrawNamespace {
            expected: "ns.paired.v1".to_owned(),
            actual: "ns.other.v1".to_owned(),
        })
    );

    let mut non_finite_val = fixtures.base_profile.clone();
    non_finite_val.outcomes[0].value = f64::INFINITY;
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&non_finite_val),
        Err(SimulationOutcomeError::NonFiniteValue {
            draw_key: "draw-001".to_owned(),
            strategy_id: "alpha".to_owned(),
        })
    );

    let mut conflicting_weight = fixtures.base_profile;
    conflicting_weight.outcomes[1].weight = 0.30;
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&conflicting_weight),
        Err(SimulationOutcomeError::ConflictingDrawWeight(
            "draw-001".to_owned()
        ))
    );
}

#[test]
fn weight_semantics_and_edge_cases_are_enforced() {
    let fixtures = load_fixtures();

    // Zero weight is allowed under Probability and RelativeWeight, rejected under Equal.
    let mut zero_prob = fixtures.base_profile.clone();
    zero_prob.outcomes[0].weight = -0.0;
    zero_prob.outcomes[1].weight = 0.0;
    zero_prob.outcomes[2].weight = 1.0;
    zero_prob.outcomes[3].weight = 1.0;
    let validated_zero_prob =
        ValidatedSimulationOutcomeProfile::validate(&zero_prob).expect("zero weight probability");
    assert_eq!(validated_zero_prob.draw_weights(), &[0.0, 1.0]);
    assert!(validated_zero_prob.draw_weights()[0].is_sign_positive());

    let mut rel_weight = zero_prob.clone();
    rel_weight.weight_semantics = "relative_weight".to_owned();
    rel_weight.outcomes[2].weight = 2.5;
    rel_weight.outcomes[3].weight = 2.5;
    let validated_rel =
        ValidatedSimulationOutcomeProfile::validate(&rel_weight).expect("relative weights");
    assert_eq!(
        validated_rel.weight_semantics(),
        WeightSemantics::RelativeWeight
    );

    // Equal rejects [1e-13, 0.0] and [1e-15, 2e-15], accepts identical positive weights.
    let mut equal_ok = fixtures.base_profile.clone();
    equal_ok.weight_semantics = "equal".to_owned();
    for o in &mut equal_ok.outcomes {
        o.weight = 1.0;
    }
    assert!(ValidatedSimulationOutcomeProfile::validate(&equal_ok).is_ok());

    let mut equal_zero = equal_ok.clone();
    equal_zero.outcomes[0].weight = 1e-13;
    equal_zero.outcomes[1].weight = 1e-13;
    equal_zero.outcomes[2].weight = 0.0;
    equal_zero.outcomes[3].weight = 0.0;
    assert!(matches!(
        ValidatedSimulationOutcomeProfile::validate(&equal_zero),
        Err(SimulationOutcomeError::IncompatibleWeightSemantics(_))
    ));

    let mut equal_tiny_mismatch = equal_ok;
    equal_tiny_mismatch.outcomes[0].weight = 1e-15;
    equal_tiny_mismatch.outcomes[1].weight = 1e-15;
    equal_tiny_mismatch.outcomes[2].weight = 2e-15;
    equal_tiny_mismatch.outcomes[3].weight = 2e-15;
    assert!(matches!(
        ValidatedSimulationOutcomeProfile::validate(&equal_tiny_mismatch),
        Err(SimulationOutcomeError::IncompatibleWeightSemantics(_))
    ));

    // RelativeWeight rejects overflow to infinity.
    let mut rel_overflow = rel_weight;
    for o in &mut rel_overflow.outcomes {
        o.weight = f64::MAX;
    }
    assert!(matches!(
        ValidatedSimulationOutcomeProfile::validate(&rel_overflow),
        Err(SimulationOutcomeError::IncompatibleWeightSemantics(_))
    ));

    // Probability rejects non-unit sum; negative/NaN weights fail InvalidWeight; unknown semantics fails.
    let mut bad_prob_sum = fixtures.base_profile.clone();
    bad_prob_sum.outcomes[0].weight = 0.4;
    bad_prob_sum.outcomes[1].weight = 0.4;
    assert!(matches!(
        ValidatedSimulationOutcomeProfile::validate(&bad_prob_sum),
        Err(SimulationOutcomeError::IncompatibleWeightSemantics(_))
    ));

    let mut neg_weight = fixtures.base_profile.clone();
    neg_weight.outcomes[0].weight = -0.1;
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&neg_weight),
        Err(SimulationOutcomeError::InvalidWeight("draw-001".to_owned()))
    );

    let mut nan_weight = fixtures.base_profile.clone();
    nan_weight.outcomes[0].weight = f64::NAN;
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&nan_weight),
        Err(SimulationOutcomeError::InvalidWeight("draw-001".to_owned()))
    );

    let mut bad_semantics = fixtures.base_profile;
    bad_semantics.weight_semantics = " equal".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&bad_semantics),
        Err(SimulationOutcomeError::UnsupportedWeightSemantics(
            " equal".to_owned()
        ))
    );
}
