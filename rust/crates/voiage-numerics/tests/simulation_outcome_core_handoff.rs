//! Integration tests for lossless simulation-outcome handoff to core Rust VOI contracts (`O05`).

use serde_json::Value;
use voiage_domain::{
    ObjectiveDirection, SimulationOutcomeProfileInput, ValidatedSimulationOutcomeProfile,
    ValueConvention, WeightSemantics,
};
use voiage_numerics::{
    adapt_simulation_outcome, adapt_simulation_outcome_input, SimulationOutcomeAdapterError,
};

fn load_evpi_fixtures() -> Value {
    let raw = include_str!("../../../../specs/ecosystem/simulation-outcome/v1/evpi-fixtures.json");
    serde_json::from_str(raw).expect("valid O07 fixture JSON")
}

#[test]
fn health_with_parameters_and_non_health_without_parameters_handoff_losslessly() {
    let fixtures = load_evpi_fixtures();
    let accepted = fixtures["accepted"]
        .as_array()
        .expect("accepted fixture array");

    let health_input: SimulationOutcomeProfileInput =
        serde_json::from_value(accepted[0]["profile"].clone()).expect("health profile input");
    let health_validated =
        ValidatedSimulationOutcomeProfile::validate(&health_input).expect("valid health profile");
    let health_handoff = adapt_simulation_outcome(&health_validated).expect("health handoff");

    assert_eq!(
        health_handoff.profile_version(),
        "voiage.simulation-outcome.v1"
    );
    assert_eq!(health_handoff.provenance(), health_validated.provenance());
    assert_eq!(
        health_handoff.value_declaration(),
        health_validated.value_declaration()
    );
    assert_eq!(health_handoff.weight_semantics(), WeightSemantics::Equal);
    assert_eq!(
        health_handoff.draw_identities(),
        health_validated.draw_identities().as_slice()
    );
    assert_eq!(
        health_handoff.draw_weights(),
        health_validated.draw_weights()
    );
    assert_eq!(
        health_handoff
            .value_array()
            .strategies()
            .iter()
            .collect::<Vec<_>>(),
        vec!["standard_care", "intervention"]
    );
    assert_eq!(health_handoff.value_array().sample_count(), 4);
    assert_eq!(
        health_handoff
            .value_array()
            .net_benefit()
            .rows()
            .collect::<Vec<_>>(),
        health_validated
            .payoff_matrix()
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>()
    );
    let params = health_handoff
        .parameter_set()
        .expect("health fixture parameter set");
    assert_eq!(params.sample_count(), 4);
    assert_eq!(
        params.parameter("risk").expect("risk vector").as_slice(),
        &[0.1, 0.2, 0.3, 0.4]
    );
    assert_eq!(
        params
            .parameter("cost_mult")
            .expect("cost_mult vector")
            .as_slice(),
        &[1.0, 1.1, 1.2, 1.3]
    );

    let non_health_input: SimulationOutcomeProfileInput =
        serde_json::from_value(accepted[1]["profile"].clone()).expect("non-health profile input");
    let non_health_handoff =
        adapt_simulation_outcome_input(&non_health_input).expect("non-health handoff");
    assert_eq!(
        non_health_handoff.weight_semantics(),
        WeightSemantics::Probability
    );
    assert_eq!(
        non_health_handoff.value_declaration().convention(),
        ValueConvention::Utility
    );
    assert_eq!(
        non_health_handoff.value_declaration().objective(),
        ObjectiveDirection::Maximize
    );
    assert!(non_health_handoff.parameter_set().is_none());
    assert_eq!(non_health_handoff.draw_weights(), &[0.2, 0.5, 0.3]);
}

#[test]
fn reordered_input_rows_preserve_canonical_handoff_and_reject_invalid_profiles() {
    let fixtures = load_evpi_fixtures();
    let baseline: SimulationOutcomeProfileInput =
        serde_json::from_value(fixtures["accepted"][0]["profile"].clone())
            .expect("baseline profile");
    let reordered: SimulationOutcomeProfileInput =
        serde_json::from_value(fixtures["joint_permutation_invariant"]["profile"].clone())
            .expect("reordered profile");

    let base_handoff = adapt_simulation_outcome_input(&baseline).expect("baseline handoff");
    let reordered_handoff = adapt_simulation_outcome_input(&reordered).expect("reordered handoff");
    assert_eq!(
        base_handoff.value_array().net_benefit(),
        reordered_handoff.value_array().net_benefit()
    );
    assert_eq!(
        base_handoff.draw_identities(),
        reordered_handoff.draw_identities()
    );

    let mut invalid = baseline;
    invalid.profile_version = "unsupported.v0".to_owned();
    let err = adapt_simulation_outcome_input(&invalid).expect_err("unsupported version");
    assert!(matches!(err, SimulationOutcomeAdapterError::Domain(_)));
}

#[test]
fn relative_weight_and_minimize_loss_handoff_preserve_values_without_sign_flip() {
    let fixtures = load_evpi_fixtures();
    let relative_input: SimulationOutcomeProfileInput =
        serde_json::from_value(fixtures["accepted"][2]["profile"].clone())
            .expect("relative weight profile");
    let relative_handoff =
        adapt_simulation_outcome_input(&relative_input).expect("relative handoff");
    assert_eq!(
        relative_handoff.weight_semantics(),
        WeightSemantics::RelativeWeight
    );
    assert_eq!(relative_handoff.draw_weights(), &[0.0, 1.0, 3.0, 6.0]);

    let loss_input: SimulationOutcomeProfileInput =
        serde_json::from_value(fixtures["rejected_minimize_loss"].clone()).expect("loss profile");
    let loss_handoff = adapt_simulation_outcome_input(&loss_input)
        .expect("O05 handoff accepts valid loss profiles without sign conversion");
    assert_eq!(
        loss_handoff.value_declaration().convention(),
        ValueConvention::Loss
    );
    assert_eq!(
        loss_handoff.value_declaration().objective(),
        ObjectiveDirection::Minimize
    );
    assert_eq!(
        loss_handoff.value_array().net_benefit().row(0),
        Some(&[5.0, 8.0][..])
    );
}
