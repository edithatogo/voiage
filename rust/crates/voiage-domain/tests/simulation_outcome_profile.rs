//! O01 distinguishing tests for simulation-outcome profile version validation.

use serde::Deserialize;
use voiage_domain::{
    validate_profile_version, SimulationOutcomeError, SimulationOutcomeProfileInput,
    ValidatedSimulationOutcomeProfile, SIMULATION_OUTCOME_PROFILE_VERSION_V1,
};

#[derive(Deserialize)]
struct ProfileFixtures {
    accepted: Vec<serde_json::Value>,
    rejected_versions: Vec<String>,
}

fn load_fixtures() -> ProfileFixtures {
    let raw =
        include_str!("../../../../specs/ecosystem/simulation-outcome/v1/profile-fixtures.json");
    serde_json::from_str(raw).expect("valid profile-fixtures.json")
}

fn parse_accepted(entry: &serde_json::Value) -> (String, SimulationOutcomeProfileInput) {
    let mut obj = entry.as_object().expect("object").clone();
    let label = obj
        .remove("label")
        .and_then(|v| v.as_str().map(str::to_owned))
        .expect("string label");
    let profile: SimulationOutcomeProfileInput =
        serde_json::from_value(serde_json::Value::Object(obj)).expect("valid profile");
    (label, profile)
}

#[test]
fn accepted_v1_health_and_non_health_profiles_validate() {
    let fixtures = load_fixtures();
    assert_eq!(fixtures.accepted.len(), 2);
    for entry in &fixtures.accepted {
        let (label, profile) = parse_accepted(entry);
        validate_profile_version(&profile.profile_version)
            .unwrap_or_else(|err| panic!("{label} failed version check: {err}"));
        let validated = ValidatedSimulationOutcomeProfile::validate(&profile)
            .unwrap_or_else(|err| panic!("{label} failed profile validation: {err}"));
        assert_eq!(
            validated.profile_version(),
            SIMULATION_OUTCOME_PROFILE_VERSION_V1
        );
        assert_eq!(validated.draw_keys().len(), 2);
        assert_eq!(validated.draw_identities().len(), 2);
    }
}

#[test]
fn unknown_and_malformed_profile_versions_and_metadata_fail_explicitly() {
    let fixtures = load_fixtures();
    let (_, template) = parse_accepted(&fixtures.accepted[0]);

    for bad_version in fixtures.rejected_versions {
        assert_eq!(
            validate_profile_version(&bad_version),
            Err(SimulationOutcomeError::UnsupportedProfileVersion(
                bad_version.clone()
            ))
        );
        let mut mutated = template.clone();
        mutated.profile_version = bad_version.clone();
        assert_eq!(
            ValidatedSimulationOutcomeProfile::validate(&mutated),
            Err(SimulationOutcomeError::UnsupportedProfileVersion(
                bad_version
            ))
        );
    }

    let mut blank_producer = template.clone();
    blank_producer.provenance.producer_id = "  ".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&blank_producer),
        Err(SimulationOutcomeError::EmptyIdentifier("producer_id"))
    );

    let mut untrimmed_ns = template.clone();
    untrimmed_ns.draw_namespace = " heor.crn.v1".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&untrimmed_ns),
        Err(SimulationOutcomeError::UntrimmedIdentifier(
            "draw_namespace"
        ))
    );

    let mut empty_strategies = template.clone();
    empty_strategies.strategy_ids.clear();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&empty_strategies),
        Err(SimulationOutcomeError::InvalidStrategies)
    );

    let mut dup_strategies = template.clone();
    dup_strategies.strategy_ids.push("standard_care".to_owned());
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&dup_strategies),
        Err(SimulationOutcomeError::InvalidStrategies)
    );

    let mut untrimmed_strategy = template.clone();
    untrimmed_strategy.strategy_ids[0] = "standard_care ".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&untrimmed_strategy),
        Err(SimulationOutcomeError::UntrimmedIdentifier("strategy_ids"))
    );

    let mut empty_outcomes = template;
    empty_outcomes.outcomes.clear();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&empty_outcomes),
        Err(SimulationOutcomeError::EmptyOutcomes)
    );

    let unknown_field_json =
        r#"{"profile_version":"voiage.simulation-outcome.v1","unexpected":true}"#;
    assert!(serde_json::from_str::<SimulationOutcomeProfileInput>(unknown_field_json).is_err());
}
