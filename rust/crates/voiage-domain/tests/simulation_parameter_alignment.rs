//! O04 distinguishing tests for draw-key alignment of optional parameter samples.

use serde::Deserialize;
use voiage_domain::{
    SimulationOutcomeError, SimulationOutcomeProfileInput, ValidatedSimulationOutcomeProfile,
};

#[derive(Deserialize)]
struct ParameterFixtures {
    base_profile: SimulationOutcomeProfileInput,
    expected_parameter_names: Vec<String>,
    expected_parameter_matrix: Vec<Vec<f64>>,
}

fn load_fixtures() -> ParameterFixtures {
    let raw =
        include_str!("../../../../specs/ecosystem/simulation-outcome/v1/parameter-fixtures.json");
    serde_json::from_str(raw).expect("valid parameter-fixtures.json")
}

#[test]
fn reordered_parameter_samples_align_by_draw_key() {
    let fixtures = load_fixtures();
    let baseline = ValidatedSimulationOutcomeProfile::validate(&fixtures.base_profile)
        .expect("baseline parameter profile validates");

    let aligned = baseline
        .parameter_samples()
        .expect("aligned parameter samples present");
    let names: Vec<&str> = aligned
        .parameter_names()
        .iter()
        .map(voiage_domain::Identifier::as_str)
        .collect();
    assert_eq!(names, fixtures.expected_parameter_names);
    assert_eq!(aligned.matrix(), fixtures.expected_parameter_matrix);
    assert_eq!(aligned.draw_keys(), baseline.draw_keys());

    let mut reordered = fixtures.base_profile.clone();
    reordered
        .parameter_samples
        .as_mut()
        .expect("parameter samples")
        .reverse();
    let reordered_validated = ValidatedSimulationOutcomeProfile::validate(&reordered)
        .expect("reordered parameter samples align identically");
    assert_eq!(reordered_validated, baseline);

    // -0.0 parameter values normalize to +0.0 without changing equality.
    let mut neg_zero = fixtures.base_profile;
    neg_zero
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[0]
        .values
        .insert("theta_1".to_owned(), -0.0);
    let validated_neg_zero = ValidatedSimulationOutcomeProfile::validate(&neg_zero).unwrap();
    assert!(validated_neg_zero.parameter_samples().unwrap().matrix()[0][0].is_sign_positive());
}

#[test]
fn unmatched_and_duplicate_parameter_draws_fail_explicitly() {
    let fixtures = load_fixtures();

    let mut empty_list = fixtures.base_profile.clone();
    empty_list.parameter_samples = Some(Vec::new());
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&empty_list),
        Err(SimulationOutcomeError::MissingParameterDraw(
            "draw-a".to_owned()
        ))
    );

    let mut equal_length_unmatched = fixtures.base_profile.clone();
    equal_length_unmatched
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[1]
        .draw_key = "draw-z".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&equal_length_unmatched),
        Err(SimulationOutcomeError::UnmatchedParameterDraw(
            "draw-z".to_owned()
        ))
    );

    let mut missing_draw = fixtures.base_profile.clone();
    missing_draw
        .parameter_samples
        .as_mut()
        .expect("parameter samples")
        .pop();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&missing_draw),
        Err(SimulationOutcomeError::MissingParameterDraw(
            "draw-b".to_owned()
        ))
    );

    let mut duplicate_draw = fixtures.base_profile.clone();
    let dup = duplicate_draw
        .parameter_samples
        .as_ref()
        .expect("parameter samples")[0]
        .clone();
    duplicate_draw
        .parameter_samples
        .as_mut()
        .expect("parameter samples")
        .push(dup);
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&duplicate_draw),
        Err(SimulationOutcomeError::DuplicateParameterDraw(
            "draw-a".to_owned()
        ))
    );

    let mut mismatched_ns = fixtures.base_profile;
    mismatched_ns
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[0]
        .draw_namespace = "ns.wrong.v1".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&mismatched_ns),
        Err(SimulationOutcomeError::MismatchedDrawNamespace {
            expected: "ns.params.v1".to_owned(),
            actual: "ns.wrong.v1".to_owned(),
        })
    );
}

#[test]
fn empty_untrimmed_and_inconsistent_parameter_schemas_fail_explicitly() {
    let fixtures = load_fixtures();

    let mut empty_values = fixtures.base_profile.clone();
    empty_values
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[0]
        .values
        .clear();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&empty_values),
        Err(SimulationOutcomeError::EmptyParameterValues(
            "draw-a".to_owned()
        ))
    );

    let mut blank_param_name = fixtures.base_profile.clone();
    blank_param_name
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[0]
        .values
        .insert("   ".to_owned(), 1.0);
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&blank_param_name),
        Err(SimulationOutcomeError::EmptyIdentifier("parameter_id"))
    );

    let mut untrimmed_param_name = fixtures.base_profile.clone();
    untrimmed_param_name
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[0]
        .values
        .insert(" theta_1".to_owned(), 1.0);
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&untrimmed_param_name),
        Err(SimulationOutcomeError::UntrimmedIdentifier("parameter_id"))
    );

    let mut untrimmed_draw_key = fixtures.base_profile.clone();
    untrimmed_draw_key
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[0]
        .draw_key = "draw-a ".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&untrimmed_draw_key),
        Err(SimulationOutcomeError::UntrimmedIdentifier("draw_key"))
    );

    let mut inconsistent_schema = fixtures.base_profile.clone();
    inconsistent_schema
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[1]
        .values
        .remove("theta_2");
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&inconsistent_schema),
        Err(SimulationOutcomeError::InconsistentParameterSchema(
            "draw-b".to_owned()
        ))
    );

    let mut extra_param_key = fixtures.base_profile.clone();
    extra_param_key
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[1]
        .values
        .insert("theta_3".to_owned(), 3.0);
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&extra_param_key),
        Err(SimulationOutcomeError::InconsistentParameterSchema(
            "draw-b".to_owned()
        ))
    );

    let mut non_finite = fixtures.base_profile;
    non_finite
        .parameter_samples
        .as_mut()
        .expect("parameter samples")[0]
        .values
        .insert("theta_1".to_owned(), f64::NAN);
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&non_finite),
        Err(SimulationOutcomeError::NonFiniteParameterValue {
            draw_key: "draw-a".to_owned(),
            parameter_id: "theta_1".to_owned(),
        })
    );
}
