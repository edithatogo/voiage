//! O03 distinguishing tests for explicit value, unit, perspective, and objective declarations.

use serde::Deserialize;
use voiage_domain::{
    SimulationOutcomeError, SimulationOutcomeProfileInput, ValidatedSimulationOutcomeProfile,
    ValueDeclaration, ValueDeclarationInput,
};

#[derive(Deserialize)]
struct ConventionPair {
    convention: String,
    objective: String,
}

#[derive(Deserialize)]
struct ValueConventionFixtures {
    accepted: Vec<serde_json::Value>,
    incompatible_convention_objective: Vec<ConventionPair>,
    unsupported_values: Vec<ConventionPair>,
}

fn load_fixtures() -> ValueConventionFixtures {
    let raw = include_str!(
        "../../../../specs/ecosystem/simulation-outcome/v1/value-convention-fixtures.json"
    );
    serde_json::from_str(raw).expect("valid value-convention-fixtures.json")
}

fn parse_declaration(entry: &serde_json::Value) -> (String, ValueDeclarationInput) {
    let mut obj = entry.as_object().expect("object").clone();
    let label = obj
        .remove("label")
        .and_then(|v| v.as_str().map(str::to_owned))
        .expect("string label");
    let decl: ValueDeclarationInput =
        serde_json::from_value(serde_json::Value::Object(obj)).expect("valid declaration");
    (label, decl)
}

fn base_profile() -> SimulationOutcomeProfileInput {
    let raw =
        include_str!("../../../../specs/ecosystem/simulation-outcome/v1/pairing-fixtures.json");
    let parsed: serde_json::Value = serde_json::from_str(raw).expect("valid json");
    serde_json::from_value(parsed["base_profile"].clone()).expect("valid base_profile")
}

#[test]
fn matching_value_conventions_and_metadata_validate() {
    let fixtures = load_fixtures();
    assert_eq!(fixtures.accepted.len(), 3);
    for entry in &fixtures.accepted {
        let (label, decl) = parse_declaration(entry);
        let mut profile = base_profile();
        profile.value_declaration = decl.clone();
        profile.outcomes[0].declaration = Some(decl);
        let validated = ValidatedSimulationOutcomeProfile::validate(&profile)
            .unwrap_or_else(|err| panic!("{label} failed validation: {err}"));
        assert_eq!(
            validated.value_declaration().unit(),
            profile.value_declaration.unit
        );
    }
}

#[test]
fn incompatible_conventions_and_record_mismatches_fail_without_implicit_conversion() {
    let fixtures = load_fixtures();
    let base = base_profile();

    for pair in fixtures.incompatible_convention_objective {
        let mut decl = base.value_declaration.clone();
        decl.convention.clone_from(&pair.convention);
        decl.objective.clone_from(&pair.objective);
        assert_eq!(
            ValueDeclaration::from_input(&decl),
            Err(SimulationOutcomeError::IncompatibleConventionObjective {
                convention: pair.convention,
                objective: pair.objective,
            })
        );
    }

    for pair in fixtures.unsupported_values {
        let mut decl = base.value_declaration.clone();
        decl.convention = pair.convention;
        decl.objective = pair.objective;
        assert!(matches!(
            ValueDeclaration::from_input(&decl),
            Err(SimulationOutcomeError::UnsupportedValueConvention(_)
                | SimulationOutcomeError::UnsupportedObjectiveDirection(_))
        ));
    }

    let mut missing_unit = base.clone();
    missing_unit.value_declaration.unit = "  ".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&missing_unit),
        Err(SimulationOutcomeError::EmptyIdentifier("unit"))
    );

    let mut untrimmed_unit = base.clone();
    untrimmed_unit.value_declaration.unit = " NZD".to_owned();
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&untrimmed_unit),
        Err(SimulationOutcomeError::UntrimmedIdentifier("unit"))
    );

    let mut blank_optional = base.clone();
    blank_optional.value_declaration.horizon = Some(" ".to_owned());
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&blank_optional),
        Err(SimulationOutcomeError::EmptyIdentifier("horizon"))
    );

    for (field, mutator) in [
        (
            "unit",
            (|d: &mut ValueDeclarationInput| d.unit = "USD".to_owned())
                as fn(&mut ValueDeclarationInput),
        ),
        ("perspective", |d| d.perspective = "payer".to_owned()),
        ("convention", |d| d.convention = "utility".to_owned()),
        ("population_basis", |d| {
            d.population_basis = Some("cohort_500".to_owned());
        }),
        ("horizon", |d| d.horizon = Some("5y".to_owned())),
        ("discounting_status", |d| {
            d.discounting_status = Some("undiscounted".to_owned());
        }),
    ] {
        let mut mismatched = base.clone();
        let mut rec_decl = mismatched.value_declaration.clone();
        mutator(&mut rec_decl);
        mismatched.outcomes[0].declaration = Some(rec_decl);
        assert_eq!(
            ValidatedSimulationOutcomeProfile::validate(&mismatched),
            Err(SimulationOutcomeError::IncompatibleValueDeclaration(field))
        );
    }

    let mut loss_vs_utility = base.clone();
    loss_vs_utility.outcomes[0].declaration = Some(ValueDeclarationInput {
        unit: base.value_declaration.unit.clone(),
        perspective: base.value_declaration.perspective.clone(),
        convention: "loss".to_owned(),
        objective: "minimize".to_owned(),
        currency_price_year: None,
        population_basis: None,
        horizon: None,
        discounting_status: None,
    });
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&loss_vs_utility),
        Err(SimulationOutcomeError::IncompatibleValueDeclaration(
            "convention"
        ))
    );

    let mut currency_mismatch = base;
    currency_mismatch.value_declaration.currency_price_year = Some("NZD-2026".to_owned());
    let mut rec_decl = currency_mismatch.value_declaration.clone();
    rec_decl.currency_price_year = Some("NZD-2024".to_owned());
    currency_mismatch.outcomes[0].declaration = Some(rec_decl);
    assert_eq!(
        ValidatedSimulationOutcomeProfile::validate(&currency_mismatch),
        Err(SimulationOutcomeError::IncompatibleValueDeclaration(
            "currency_price_year"
        ))
    );
}
