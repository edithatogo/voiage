//! Contract classification and deterministic-kernel execution for the v1 catalog.
//!
//! Contract classification remains side-effect free; explicit execution helpers
//! run only the production deterministic kernels covered by Phase 5.

use core::fmt;
use std::collections::HashSet;

use serde_json::{json, Map, Value};
use voiage_diagnostics::{ErrorCategory, ErrorCode};
use voiage_domain::{
    Identifier, Probability, SampleCube, SampleMatrix, SampleVector, StrategyCollection,
};
use voiage_numerics::{ceaf, dominance, enbs, evpi, evppi, DominanceStatus};
use voiage_serialization::{
    CeafResultV1, DominanceResultV1, EnbsResultV1, EvpiResultV1, EvppiResultV1, EvsiResultV1,
};

use crate::LoadedCompatibilityCase;

/// Stable method names covered by the compatibility catalog.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CompatibilityMethod {
    /// Expected value of perfect information.
    Evpi,
    /// Expected value of partial perfect information.
    Evppi,
    /// Expected value of sample information.
    Evsi,
    /// Expected net benefit of sampling.
    Enbs,
    /// Cost-effectiveness acceptability frontier.
    Ceaf,
    /// Cost-effectiveness dominance classification.
    Dominance,
}

impl CompatibilityMethod {
    fn parse(value: &str) -> Result<Self, ContractParityError> {
        match value {
            "evpi" => Ok(Self::Evpi),
            "evppi" => Ok(Self::Evppi),
            "evsi" => Ok(Self::Evsi),
            "enbs" => Ok(Self::Enbs),
            "ceaf" => Ok(Self::Ceaf),
            "dominance" => Ok(Self::Dominance),
            _ => Err(ContractParityError::new(format!(
                "unknown method {value:?}"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Evpi => "evpi",
            Self::Evppi => "evppi",
            Self::Evsi => "evsi",
            Self::Enbs => "enbs",
            Self::Ceaf => "ceaf",
            Self::Dominance => "dominance",
        }
    }
}

/// Stable fixture classification.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CompatibilityClassification {
    /// Ordinary valid input.
    Normal,
    /// Valid boundary input.
    Edge,
    /// Input expected to fail contract validation.
    Invalid,
}

impl CompatibilityClassification {
    fn parse(value: &str) -> Result<Self, ContractParityError> {
        match value {
            "normal" => Ok(Self::Normal),
            "edge" => Ok(Self::Edge),
            "invalid" => Ok(Self::Invalid),
            _ => Err(ContractParityError::new(format!(
                "unknown classification {value:?}"
            ))),
        }
    }
}

/// Exact fixture error identity mapped onto the stable Rust diagnostics family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractErrorMapping {
    /// Exact language-neutral fixture error code.
    pub fixture_code: String,
    /// Stable Rust diagnostics code used by bindings.
    pub stable_code: ErrorCode,
    /// Stable Rust diagnostics category implied by `stable_code`.
    pub stable_category: ErrorCategory,
}

/// Contract outcome represented by an expected fixture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractOutcome {
    /// A successful result shape.
    Result,
    /// An expected stable error shape.
    Error(ContractErrorMapping),
}

impl ContractOutcome {
    /// Returns the error mapping when this is an error outcome.
    #[must_use]
    pub const fn error(&self) -> Option<&ContractErrorMapping> {
        match self {
            Self::Result => None,
            Self::Error(error) => Some(error),
        }
    }
}

/// Evidence that one fixture passed Rust contract-only classification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractParityReport {
    /// Stable compatibility case identifier.
    pub case_id: String,
    /// Stable method classification.
    pub method: CompatibilityMethod,
    /// Normal, edge, or invalid classification.
    pub classification: CompatibilityClassification,
    /// Expected result or error classification.
    pub outcome: ContractOutcome,
    /// Whether Rust domain contracts validated the input shape or expected rejection.
    pub input_contract_validated: bool,
    /// Whether Rust serialization/diagnostic contracts validated the expected shape.
    pub expected_contract_validated: bool,
    /// Whether the explicit parity runner executed a production kernel.
    pub numerical_kernel_executed: bool,
}

/// A mismatch between fixture metadata and Rust domain/serialization contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractParityError(String);

impl ContractParityError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ContractParityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ContractParityError {}

/// Classifies all loaded fixtures through Rust domain and serialization contracts.
///
/// This is executable contract parity only. It validates input/result/error
/// shapes and stable mappings, but intentionally executes no numerical kernel.
///
/// # Errors
///
/// Returns [`ContractParityError`] for an unknown classification, malformed
/// artifact, unexpected input validity, or incompatible stable error mapping.
pub fn classify_compatibility_contracts(
    cases: &[LoadedCompatibilityCase],
) -> Result<Vec<ContractParityReport>, ContractParityError> {
    cases.iter().map(classify_case).collect()
}

/// Executes the foundational EVPI and ENBS compatibility cases.
///
/// Successful fixtures execute the production Rust kernel and compare its
/// result using the fixture's declared tolerances. Invalid fixtures retain the
/// exact contract error mapping established by contract classification.
///
/// # Errors
///
/// Returns [`ContractParityError`] when classification, kernel execution, or
/// numerical comparison disagrees with a canonical fixture.
pub fn execute_foundational_compatibility_contracts(
    cases: &[LoadedCompatibilityCase],
) -> Result<Vec<ContractParityReport>, ContractParityError> {
    cases
        .iter()
        .filter(|case| matches!(case.case.method.as_str(), "evpi" | "enbs"))
        .map(|case| {
            let mut report = classify_case(case)?;
            if matches!(report.outcome, ContractOutcome::Result) {
                execute_foundational_case(case, report.method)?;
                report.numerical_kernel_executed = true;
            }
            Ok(report)
        })
        .collect()
}

/// Executes all deterministic compatibility fixtures owned by the production
/// numerical kernels, including regression-based EVPPI, and preserves the
/// canonical invalid mappings.
///
/// # Errors
///
/// Returns [`ContractParityError`] when fixture loading, kernel execution, or
/// canonical result comparison fails.
pub fn execute_deterministic_compatibility_contracts(
    cases: &[LoadedCompatibilityCase],
) -> Result<Vec<ContractParityReport>, ContractParityError> {
    cases
        .iter()
        .filter(|case| {
            matches!(
                case.case.method.as_str(),
                "evpi" | "evppi" | "enbs" | "ceaf" | "dominance"
            )
        })
        .map(|case| {
            let mut report = classify_case(case)?;
            if matches!(report.outcome, ContractOutcome::Result) {
                execute_deterministic_case(case, report.method)?;
                report.numerical_kernel_executed = true;
            }
            Ok(report)
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn execute_deterministic_case(
    case: &LoadedCompatibilityCase,
    method: CompatibilityMethod,
) -> Result<(), ContractParityError> {
    match method {
        CompatibilityMethod::Evpi | CompatibilityMethod::Enbs => {
            execute_foundational_case(case, method)
        }
        CompatibilityMethod::Evppi => execute_evppi_case(case),
        CompatibilityMethod::Dominance => {
            let input = object(&case.input, "input").map_err(|error| case_error(case, error))?;
            let costs = parse::<SampleVector>(
                field(input, "costs").map_err(|error| case_error(case, error))?,
                "costs",
            )
            .map_err(|error| case_error(case, error))?;
            let effects = parse::<SampleVector>(
                field(input, "effects").map_err(|error| case_error(case, error))?,
                "effects",
            )
            .map_err(|error| case_error(case, error))?;
            let actual = dominance(&costs, &effects).map_err(|error| case_error(case, error))?;
            let expected_root =
                object(&case.expected, "expected").map_err(|error| case_error(case, error))?;
            let expected = field(expected_root, "result")
                .and_then(|value| object(value, "result"))
                .map_err(|error| case_error(case, error))?;
            assert_exact_array(case, expected, "frontier_indices", &actual.frontier_indices)?;
            assert_exact_array(
                case,
                expected,
                "strongly_dominated_indices",
                &actual.strongly_dominated_indices,
            )?;
            assert_exact_array(
                case,
                expected,
                "extended_dominated_indices",
                &actual.extended_dominated_indices,
            )?;
            let statuses = actual
                .status
                .iter()
                .map(|status| match status {
                    DominanceStatus::Frontier => "frontier",
                    DominanceStatus::StronglyDominated => "strongly_dominated",
                    DominanceStatus::ExtendedDominated => "extended_dominated",
                })
                .collect::<Vec<_>>();
            let expected_statuses = parse::<Vec<String>>(
                field(expected, "status").map_err(|error| case_error(case, error))?,
                "status",
            )
            .map_err(|error| case_error(case, error))?;
            if statuses
                != expected_statuses
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
            {
                return Err(case_error(case, "status output disagrees with fixture"));
            }
            assert_float_array(
                case,
                expected,
                expected_root,
                "incremental_costs",
                &actual.incremental_costs,
            )?;
            assert_float_array(
                case,
                expected,
                expected_root,
                "incremental_effects",
                &actual.incremental_effects,
            )?;
            assert_float_array(case, expected, expected_root, "icers", &actual.icers)
        }
        CompatibilityMethod::Ceaf => {
            let input = object(&case.input, "input").map_err(|error| case_error(case, error))?;
            let values = parse::<SampleCube>(
                field(input, "net_benefit").map_err(|error| case_error(case, error))?,
                "net benefit",
            )
            .map_err(|error| case_error(case, error))?;
            let thresholds = parse::<SampleVector>(
                field(input, "wtp_thresholds").map_err(|error| case_error(case, error))?,
                "WTP thresholds",
            )
            .map_err(|error| case_error(case, error))?;
            let confidence = parse::<f64>(
                field(input, "confidence_level").map_err(|error| case_error(case, error))?,
                "confidence level",
            )
            .map_err(|error| case_error(case, error))?;
            let actual =
                ceaf(&values, &thresholds, confidence).map_err(|error| case_error(case, error))?;
            let strategy_names = parse::<Vec<String>>(
                field(input, "strategy_names").map_err(|error| case_error(case, error))?,
                "strategy names",
            )
            .map_err(|error| case_error(case, error))?;
            let expected_root =
                object(&case.expected, "expected").map_err(|error| case_error(case, error))?;
            let expected = field(expected_root, "result")
                .and_then(|value| object(value, "result"))
                .map_err(|error| case_error(case, error))?;
            assert_float_array(
                case,
                expected,
                expected_root,
                "wtp_thresholds",
                &actual.wtp_thresholds,
            )?;
            assert_exact_array(
                case,
                expected,
                "optimal_strategy_indices",
                &actual.optimal_strategy_indices,
            )?;
            let expected_names = parse::<Vec<String>>(
                field(expected, "optimal_strategy_names")
                    .map_err(|error| case_error(case, error))?,
                "optimal strategy names",
            )
            .map_err(|error| case_error(case, error))?;
            let actual_names = actual
                .optimal_strategy_indices
                .iter()
                .map(|index| {
                    strategy_names
                        .get(*index)
                        .map(String::as_str)
                        .ok_or_else(|| {
                            case_error(
                                case,
                                format!("optimal strategy index {index} is out of range"),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if actual_names
                != expected_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
            {
                return Err(case_error(
                    case,
                    "optimal strategy names disagree with fixture",
                ));
            }
            assert_float_array(
                case,
                expected,
                expected_root,
                "acceptability_probabilities",
                &actual.acceptability_probabilities,
            )?;
            assert_float_array(
                case,
                expected,
                expected_root,
                "probability_lower",
                &actual.probability_lower,
            )?;
            assert_float_array(
                case,
                expected,
                expected_root,
                "probability_upper",
                &actual.probability_upper,
            )?;
            assert_float_array(
                case,
                expected,
                expected_root,
                "expected_net_benefit",
                &actual.expected_net_benefit,
            )
        }
        CompatibilityMethod::Evsi => Err(case_error(case, "not a deterministic kernel fixture")),
    }
}

fn assert_exact_array<T: serde::de::DeserializeOwned + PartialEq + fmt::Debug>(
    case: &LoadedCompatibilityCase,
    expected: &Map<String, Value>,
    field_name: &str,
    actual: &[T],
) -> Result<(), ContractParityError> {
    let expected_values = parse::<Vec<T>>(
        field(expected, field_name).map_err(|error| case_error(case, error))?,
        field_name,
    )
    .map_err(|error| case_error(case, error))?;
    if actual != expected_values {
        return Err(case_error(
            case,
            format!("{field_name} output disagrees with fixture"),
        ));
    }
    Ok(())
}

fn assert_float_array(
    case: &LoadedCompatibilityCase,
    expected: &Map<String, Value>,
    tolerance_source: &Map<String, Value>,
    field_name: &str,
    actual: &[f64],
) -> Result<(), ContractParityError> {
    let expected_values = parse::<Vec<f64>>(
        field(expected, field_name).map_err(|error| case_error(case, error))?,
        field_name,
    )
    .map_err(|error| case_error(case, error))?;
    let absolute_tolerance = parse::<f64>(
        field(tolerance_source, "absolute_tolerance").map_err(|error| case_error(case, error))?,
        "absolute tolerance",
    )
    .map_err(|error| case_error(case, error))?;
    let relative_tolerance = parse::<f64>(
        field(tolerance_source, "relative_tolerance").map_err(|error| case_error(case, error))?,
        "relative tolerance",
    )
    .map_err(|error| case_error(case, error))?;
    if actual.len() != expected_values.len()
        || actual
            .iter()
            .zip(expected_values)
            .any(|(actual, expected)| {
                (actual - expected).abs()
                    > absolute_tolerance.max(relative_tolerance * expected.abs())
            })
    {
        return Err(case_error(
            case,
            format!("{field_name} output disagrees with fixture"),
        ));
    }
    Ok(())
}

fn execute_foundational_case(
    case: &LoadedCompatibilityCase,
    method: CompatibilityMethod,
) -> Result<(), ContractParityError> {
    let input = object(&case.input, "input").map_err(|error| case_error(case, error))?;
    let actual = match method {
        CompatibilityMethod::Evpi => {
            let samples = parse::<SampleMatrix>(
                field(input, "net_benefit").map_err(|error| case_error(case, error))?,
                "net benefit",
            )
            .map_err(|error| case_error(case, error))?;
            evpi(&samples).map_err(|error| case_error(case, error))?
        }
        CompatibilityMethod::Enbs => {
            let evsi_result = parse::<f64>(
                field(input, "evsi_result").map_err(|error| case_error(case, error))?,
                "EVSI result",
            )
            .map_err(|error| case_error(case, error))?;
            let research_cost = parse::<f64>(
                field(input, "research_cost").map_err(|error| case_error(case, error))?,
                "research cost",
            )
            .map_err(|error| case_error(case, error))?;
            enbs(evsi_result, research_cost).map_err(|error| case_error(case, error))?
        }
        _ => return Err(case_error(case, "not a foundational kernel fixture")),
    };
    compare_foundational_result(case, actual)
}

fn execute_evppi_case(case: &LoadedCompatibilityCase) -> Result<(), ContractParityError> {
    let input = object(&case.input, "input").map_err(|error| case_error(case, error))?;
    let net_benefit = parse::<SampleMatrix>(
        field(input, "net_benefit").map_err(|error| case_error(case, error))?,
        "net benefit",
    )
    .map_err(|error| case_error(case, error))?;
    let parameters = object(
        field(input, "parameters").map_err(|error| case_error(case, error))?,
        "parameters",
    )
    .map_err(|error| case_error(case, error))?;
    let parameter_names = parse::<Vec<String>>(
        field(input, "parameters_of_interest").map_err(|error| case_error(case, error))?,
        "parameters of interest",
    )
    .map_err(|error| case_error(case, error))?;
    let columns = parameter_names
        .iter()
        .map(|name| {
            parse::<SampleVector>(
                parameters
                    .get(name)
                    .ok_or_else(|| case_error(case, format!("missing parameter {name}")))?,
                "parameter samples",
            )
            .map_err(|error| case_error(case, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parameter_values = (0..net_benefit.shape()[0])
        .map(|sample_index| {
            columns
                .iter()
                .map(|column| column.as_slice()[sample_index])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let parameter_samples =
        SampleMatrix::try_from(parameter_values).map_err(|error| case_error(case, error))?;
    let actual =
        evppi(&net_benefit, &parameter_samples).map_err(|error| case_error(case, error))?;
    compare_foundational_result(case, actual)
}

fn compare_foundational_result(
    case: &LoadedCompatibilityCase,
    actual: f64,
) -> Result<(), ContractParityError> {
    let expected = object(&case.expected, "expected").map_err(|error| case_error(case, error))?;
    let result = parse::<f64>(
        field(expected, "result").map_err(|error| case_error(case, error))?,
        "result",
    )
    .map_err(|error| case_error(case, error))?;
    let absolute_tolerance = parse::<f64>(
        field(expected, "absolute_tolerance").map_err(|error| case_error(case, error))?,
        "absolute tolerance",
    )
    .map_err(|error| case_error(case, error))?;
    let relative_tolerance = parse::<f64>(
        field(expected, "relative_tolerance").map_err(|error| case_error(case, error))?,
        "relative tolerance",
    )
    .map_err(|error| case_error(case, error))?;
    let allowed = absolute_tolerance.max(relative_tolerance * result.abs());
    if (actual - result).abs() > allowed {
        return Err(case_error(
            case,
            format!("kernel result {actual} disagrees with expected {result}"),
        ));
    }
    Ok(())
}

fn classify_case(
    case: &LoadedCompatibilityCase,
) -> Result<ContractParityReport, ContractParityError> {
    let method = CompatibilityMethod::parse(&case.case.method)?;
    let classification = CompatibilityClassification::parse(&case.case.classification)?;
    let validation = validate_input(method, &case.input);
    let outcome = if classification == CompatibilityClassification::Invalid {
        let actual = match validation {
            Ok(()) => {
                return Err(case_error(
                    case,
                    "invalid fixture passed its Rust input contract",
                ));
            }
            Err(error) => error.mapping(),
        };
        let expected = validate_expected_error(&case.expected)?;
        if actual != expected {
            return Err(case_error(
                case,
                format!("input mapping {actual:?} disagrees with expected mapping {expected:?}"),
            ));
        }
        ContractOutcome::Error(expected)
    } else {
        validation.map_err(|error| case_error(case, error.message))?;
        validate_expected_result(method, &case.input, &case.expected)
            .map_err(|error| case_error(case, error))?;
        ContractOutcome::Result
    };
    Ok(ContractParityReport {
        case_id: case.case.case_id.clone(),
        method,
        classification,
        outcome,
        input_contract_validated: true,
        expected_contract_validated: true,
        numerical_kernel_executed: false,
    })
}

fn case_error(case: &LoadedCompatibilityCase, message: impl fmt::Display) -> ContractParityError {
    ContractParityError::new(format!("{}: {message}", case.case.case_id))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn field<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a Value, String> {
    object.get(name).ok_or_else(|| format!("missing {name}"))
}

fn parse<T: serde::de::DeserializeOwned>(value: &Value, label: &str) -> Result<T, String> {
    serde_json::from_value(value.clone()).map_err(|error| format!("invalid {label}: {error}"))
}

#[derive(Debug)]
struct InputContractFailure {
    fixture_code: &'static str,
    stable_code: ErrorCode,
    message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputFailureKind {
    InvalidInput,
    UnknownParameter,
    UnsupportedEstimator,
    NegativeResearchCost,
}

#[derive(Debug)]
struct InputValidationError {
    kind: InputFailureKind,
    message: String,
}

impl InputValidationError {
    fn new(kind: InputFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(InputFailureKind::InvalidInput, message)
    }
}

impl InputContractFailure {
    fn new(fixture_code: &'static str, stable_code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            fixture_code,
            stable_code,
            message: message.into(),
        }
    }

    fn mapping(&self) -> ContractErrorMapping {
        ContractErrorMapping {
            fixture_code: self.fixture_code.to_owned(),
            stable_code: self.stable_code,
            stable_category: self.stable_code.category(),
        }
    }
}

fn validate_input(method: CompatibilityMethod, value: &Value) -> Result<(), InputContractFailure> {
    if contains_non_finite_sentinel(value) {
        return Err(InputContractFailure::new(
            "non_finite_value",
            ErrorCode::InvalidInput,
            "input contains a non-finite sentinel",
        ));
    }
    let input = object(value, "input").map_err(|message| {
        InputContractFailure::new("invalid_input", ErrorCode::InvalidInput, message)
    })?;
    let validation = match method {
        CompatibilityMethod::Evpi => field(input, "net_benefit")
            .and_then(|value| parse::<SampleMatrix>(value, "net_benefit"))
            .map(|_| ())
            .map_err(InputValidationError::invalid),
        CompatibilityMethod::Evppi => validate_evppi_input(input),
        CompatibilityMethod::Evsi => validate_evsi_input(input).map_err(|message| {
            let kind = if input.get("estimator").and_then(Value::as_str) == Some("moment_based") {
                InputFailureKind::InvalidInput
            } else {
                InputFailureKind::UnsupportedEstimator
            };
            InputValidationError::new(kind, message)
        }),
        CompatibilityMethod::Enbs => validate_enbs_input(input),
        CompatibilityMethod::Ceaf => {
            validate_ceaf_input(input).map_err(InputValidationError::invalid)
        }
        CompatibilityMethod::Dominance => {
            validate_dominance_input(input).map_err(InputValidationError::invalid)
        }
    };
    validation.map_err(|error| map_input_failure(method, error))
}

fn contains_non_finite_sentinel(value: &Value) -> bool {
    match value {
        Value::String(value) => matches!(value.as_str(), "NaN" | "Infinity" | "-Infinity"),
        Value::Array(values) => values.iter().any(contains_non_finite_sentinel),
        Value::Object(values) => values.values().any(contains_non_finite_sentinel),
        _ => false,
    }
}

fn map_input_failure(
    method: CompatibilityMethod,
    error: InputValidationError,
) -> InputContractFailure {
    let (fixture_code, stable_code) = match (method, error.kind) {
        (CompatibilityMethod::Evpi, _) => ("shape_mismatch", ErrorCode::DimensionMismatch),
        (CompatibilityMethod::Evppi, InputFailureKind::UnknownParameter) => {
            ("unknown_parameter", ErrorCode::InvalidInput)
        }
        (CompatibilityMethod::Evsi, InputFailureKind::UnsupportedEstimator) => {
            ("unsupported_estimator", ErrorCode::BackendUnavailable)
        }
        (CompatibilityMethod::Enbs, InputFailureKind::NegativeResearchCost) => {
            ("negative_research_cost", ErrorCode::InvalidInput)
        }
        (CompatibilityMethod::Ceaf, _) => {
            ("threshold_count_mismatch", ErrorCode::DimensionMismatch)
        }
        (CompatibilityMethod::Dominance, _) => {
            ("strategy_count_mismatch", ErrorCode::DimensionMismatch)
        }
        _ => ("invalid_input", ErrorCode::InvalidInput),
    };
    InputContractFailure::new(fixture_code, stable_code, error.message)
}

fn validate_enbs_input(input: &Map<String, Value>) -> Result<(), InputValidationError> {
    let evsi = parse::<f64>(
        field(input, "evsi_result").map_err(InputValidationError::invalid)?,
        "evsi_result",
    )
    .map_err(InputValidationError::invalid)?;
    let cost = parse::<f64>(
        field(input, "research_cost").map_err(InputValidationError::invalid)?,
        "research_cost",
    )
    .map_err(InputValidationError::invalid)?;
    SampleVector::try_from(vec![evsi, cost])
        .map_err(|error| InputValidationError::invalid(error.to_string()))?;
    if cost < 0.0 {
        return Err(InputValidationError::new(
            InputFailureKind::NegativeResearchCost,
            "research_cost must be nonnegative",
        ));
    }
    Ok(())
}

fn validate_evppi_input(input: &Map<String, Value>) -> Result<(), InputValidationError> {
    let net_benefit = parse::<SampleMatrix>(
        field(input, "net_benefit").map_err(InputValidationError::invalid)?,
        "net_benefit",
    )
    .map_err(InputValidationError::invalid)?;
    let parameters = object(
        field(input, "parameters").map_err(InputValidationError::invalid)?,
        "parameters",
    )
    .map_err(InputValidationError::invalid)?;
    let mut names = HashSet::new();
    for (name, values) in parameters {
        Identifier::new(name).map_err(|error| InputValidationError::invalid(error.to_string()))?;
        let values = parse::<SampleVector>(values, "parameter samples")
            .map_err(InputValidationError::invalid)?;
        if values.len() != net_benefit.shape()[0] {
            return Err(InputValidationError::invalid(
                "parameter sample count mismatch",
            ));
        }
        names.insert(name.as_str());
    }
    let requested = parse::<Vec<String>>(
        field(input, "parameters_of_interest").map_err(InputValidationError::invalid)?,
        "parameters_of_interest",
    )
    .map_err(InputValidationError::invalid)?;
    if requested.is_empty() || requested.iter().any(|name| !names.contains(name.as_str())) {
        return Err(InputValidationError::new(
            InputFailureKind::UnknownParameter,
            "requested parameter is absent from the parameter sample set",
        ));
    }
    if input
        .get("estimator")
        .and_then(Value::as_str)
        .unwrap_or("linear")
        != "linear"
    {
        return Err(InputValidationError::new(
            InputFailureKind::UnsupportedEstimator,
            "EVPPI estimator is outside the stable contract",
        ));
    }
    Ok(())
}

fn validate_evsi_input(input: &Map<String, Value>) -> Result<(), String> {
    let model = object(field(input, "model")?, "model")?;
    if field(model, "type")?.as_str() != Some("constant-net-benefit") {
        return Err("unsupported model type".into());
    }
    let names = parse::<StrategyCollection>(field(model, "strategy_names")?, "strategy_names")?;
    let values = parse::<SampleVector>(field(model, "values")?, "model values")?;
    if names.len() != values.len() {
        return Err("model strategy count mismatch".into());
    }
    let prior = object(field(input, "prior")?, "prior")?;
    if prior.is_empty() {
        return Err("prior must not be empty".into());
    }
    for (name, values) in prior {
        Identifier::new(name).map_err(|error| error.to_string())?;
        parse::<SampleVector>(values, "prior samples")?;
    }
    let design = object(field(input, "trial_design")?, "trial_design")?;
    let arms = field(design, "arms")?
        .as_array()
        .ok_or_else(|| "trial arms must be an array".to_owned())?;
    if arms.is_empty() {
        return Err("trial arms must not be empty".into());
    }
    for arm in arms {
        let arm = object(arm, "trial arm")?;
        Identifier::new(field(arm, "name")?.as_str().unwrap_or_default())
            .map_err(|error| error.to_string())?;
        let size = field(arm, "sample_size")?
            .as_u64()
            .ok_or_else(|| "sample_size must be an unsigned integer".to_owned())?;
        if size == 0 {
            return Err("sample_size must be positive".into());
        }
    }
    if field(input, "estimator")?.as_str() != Some("moment_based") {
        return Err("unsupported estimator".into());
    }
    Ok(())
}

fn validate_ceaf_input(input: &Map<String, Value>) -> Result<(), String> {
    let cube = parse::<SampleCube>(field(input, "net_benefit")?, "net_benefit")?;
    let names = parse::<StrategyCollection>(field(input, "strategy_names")?, "strategy_names")?;
    let thresholds = parse::<SampleVector>(field(input, "wtp_thresholds")?, "wtp_thresholds")?;
    Probability::new(parse::<f64>(
        field(input, "confidence_level")?,
        "confidence_level",
    )?)
    .map_err(|error| error.to_string())?;
    let shape = cube.shape();
    if shape[1] != names.len() || shape[2] != thresholds.len() {
        return Err("threshold or strategy count mismatch".into());
    }
    Ok(())
}

fn validate_dominance_input(input: &Map<String, Value>) -> Result<(), String> {
    let costs = parse::<SampleVector>(field(input, "costs")?, "costs")?;
    let effects = parse::<SampleVector>(field(input, "effects")?, "effects")?;
    let names = parse::<StrategyCollection>(field(input, "strategy_names")?, "strategy_names")?;
    if costs.len() != effects.len() || costs.len() != names.len() {
        return Err("strategy count mismatch".into());
    }
    Ok(())
}

fn validate_expected_error(value: &Value) -> Result<ContractErrorMapping, ContractParityError> {
    let root = object(value, "expected error").map_err(ContractParityError::new)?;
    if root.len() != 1 {
        return Err(ContractParityError::new(
            "error outcome must contain only error",
        ));
    }
    let error = object(
        field(root, "error").map_err(ContractParityError::new)?,
        "error",
    )
    .map_err(ContractParityError::new)?;
    if error.len() != 2 {
        return Err(ContractParityError::new(
            "error must contain category and code",
        ));
    }
    let category = field(error, "category")
        .map_err(ContractParityError::new)?
        .as_str()
        .ok_or_else(|| ContractParityError::new("error category must be text"))?;
    let fixture_code = field(error, "code")
        .map_err(ContractParityError::new)?
        .as_str()
        .ok_or_else(|| ContractParityError::new("error code must be text"))?;
    let stable_code = match fixture_code {
        "shape_mismatch" | "threshold_count_mismatch" | "strategy_count_mismatch" => {
            ErrorCode::DimensionMismatch
        }
        "unsupported_estimator" => ErrorCode::BackendUnavailable,
        "unknown_parameter" | "negative_research_cost" | "non_finite_value" => {
            ErrorCode::InvalidInput
        }
        _ => {
            return Err(ContractParityError::new(format!(
                "unknown fixture error code {fixture_code:?}"
            )))
        }
    };
    let expected_category = match category {
        "input" => ErrorCategory::Input,
        "dimension" => ErrorCategory::DimensionMismatch,
        "capability" => ErrorCategory::BackendUnavailable,
        _ => {
            return Err(ContractParityError::new(format!(
                "unknown fixture error category {category:?}"
            )))
        }
    };
    if stable_code.category() != expected_category {
        return Err(ContractParityError::new(format!(
            "error category {category:?} disagrees with code {fixture_code:?}"
        )));
    }
    Ok(ContractErrorMapping {
        fixture_code: fixture_code.to_owned(),
        stable_code,
        stable_category: expected_category,
    })
}

fn validate_expected_result(
    method: CompatibilityMethod,
    input: &Value,
    value: &Value,
) -> Result<(), String> {
    let root = object(value, "expected result")?;
    if root.keys().collect::<HashSet<_>>()
        != HashSet::from([
            &"result".to_owned(),
            &"absolute_tolerance".to_owned(),
            &"relative_tolerance".to_owned(),
        ])
    {
        return Err("result outcome must contain result and both tolerances".into());
    }
    for name in ["absolute_tolerance", "relative_tolerance"] {
        let tolerance = parse::<f64>(field(root, name)?, name)?;
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(format!("{name} must be finite and nonnegative"));
        }
    }
    let result = field(root, "result")?;
    let wire = canonical_result_wire(method, input, result)?;
    match method {
        CompatibilityMethod::Evpi => drop(parse::<EvpiResultV1>(&wire, "EVPI result")?),
        CompatibilityMethod::Evppi => drop(parse::<EvppiResultV1>(&wire, "EVPPI result")?),
        CompatibilityMethod::Evsi => drop(parse::<EvsiResultV1>(&wire, "EVSI result")?),
        CompatibilityMethod::Enbs => drop(parse::<EnbsResultV1>(&wire, "ENBS result")?),
        CompatibilityMethod::Ceaf => drop(parse::<CeafResultV1>(&wire, "CEAF result")?),
        CompatibilityMethod::Dominance => {
            drop(parse::<DominanceResultV1>(&wire, "dominance result")?);
        }
    }
    Ok(())
}

fn canonical_result_wire(
    method: CompatibilityMethod,
    input: &Value,
    result: &Value,
) -> Result<Value, String> {
    let base = ("analysis-fixture", "decision-problem-fixture");
    match method {
        CompatibilityMethod::Evpi => Ok(json!({
            "analysis_id": base.0, "decision_problem_id": base.1, "analysis_type": "evpi",
            "willingness_to_pay": 1.0, "expected_current_value": 0.0,
            "expected_perfect_information": result, "evpi": result
        })),
        CompatibilityMethod::Evppi => Ok(json!({
            "analysis_id": base.0, "decision_problem_id": base.1, "analysis_type": "evppi",
            "parameter_names": input["parameters_of_interest"], "evppi": result
        })),
        CompatibilityMethod::Evsi => Ok(json!({
            "analysis_id": base.0, "decision_problem_id": base.1, "analysis_type": "evsi",
            "trial_design_id": "fixture-design", "sample_size": input["trial_design"]["arms"][0]["sample_size"],
            "evsi": result
        })),
        CompatibilityMethod::Enbs => Ok(json!({
            "analysis_id": base.0, "decision_problem_id": base.1, "analysis_type": "enbs",
            "trial_design_id": "fixture-design", "sample_size": 1,
            "design_cost": input["research_cost"], "enbs": result
        })),
        CompatibilityMethod::Ceaf => {
            let fields = object(result, "CEAF result")?;
            let mut wire = fields.clone();
            wire.insert("analysis_id".into(), json!(base.0));
            wire.insert("decision_problem_id".into(), json!(base.1));
            wire.insert("analysis_type".into(), json!(method.as_str()));
            Ok(Value::Object(wire))
        }
        CompatibilityMethod::Dominance => {
            let fields = object(result, "dominance result")?;
            let mut wire = fields.clone();
            wire.insert("analysis_id".into(), json!(base.0));
            wire.insert("decision_problem_id".into(), json!(base.1));
            wire.insert("analysis_type".into(), json!(method.as_str()));
            wire.insert("strategy_names".into(), input["strategy_names"].clone());
            wire.insert("costs".into(), input["costs"].clone());
            wire.insert("effects".into(), input["effects"].clone());
            Ok(Value::Object(wire))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_error_mapping_uses_typed_failure_kind_not_display_message() {
        let scenarios = [
            (
                CompatibilityMethod::Evppi,
                InputFailureKind::UnknownParameter,
                "wording deliberately unrelated to parameter lookup",
                "unknown_parameter",
                ErrorCode::InvalidInput,
            ),
            (
                CompatibilityMethod::Evsi,
                InputFailureKind::UnsupportedEstimator,
                "wording deliberately unrelated to estimator support",
                "unsupported_estimator",
                ErrorCode::BackendUnavailable,
            ),
            (
                CompatibilityMethod::Enbs,
                InputFailureKind::NegativeResearchCost,
                "wording deliberately unrelated to cost bounds",
                "negative_research_cost",
                ErrorCode::InvalidInput,
            ),
        ];

        for (method, kind, message, fixture_code, stable_code) in scenarios {
            let failure = map_input_failure(method, InputValidationError::new(kind, message));
            assert_eq!(failure.fixture_code, fixture_code);
            assert_eq!(failure.stable_code, stable_code);
            assert_eq!(failure.message, message);
        }
    }
}
