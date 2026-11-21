use std::collections::BTreeMap;
use std::fmt::Display;
use std::num::NonZeroUsize;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use serde_json::{json, Value};
use voiage_diagnostics::ErrorCategory;
use voiage_domain::{
    AffineNetBenefit, BigRational, CredalContractError, CredalInterval, CredalPoint,
    CredalPolytope, EAdmissibilityResult, Identifier, LinearInequality, SampleVector, StrategyId,
};
use voiage_numerics::{solve_credal, CredalExtremum, CredalSolveResult, NumericalInputError};

use super::{result_to_dict, BoundaryError};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Problem {
    parameters: Vec<Parameter>,
    #[serde(default)]
    inequalities: Vec<Constraint>,
    strategies: Vec<Objective>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameter {
    id: String,
    lower: f64,
    upper: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Constraint {
    coefficients: Vec<f64>,
    upper: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Objective {
    strategy_id: String,
    intercept: f64,
    coefficients: Vec<f64>,
}

/// Call the native credal solver and return exact decimal-string fractions.
#[pyfunction]
pub(super) fn compute_credal_decisions<'py>(
    py: Python<'py>,
    problem_json: &str,
    max_bases: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let budget = NonZeroUsize::new(max_bases)
        .ok_or_else(|| invalid("A positive basis budget is required"))?;
    let problem: Problem = serde_json::from_str(problem_json).map_err(invalid)?;
    let mut parameter_ids = Vec::with_capacity(problem.parameters.len());
    let mut bounds = Vec::with_capacity(parameter_ids.capacity());
    for parameter in problem.parameters {
        let name = canonical_id(&parameter.id)?;
        parameter_ids.push(Identifier::new(name).map_err(invalid)?);
        bounds.push(CredalInterval::new(parameter.lower, parameter.upper).map_err(contract_error)?);
    }
    let rows = problem
        .inequalities
        .into_iter()
        .map(|constraint| {
            let coefficients = SampleVector::try_from(constraint.coefficients).map_err(invalid)?;
            LinearInequality::new(coefficients, constraint.upper).map_err(contract_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let strategies = problem
        .strategies
        .into_iter()
        .map(|strategy| {
            let name = canonical_id(&strategy.strategy_id)?;
            let id = StrategyId::new(name).map_err(invalid)?;
            let coefficients = SampleVector::try_from(strategy.coefficients).map_err(invalid)?;
            AffineNetBenefit::new(id, strategy.intercept, coefficients).map_err(contract_error)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let support = CredalPolytope::new(parameter_ids, bounds, rows).map_err(contract_error)?;
    let solved = py
        .detach(|| solve_credal(&support, &strategies, budget))
        .map_err(kernel_error)?;
    let (parameter_ids, _) = support.coordinates();
    let payload = solved_payload(&solved, parameter_ids);
    result_to_dict(py, &payload).map(|(result, _canonical)| result)
}

fn canonical_id(name: &str) -> PyResult<&str> {
    if name.trim() != name {
        return Err(invalid(
            "Identifiers must preserve their supplied text without surrounding whitespace",
        ));
    }
    Ok(name)
}

fn invalid(error: impl Display) -> PyErr {
    BoundaryError::Input(error.to_string()).into_pyerr()
}

fn contract_error(error: CredalContractError) -> PyErr {
    if matches!(error, CredalContractError::DimensionMismatch { .. }) {
        BoundaryError::DimensionMismatch(error.to_string()).into_pyerr()
    } else {
        invalid(error)
    }
}

fn kernel_error(error: NumericalInputError) -> PyErr {
    if error.category() == ErrorCategory::DimensionMismatch {
        BoundaryError::DimensionMismatch(error.to_string()).into_pyerr()
    } else {
        invalid(error)
    }
}

fn fraction(value: &BigRational) -> [String; 2] {
    [value.numer().to_string(), value.denom().to_string()]
}

fn point_payload(point: &CredalPoint) -> Vec<[String; 2]> {
    point.coordinates().iter().map(fraction).collect()
}

fn extremum_payload(extremum: &CredalExtremum) -> Value {
    json!({"value": fraction(&extremum.value), "point": point_payload(&extremum.point)})
}

fn admissibility_payload(witness: &EAdmissibilityResult) -> (&str, Value) {
    match witness {
        EAdmissibilityResult::Admissible { strategy, point } => (
            strategy.as_str(),
            json!({"status": "admissible", "point": point_payload(point)}),
        ),
        EAdmissibilityResult::Dominated(witness) => {
            let (alternatives, support, margin) = witness.certificate();
            let alternatives: BTreeMap<_, _> = alternatives
                .iter()
                .map(|(id, value)| (id.as_str(), fraction(value)))
                .collect();
            (
                witness.strategy().as_str(),
                json!({
                    "status": "dominated",
                    "alternative_multipliers": alternatives,
                    "support_multipliers": support.iter().map(fraction).collect::<Vec<_>>(),
                    "margin": fraction(margin),
                }),
            )
        }
        EAdmissibilityResult::Indeterminate { strategy } => {
            (strategy.as_str(), json!({"status": "indeterminate"}))
        }
    }
}

fn solved_payload(result: &CredalSolveResult, parameter_ids: &[Identifier]) -> Value {
    let parameter_ids: Vec<_> = parameter_ids.iter().map(Identifier::as_str).collect();
    match result {
        CredalSolveResult::Feasible(analysis) => {
            let ranges: BTreeMap<_, _> = analysis.ranges.iter().map(|(id, range)| {
                (id.as_str(), json!({"minimum": extremum_payload(&range.minimum), "maximum": extremum_payload(&range.maximum)}))
            }).collect();
            let e_admissibility: BTreeMap<_, _> = analysis
                .e_admissibility
                .iter()
                .map(admissibility_payload)
                .collect();
            json!({
                "contract_version": 1,
                "status": "feasible",
                "parameter_ids": parameter_ids,
                "bases_examined": analysis.bases_examined,
                "ranges": ranges,
                "gamma_maximin": analysis.gamma_maximin,
                "gamma_maximax": analysis.gamma_maximax,
                "interval_undominated": analysis.interval_undominated,
                "e_admissibility": e_admissibility,
            })
        }
        CredalSolveResult::Infeasible { bases_examined } => json!({
            "contract_version": 1,
            "status": "infeasible",
            "parameter_ids": parameter_ids,
            "bases_examined": bases_examined,
        }),
        CredalSolveResult::Indeterminate { bases_examined } => json!({
            "contract_version": 1,
            "status": "indeterminate",
            "parameter_ids": parameter_ids,
            "bases_examined": bases_examined,
        }),
    }
}
