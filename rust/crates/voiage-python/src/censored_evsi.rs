use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use serde_json::{json, Value};
use voiage_domain::{Identifier, SampleMatrix};
use voiage_numerics::{
    sequential_evsi, ObservationChannel, SequentialEvsi, TobitModel, TobitState,
};

use super::{canonical_id, fraction, invalid, kernel_error, result_to_dict, BoundaryError};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Experiment {
    strategy_ids: Vec<String>,
    states: Vec<PayoffState>,
    stages: Vec<Vec<Acquisition>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PayoffState {
    state_id: String,
    prior_weight: f64,
    net_benefits: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Acquisition {
    action_id: String,
    limit: f64,
    cutpoints: Vec<f64>,
    outcome_ids: Vec<String>,
    states: Vec<ObservationState>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationState {
    state_id: String,
    mean: f64,
    standard_deviation: f64,
    detection_probability: f64,
}

impl Acquisition {
    fn into_channel(self, state_ids: &[Identifier]) -> PyResult<ObservationChannel> {
        let action = identifier(&self.action_id)?;
        let outcomes = identifiers("outcome IDs", self.outcome_ids.iter().map(String::as_str))?;
        let declared_count = self.states.len();
        let parameters = self
            .states
            .into_iter()
            .map(|state| {
                let id = identifier(&state.state_id)?;
                let signal = TobitState::new(
                    state.mean,
                    state.standard_deviation,
                    state.detection_probability,
                )
                .map_err(kernel_error)?;
                Ok((id, signal))
            })
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        if parameters.len() != declared_count {
            return Err(invalid(
                "Observation state IDs must be unique within an acquisition",
            ));
        }
        if parameters.len() != state_ids.len() {
            return Err(state_mismatch());
        }
        let ordered = state_ids
            .iter()
            .map(|id| parameters.get(id).copied().ok_or_else(state_mismatch))
            .collect::<PyResult<Vec<_>>>()?;
        TobitModel::new(self.limit, ordered)
            .and_then(|model| model.finite_channel(action, &self.cutpoints, outcomes))
            .map_err(kernel_error)
    }
}

/// Evaluate gross information value for the declared finite observation partition.
#[pyfunction]
pub(super) fn compute_censored_sequential_evsi<'py>(
    py: Python<'py>,
    experiment_json: &str,
    max_branches: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let budget = NonZeroUsize::new(max_branches)
        .ok_or_else(|| invalid("A positive outcome branch budget is required"))?;
    let Experiment {
        strategy_ids,
        states,
        stages,
    } = serde_json::from_str(experiment_json).map_err(invalid)?;
    let strategy_ids = identifiers("strategy IDs", strategy_ids.iter().map(String::as_str))?;
    let state_ids = identifiers(
        "state IDs",
        states.iter().map(|state| state.state_id.as_str()),
    )?;
    if states
        .iter()
        .any(|state| state.net_benefits.len() != strategy_ids.len())
    {
        return Err(BoundaryError::DimensionMismatch(
            "Every payoff row must match the declared strategy IDs".into(),
        )
        .into_pyerr());
    }

    let prior = states
        .iter()
        .map(|state| state.prior_weight)
        .collect::<Vec<_>>();
    let benefits = SampleMatrix::try_from(
        states
            .into_iter()
            .map(|state| state.net_benefits)
            .collect::<Vec<_>>(),
    )
    .map_err(invalid)?;
    let result = py.detach(|| {
        let stages = stages
            .into_iter()
            .map(|stage| {
                stage
                    .into_iter()
                    .map(|action| action.into_channel(&state_ids))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        sequential_evsi(&prior, &benefits, &stages, budget).map_err(kernel_error)
    })?;
    let payload = result_payload(&result, &state_ids, &strategy_ids);
    result_to_dict(py, &payload).map(|(result, _canonical)| result)
}

fn identifier(id: &str) -> PyResult<Identifier> {
    Identifier::new(canonical_id(id)?).map_err(invalid)
}

fn identifiers<'a>(field: &str, ids: impl Iterator<Item = &'a str>) -> PyResult<Vec<Identifier>> {
    let ids = ids.map(identifier).collect::<PyResult<Vec<_>>>()?;
    if ids.is_empty() || ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(invalid(format!("{field} must be nonempty and unique")));
    }
    Ok(ids)
}

fn state_mismatch() -> PyErr {
    BoundaryError::DimensionMismatch(
        "Acquisition state IDs must match the declared payoff state IDs".into(),
    )
    .into_pyerr()
}

fn result_payload(
    result: &SequentialEvsi,
    state_ids: &[Identifier],
    strategy_ids: &[Identifier],
) -> Value {
    let state_ids = state_ids.iter().map(Identifier::as_str).collect::<Vec<_>>();
    let strategy_ids = strategy_ids
        .iter()
        .map(Identifier::as_str)
        .collect::<Vec<_>>();
    match result {
        SequentialEvsi::Complete {
            current_value,
            expected_value,
            evsi,
            branches_examined,
        } => json!({
            "contract_version": 1,
            "status": "complete",
            "state_ids": state_ids,
            "strategy_ids": strategy_ids,
            "current_value": fraction(current_value),
            "expected_value": fraction(expected_value),
            "evsi": fraction(evsi),
            "branches_examined": branches_examined,
        }),
        SequentialEvsi::Indeterminate {
            current_value,
            branches_examined,
        } => json!({
            "contract_version": 1,
            "status": "indeterminate",
            "state_ids": state_ids,
            "strategy_ids": strategy_ids,
            "current_value": fraction(current_value),
            "branches_examined": branches_examined,
        }),
    }
}
