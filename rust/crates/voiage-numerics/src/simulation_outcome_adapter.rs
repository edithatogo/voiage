//! Optional native adapter mapping validated `voiage.simulation-outcome.v1` profiles
//! onto core Rust VOI input contracts and EVPI kernels (`O05`, `O07`).

use std::collections::BTreeMap;
use std::fmt;

use voiage_domain::{
    validate_profile_version, CoreContractError, DrawIdentity, Identifier, ObjectiveDirection,
    ParameterSet, ProducerProvenance, SampleCube, SampleVector, SimulationOutcomeError,
    SimulationOutcomeProfileInput, StrategyId, ValidatedSimulationOutcomeProfile, ValueArray,
    ValueConvention, ValueDeclaration, WeightSemantics,
};

use crate::{evpi, structural_evpi, NumericalInputError};

/// Failures raised when converting or evaluating a validated simulation-outcome profile.
#[derive(Clone, Debug, PartialEq)]
pub enum SimulationOutcomeAdapterError {
    /// Upstream simulation-outcome profile validation failed.
    Domain(SimulationOutcomeError),
    /// Constructing `ValueArray` or `ParameterSet` failed.
    CoreContract(CoreContractError),
    /// Numerical kernel pre-conditions failed (for example probability-sum tolerance).
    Numerical(NumericalInputError),
    /// Loss / minimize profiles are explicitly rejected at the EVPI handoff without sign flips.
    UnsupportedObjectiveDirection {
        /// Declared value convention on the profile.
        convention: ValueConvention,
        /// Declared objective direction on the profile.
        objective: ObjectiveDirection,
    },
}

impl fmt::Display for SimulationOutcomeAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(error) => write!(formatter, "simulation-outcome domain error: {error}"),
            Self::CoreContract(error) => {
                write!(formatter, "core VOI contract handoff error: {error}")
            }
            Self::Numerical(error) => write!(formatter, "numerical EVPI kernel error: {error}"),
            Self::UnsupportedObjectiveDirection {
                convention,
                objective,
            } => write!(
                formatter,
                "unsupported objective direction for EVPI without implicit sign conversion: convention={}, objective={}",
                convention.as_str(),
                objective.as_str()
            ),
        }
    }
}

impl std::error::Error for SimulationOutcomeAdapterError {}

impl From<SimulationOutcomeError> for SimulationOutcomeAdapterError {
    fn from(value: SimulationOutcomeError) -> Self {
        Self::Domain(value)
    }
}

impl From<CoreContractError> for SimulationOutcomeAdapterError {
    fn from(value: CoreContractError) -> Self {
        Self::CoreContract(value)
    }
}

impl From<NumericalInputError> for SimulationOutcomeAdapterError {
    fn from(value: NumericalInputError) -> Self {
        Self::Numerical(value)
    }
}

/// Core Rust VOI contracts alongside preserved simulation-outcome receipts and provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulationOutcomeHandoff {
    profile_version: String,
    provenance: ProducerProvenance,
    draw_namespace: Identifier,
    value_declaration: ValueDeclaration,
    weight_semantics: WeightSemantics,
    draw_keys: Vec<Identifier>,
    draw_identities: Vec<DrawIdentity>,
    draw_weights: Vec<f64>,
    value_array: ValueArray,
    parameter_set: Option<ParameterSet>,
}

impl SimulationOutcomeHandoff {
    /// Declared profile version string.
    #[must_use]
    pub fn profile_version(&self) -> &str {
        &self.profile_version
    }

    /// Validated producer provenance preserved through the handoff.
    #[must_use]
    pub const fn provenance(&self) -> &ProducerProvenance {
        &self.provenance
    }

    /// Producer draw namespace governing all draw keys.
    #[must_use]
    pub const fn draw_namespace(&self) -> &Identifier {
        &self.draw_namespace
    }

    /// Validated unit, perspective, convention, and objective declarations.
    #[must_use]
    pub const fn value_declaration(&self) -> &ValueDeclaration {
        &self.value_declaration
    }

    /// Validated draw-weight interpretation (`equal`, `probability`, `relative_weight`).
    #[must_use]
    pub const fn weight_semantics(&self) -> WeightSemantics {
        self.weight_semantics
    }

    /// Canonical draw keys matching the rows of `value_array` and `parameter_set`.
    #[must_use]
    pub fn draw_keys(&self) -> &[Identifier] {
        &self.draw_keys
    }

    /// Deterministic qualified draw-identity receipt.
    #[must_use]
    pub fn draw_identities(&self) -> &[DrawIdentity] {
        &self.draw_identities
    }

    /// Canonical draw weights matching `draw_keys`.
    #[must_use]
    pub fn draw_weights(&self) -> &[f64] {
        &self.draw_weights
    }

    /// Core `ValueArray` populated losslessly from the paired outcome matrix.
    #[must_use]
    pub const fn value_array(&self) -> &ValueArray {
        &self.value_array
    }

    /// Optional core `ParameterSet` aligned by `draw_key`.
    #[must_use]
    pub const fn parameter_set(&self) -> Option<&ParameterSet> {
        self.parameter_set.as_ref()
    }
}

/// EVPI evaluation receipt paired with preserved outcome declarations and provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulationOutcomeEvpiResult {
    evpi: f64,
    value_declaration: ValueDeclaration,
    weight_semantics: WeightSemantics,
    draw_identities: Vec<DrawIdentity>,
    strategy_ids: Vec<StrategyId>,
    provenance: ProducerProvenance,
}

impl SimulationOutcomeEvpiResult {
    /// Computed EVPI value in the declared unit and perspective.
    #[must_use]
    pub const fn evpi(&self) -> f64 {
        self.evpi
    }

    /// Validated value declaration retained from the profile.
    #[must_use]
    pub const fn value_declaration(&self) -> &ValueDeclaration {
        &self.value_declaration
    }

    /// Validated weight semantics applied during EVPI evaluation.
    #[must_use]
    pub const fn weight_semantics(&self) -> WeightSemantics {
        self.weight_semantics
    }

    /// Qualified draw identities used to pair strategies and parameters.
    #[must_use]
    pub fn draw_identities(&self) -> &[DrawIdentity] {
        &self.draw_identities
    }

    /// Ordered strategy identifiers evaluated by the EVPI kernel.
    #[must_use]
    pub fn strategy_ids(&self) -> &[StrategyId] {
        &self.strategy_ids
    }

    /// Preserved producer provenance.
    #[must_use]
    pub const fn provenance(&self) -> &ProducerProvenance {
        &self.provenance
    }
}

/// Maps a validated v1 simulation-outcome profile to `ValueArray` and optional `ParameterSet`.
///
/// Values, pairing order, weights, declarations, and provenance are preserved without
/// implicit unit, sign, or perspective conversions.
///
/// # Errors
///
/// Returns [`SimulationOutcomeAdapterError`] when profile version checks or core
/// `ValueArray` / `ParameterSet` construction fail.
pub fn adapt_simulation_outcome(
    profile: &ValidatedSimulationOutcomeProfile,
) -> Result<SimulationOutcomeHandoff, SimulationOutcomeAdapterError> {
    validate_profile_version(profile.profile_version())?;

    let strategy_names: Vec<String> = profile
        .strategy_ids()
        .iter()
        .map(|strategy| strategy.as_str().to_owned())
        .collect();
    let value_array_id = format!(
        "{}:{}",
        profile.draw_namespace().as_str(),
        profile.provenance().artifact_digest()
    );
    let value_array = ValueArray::new(
        value_array_id,
        strategy_names,
        profile.payoff_matrix().to_vec(),
    )?;

    let parameter_set = match profile.parameter_samples() {
        Some(aligned) => {
            let mut columns: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            for name in aligned.parameter_names() {
                columns.insert(
                    name.as_str().to_owned(),
                    Vec::with_capacity(aligned.draw_keys().len()),
                );
            }
            for row in aligned.matrix() {
                for (name, &value) in aligned.parameter_names().iter().zip(row.iter()) {
                    if let Some(column) = columns.get_mut(name.as_str()) {
                        column.push(value);
                    }
                }
            }
            let param_set_id = format!("{}:parameters", profile.draw_namespace().as_str());
            Some(ParameterSet::new(param_set_id, columns)?)
        }
        None => None,
    };

    Ok(SimulationOutcomeHandoff {
        profile_version: profile.profile_version().to_owned(),
        provenance: profile.provenance().clone(),
        draw_namespace: profile.draw_namespace().clone(),
        value_declaration: profile.value_declaration().clone(),
        weight_semantics: profile.weight_semantics(),
        draw_keys: profile.draw_keys().to_vec(),
        draw_identities: profile.draw_identities(),
        draw_weights: profile.draw_weights().to_vec(),
        value_array,
        parameter_set,
    })
}

/// Validates raw v1 simulation-outcome input and converts it to core Rust VOI contracts.
///
/// # Errors
///
/// Returns [`SimulationOutcomeAdapterError::Domain`] on input validation failures or
/// [`SimulationOutcomeAdapterError::CoreContract`] on handoff construction failures.
pub fn adapt_simulation_outcome_input(
    input: &SimulationOutcomeProfileInput,
) -> Result<SimulationOutcomeHandoff, SimulationOutcomeAdapterError> {
    let validated = ValidatedSimulationOutcomeProfile::validate(input)?;
    adapt_simulation_outcome(&validated)
}

/// Computes EVPI from a validated v1 simulation-outcome profile using existing Rust VOI kernels.
///
/// - [`WeightSemantics::Equal`] dispatches directly to [`evpi`].
/// - [`WeightSemantics::Probability`] passes exact weights without silent rescaling to [`structural_evpi`].
/// - [`WeightSemantics::RelativeWeight`] normalizes finite nonnegative weights and dispatches to [`structural_evpi`].
/// - [`ObjectiveDirection::Minimize`] / [`ValueConvention::Loss`] fail explicitly without sign flips.
///
/// # Errors
///
/// Returns [`SimulationOutcomeAdapterError::UnsupportedObjectiveDirection`] for minimize/loss
/// objectives, [`SimulationOutcomeAdapterError::Numerical`] if kernel constraints fail, or
/// adapter conversion errors.
pub fn simulation_outcome_evpi(
    profile: &ValidatedSimulationOutcomeProfile,
) -> Result<SimulationOutcomeEvpiResult, SimulationOutcomeAdapterError> {
    let declaration = profile.value_declaration();
    if declaration.objective() != ObjectiveDirection::Maximize
        || declaration.convention() == ValueConvention::Loss
    {
        return Err(
            SimulationOutcomeAdapterError::UnsupportedObjectiveDirection {
                convention: declaration.convention(),
                objective: declaration.objective(),
            },
        );
    }

    let handoff = adapt_simulation_outcome(profile)?;
    let computed_evpi = match handoff.weight_semantics() {
        WeightSemantics::Equal => evpi(handoff.value_array().net_benefit())?,
        WeightSemantics::Probability => {
            let cube = single_row_planes_cube(handoff.value_array())?;
            let probabilities =
                SampleVector::try_from(handoff.draw_weights().to_vec()).map_err(|_| {
                    NumericalInputError::invalid(
                        "draw_weights",
                        "probability weights could not form a finite probability vector",
                    )
                })?;
            structural_evpi(&cube, &probabilities)?
        }
        WeightSemantics::RelativeWeight => {
            let cube = single_row_planes_cube(handoff.value_array())?;
            let probabilities = normalized_relative_weights(handoff.draw_weights())?;
            structural_evpi(&cube, &probabilities)?
        }
    };

    Ok(SimulationOutcomeEvpiResult {
        evpi: computed_evpi,
        value_declaration: profile.value_declaration().clone(),
        weight_semantics: profile.weight_semantics(),
        draw_identities: profile.draw_identities(),
        strategy_ids: profile.strategy_ids().to_vec(),
        provenance: profile.provenance().clone(),
    })
}

fn single_row_planes_cube(
    value_array: &ValueArray,
) -> Result<SampleCube, SimulationOutcomeAdapterError> {
    let planes: Vec<Vec<Vec<f64>>> = value_array
        .net_benefit()
        .rows()
        .map(|row| vec![row.to_vec()])
        .collect();
    SampleCube::try_from(planes).map_err(|_| {
        NumericalInputError::invalid(
            "net_benefit",
            "could not construct sample cube for weighted EVPI evaluation",
        )
        .into()
    })
}

fn normalized_relative_weights(
    weights: &[f64],
) -> Result<SampleVector, SimulationOutcomeAdapterError> {
    let max_weight = weights.iter().copied().fold(0.0_f64, f64::max);
    if !max_weight.is_finite() || max_weight <= 0.0 {
        return Err(NumericalInputError::invalid(
            "draw_weights",
            "relative weights must contain at least one positive finite weight",
        )
        .into());
    }
    let scaled: Vec<f64> = weights.iter().map(|weight| weight / max_weight).collect();
    // Compensated (Neumaier) summation avoids systematic drift in the relative-weight denominator.
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for &value in &scaled {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    total += compensation;
    if !total.is_finite() || total <= 0.0 {
        return Err(NumericalInputError::invalid(
            "draw_weights",
            "normalized relative weights must have a finite positive sum",
        )
        .into());
    }
    let probabilities: Vec<f64> = scaled.into_iter().map(|weight| weight / total).collect();
    SampleVector::try_from(probabilities).map_err(|_| {
        NumericalInputError::invalid(
            "draw_weights",
            "normalized relative weights produced non-finite probabilities",
        )
        .into()
    })
}
