//! Versioned simulation-outcome input contracts (`O01`–`O04`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Identifier, StrategyId};

/// Canonical version identifier for the v1 simulation-outcome profile.
pub const SIMULATION_OUTCOME_PROFILE_VERSION_V1: &str = "voiage.simulation-outcome.v1";

/// Validation failures for optional simulation-outcome input profiles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationOutcomeError {
    /// The declared profile version is empty, unknown, or unsupported.
    UnsupportedProfileVersion(String),
    /// A required identifier or provenance string was empty after trimming.
    EmptyIdentifier(&'static str),
    /// An identifier or declaration string had leading or trailing whitespace.
    UntrimmedIdentifier(&'static str),
    /// The strategy list was empty or contained duplicate identifiers.
    InvalidStrategies,
    /// No outcome records were supplied.
    EmptyOutcomes,
    /// An outcome or parameter record used a draw namespace different from the profile.
    MismatchedDrawNamespace {
        /// Profile-declared draw namespace.
        expected: String,
        /// Namespace observed on a record.
        actual: String,
    },
    /// An outcome record referenced a strategy not declared by the profile.
    UnknownStrategy(String),
    /// Two outcome records shared the same `(draw_key, strategy_id)` pair.
    DuplicateOutcomeRecord {
        /// Conflicting draw key.
        draw_key: String,
        /// Conflicting strategy identifier.
        strategy_id: String,
    },
    /// A draw did not provide an outcome for every declared strategy.
    IncompleteDraw {
        /// Draw key missing one or more strategies.
        draw_key: String,
        /// Strategy identifier missing from the draw.
        missing_strategy: String,
    },
    /// Outcome value was NaN or infinite.
    NonFiniteValue {
        /// Draw key carrying the invalid value.
        draw_key: String,
        /// Strategy identifier carrying the invalid value.
        strategy_id: String,
    },
    /// A draw weight was non-finite or negative.
    InvalidWeight(String),
    /// Two records for the same draw declared different weights.
    ConflictingDrawWeight(String),
    /// Weight semantics string was unsupported.
    UnsupportedWeightSemantics(String),
    /// Supplied draw weights did not satisfy the declared weight semantics.
    IncompatibleWeightSemantics(String),
    /// Value convention string was unsupported.
    UnsupportedValueConvention(String),
    /// Objective direction string was unsupported.
    UnsupportedObjectiveDirection(String),
    /// Value convention and objective direction conflicted without implicit sign conversion.
    IncompatibleConventionObjective {
        /// Declared value convention.
        convention: String,
        /// Declared objective direction.
        objective: String,
    },
    /// Per-record value declaration disagreed with the profile declaration.
    IncompatibleValueDeclaration(&'static str),
    /// Two parameter sample records shared the same `draw_key`.
    DuplicateParameterDraw(String),
    /// A parameter sample referenced a `draw_key` absent from validated outcomes.
    UnmatchedParameterDraw(String),
    /// A validated outcome `draw_key` was missing from supplied parameter samples.
    MissingParameterDraw(String),
    /// A parameter sample contained no parameter entries.
    EmptyParameterValues(String),
    /// Parameter names differed across supplied parameter draws.
    InconsistentParameterSchema(String),
    /// A parameter sample value was NaN or infinite.
    NonFiniteParameterValue {
        /// Draw key carrying the non-finite parameter value.
        draw_key: String,
        /// Parameter name carrying the non-finite value.
        parameter_id: String,
    },
}

impl fmt::Display for SimulationOutcomeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid simulation outcome profile: {self:?}")
    }
}

impl std::error::Error for SimulationOutcomeError {}

/// Objective optimization direction declared by an outcome producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    /// Higher values are preferred (utility, net monetary benefit).
    Maximize,
    /// Lower values are preferred (loss, regret, cost).
    Minimize,
}

impl ObjectiveDirection {
    /// Parses a canonical objective-direction string.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationOutcomeError::UnsupportedObjectiveDirection`] for
    /// any string other than `"maximize"` or `"minimize"`.
    pub fn parse(raw: &str) -> Result<Self, SimulationOutcomeError> {
        match raw {
            "maximize" => Ok(Self::Maximize),
            "minimize" => Ok(Self::Minimize),
            _ => Err(SimulationOutcomeError::UnsupportedObjectiveDirection(
                raw.to_owned(),
            )),
        }
    }

    /// Canonical wire representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Maximize => "maximize",
            Self::Minimize => "minimize",
        }
    }
}

/// Value convention declared by an outcome producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueConvention {
    /// Decision utility (requires [`ObjectiveDirection::Maximize`]).
    Utility,
    /// Net monetary or net health benefit (requires [`ObjectiveDirection::Maximize`]).
    NetBenefit,
    /// Loss or cost (requires [`ObjectiveDirection::Minimize`]).
    Loss,
}

impl ValueConvention {
    /// Parses a canonical value-convention string.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationOutcomeError::UnsupportedValueConvention`] for any
    /// string other than `"utility"`, `"net_benefit"`, or `"loss"`.
    pub fn parse(raw: &str) -> Result<Self, SimulationOutcomeError> {
        match raw {
            "utility" => Ok(Self::Utility),
            "net_benefit" => Ok(Self::NetBenefit),
            "loss" => Ok(Self::Loss),
            _ => Err(SimulationOutcomeError::UnsupportedValueConvention(
                raw.to_owned(),
            )),
        }
    }

    /// Canonical wire representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Utility => "utility",
            Self::NetBenefit => "net_benefit",
            Self::Loss => "loss",
        }
    }

    /// Checks whether this convention matches the declared objective direction.
    #[must_use]
    pub const fn is_compatible_with(self, objective: ObjectiveDirection) -> bool {
        matches!(
            (self, objective),
            (
                Self::Utility | Self::NetBenefit,
                ObjectiveDirection::Maximize
            ) | (Self::Loss, ObjectiveDirection::Minimize)
        )
    }
}

/// Draw weight interpretation declared by an outcome producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightSemantics {
    /// Every draw carries an identical positive weight.
    Equal,
    /// Draw weights are probabilities summing to `1.0`.
    Probability,
    /// Draw weights are nonnegative relative weights with a positive finite total.
    RelativeWeight,
}

impl WeightSemantics {
    /// Parses a canonical weight-semantics string.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationOutcomeError::UnsupportedWeightSemantics`] for any
    /// unsupported weight semantics string.
    pub fn parse(raw: &str) -> Result<Self, SimulationOutcomeError> {
        match raw {
            "equal" => Ok(Self::Equal),
            "probability" => Ok(Self::Probability),
            "relative_weight" => Ok(Self::RelativeWeight),
            _ => Err(SimulationOutcomeError::UnsupportedWeightSemantics(
                raw.to_owned(),
            )),
        }
    }

    /// Canonical wire representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Probability => "probability",
            Self::RelativeWeight => "relative_weight",
        }
    }
}

/// Raw producer provenance payload supplied with an outcome profile.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerProvenanceInput {
    /// Producer identifier (for example `kairos` or a synthetic label).
    pub producer_id: String,
    /// Producer package or release version.
    pub producer_version: String,
    /// Producer model or scenario revision identifier.
    pub model_revision: String,
    /// Identifier of the valuation source or specification.
    pub valuation_source: String,
    /// Content digest of the upstream artifact.
    pub artifact_digest: String,
}

/// Validated producer provenance retained alongside validated outcomes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ProducerProvenance {
    producer_id: Identifier,
    producer_version: Identifier,
    model_revision: Identifier,
    valuation_source: Identifier,
    artifact_digest: Identifier,
}

impl ProducerProvenance {
    /// Validates nonempty, untrimmed provenance identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationOutcomeError::EmptyIdentifier`] when a field is
    /// blank or [`SimulationOutcomeError::UntrimmedIdentifier`] when it has
    /// leading or trailing whitespace.
    pub fn from_input(input: &ProducerProvenanceInput) -> Result<Self, SimulationOutcomeError> {
        Ok(Self {
            producer_id: strict_non_empty_id(&input.producer_id, "producer_id")?,
            producer_version: strict_non_empty_id(&input.producer_version, "producer_version")?,
            model_revision: strict_non_empty_id(&input.model_revision, "model_revision")?,
            valuation_source: strict_non_empty_id(&input.valuation_source, "valuation_source")?,
            artifact_digest: strict_non_empty_id(&input.artifact_digest, "artifact_digest")?,
        })
    }

    /// Returns the validated producer identifier.
    #[must_use]
    pub fn producer_id(&self) -> &str {
        self.producer_id.as_str()
    }

    /// Returns the validated producer version.
    #[must_use]
    pub fn producer_version(&self) -> &str {
        self.producer_version.as_str()
    }

    /// Returns the validated model revision.
    #[must_use]
    pub fn model_revision(&self) -> &str {
        self.model_revision.as_str()
    }

    /// Returns the validated valuation source.
    #[must_use]
    pub fn valuation_source(&self) -> &str {
        self.valuation_source.as_str()
    }

    /// Returns the validated artifact digest.
    #[must_use]
    pub fn artifact_digest(&self) -> &str {
        self.artifact_digest.as_str()
    }
}

/// Raw unit, perspective, convention, and scope declarations.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueDeclarationInput {
    /// Value unit declared by the producer (for example `NZD` or `utils`).
    pub unit: String,
    /// Decision perspective declared by the producer.
    pub perspective: String,
    /// Value convention (`utility`, `net_benefit`, or `loss`).
    pub convention: String,
    /// Objective direction (`maximize` or `minimize`).
    pub objective: String,
    /// Optional currency and price-year basis.
    #[serde(default)]
    pub currency_price_year: Option<String>,
    /// Optional target population basis.
    #[serde(default)]
    pub population_basis: Option<String>,
    /// Optional evaluation time horizon.
    #[serde(default)]
    pub horizon: Option<String>,
    /// Optional discounting declaration.
    #[serde(default)]
    pub discounting_status: Option<String>,
}

/// Validated explicit value and objective declaration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ValueDeclaration {
    unit: Identifier,
    perspective: Identifier,
    convention: ValueConvention,
    objective: ObjectiveDirection,
    currency_price_year: Option<Identifier>,
    population_basis: Option<Identifier>,
    horizon: Option<Identifier>,
    discounting_status: Option<Identifier>,
}

impl ValueDeclaration {
    /// Validates required declarations and convention/objective compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are blank or untrimmed, optional
    /// fields are blank or untrimmed when supplied, convention/objective
    /// strings are unsupported, or the convention conflicts with the objective.
    pub fn from_input(input: &ValueDeclarationInput) -> Result<Self, SimulationOutcomeError> {
        let unit = strict_non_empty_id(&input.unit, "unit")?;
        let perspective = strict_non_empty_id(&input.perspective, "perspective")?;
        let convention = ValueConvention::parse(&input.convention)?;
        let objective = ObjectiveDirection::parse(&input.objective)?;
        if !convention.is_compatible_with(objective) {
            return Err(SimulationOutcomeError::IncompatibleConventionObjective {
                convention: convention.as_str().to_owned(),
                objective: objective.as_str().to_owned(),
            });
        }
        let currency_price_year =
            optional_id(input.currency_price_year.as_deref(), "currency_price_year")?;
        let population_basis = optional_id(input.population_basis.as_deref(), "population_basis")?;
        let horizon = optional_id(input.horizon.as_deref(), "horizon")?;
        let discounting_status =
            optional_id(input.discounting_status.as_deref(), "discounting_status")?;
        Ok(Self {
            unit,
            perspective,
            convention,
            objective,
            currency_price_year,
            population_basis,
            horizon,
            discounting_status,
        })
    }

    /// Rejects any disagreement between two declarations without implicit conversion.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationOutcomeError::IncompatibleValueDeclaration`] naming
    /// the first mismatching declaration field.
    pub fn validate_compatible(&self, other: &Self) -> Result<(), SimulationOutcomeError> {
        if self.unit != other.unit {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration("unit"));
        }
        if self.perspective != other.perspective {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration(
                "perspective",
            ));
        }
        if self.convention != other.convention {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration(
                "convention",
            ));
        }
        if self.objective != other.objective {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration(
                "objective",
            ));
        }
        if self.currency_price_year != other.currency_price_year {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration(
                "currency_price_year",
            ));
        }
        if self.population_basis != other.population_basis {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration(
                "population_basis",
            ));
        }
        if self.horizon != other.horizon {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration(
                "horizon",
            ));
        }
        if self.discounting_status != other.discounting_status {
            return Err(SimulationOutcomeError::IncompatibleValueDeclaration(
                "discounting_status",
            ));
        }
        Ok(())
    }

    /// Returns the validated unit string.
    #[must_use]
    pub fn unit(&self) -> &str {
        self.unit.as_str()
    }

    /// Returns the validated perspective string.
    #[must_use]
    pub fn perspective(&self) -> &str {
        self.perspective.as_str()
    }

    /// Returns the validated value convention.
    #[must_use]
    pub const fn convention(&self) -> ValueConvention {
        self.convention
    }

    /// Returns the validated objective direction.
    #[must_use]
    pub const fn objective(&self) -> ObjectiveDirection {
        self.objective
    }

    /// Returns the optional currency/price-year declaration.
    #[must_use]
    pub fn currency_price_year(&self) -> Option<&str> {
        self.currency_price_year.as_ref().map(Identifier::as_str)
    }

    /// Returns the optional population-basis declaration.
    #[must_use]
    pub fn population_basis(&self) -> Option<&str> {
        self.population_basis.as_ref().map(Identifier::as_str)
    }

    /// Returns the optional time-horizon declaration.
    #[must_use]
    pub fn horizon(&self) -> Option<&str> {
        self.horizon.as_ref().map(Identifier::as_str)
    }

    /// Returns the optional discounting declaration.
    #[must_use]
    pub fn discounting_status(&self) -> Option<&str> {
        self.discounting_status.as_ref().map(Identifier::as_str)
    }
}

/// Qualified draw identity combining producer namespace and draw key.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct DrawIdentity {
    namespace: Identifier,
    draw_key: Identifier,
}

impl DrawIdentity {
    /// Constructs a validated draw identity without whitespace trimming.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationOutcomeError::EmptyIdentifier`] or
    /// [`SimulationOutcomeError::UntrimmedIdentifier`] when `namespace` or
    /// `draw_key` is blank or untrimmed.
    pub fn new(namespace: &str, draw_key: &str) -> Result<Self, SimulationOutcomeError> {
        Ok(Self {
            namespace: strict_non_empty_id(namespace, "draw_namespace")?,
            draw_key: strict_non_empty_id(draw_key, "draw_key")?,
        })
    }

    /// Returns the draw namespace.
    #[must_use]
    pub fn namespace(&self) -> &str {
        self.namespace.as_str()
    }

    /// Returns the draw key within the namespace.
    #[must_use]
    pub fn draw_key(&self) -> &str {
        self.draw_key.as_str()
    }
}

/// Raw paired outcome record supplied for one `(draw_key, strategy_id)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeRecordInput {
    /// Producer draw namespace (must match the profile's `draw_namespace`).
    pub draw_namespace: String,
    /// Producer draw key within `draw_namespace`.
    pub draw_key: String,
    /// Strategy identifier declared in the profile's `strategy_ids`.
    pub strategy_id: String,
    /// Already-valued outcome number.
    pub value: f64,
    /// Draw weight shared across all strategies for `draw_key`.
    pub weight: f64,
    /// Optional per-record declaration checked against the profile declaration.
    #[serde(default)]
    pub declaration: Option<ValueDeclarationInput>,
}

/// Raw parameter sample record keyed by `(draw_namespace, draw_key)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterSampleInput {
    /// Producer draw namespace (must match the profile's `draw_namespace`).
    pub draw_namespace: String,
    /// Producer draw key matching a validated outcome draw.
    pub draw_key: String,
    /// Named finite parameter values for this draw.
    pub values: BTreeMap<String, f64>,
}

/// Raw simulation-outcome profile payload before domain validation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationOutcomeProfileInput {
    /// Declared profile version (must equal [`SIMULATION_OUTCOME_PROFILE_VERSION_V1`]).
    pub profile_version: String,
    /// Producer provenance metadata.
    pub provenance: ProducerProvenanceInput,
    /// Ordered list of unique strategy identifiers.
    pub strategy_ids: Vec<String>,
    /// Declared unit, perspective, convention, and objective.
    pub value_declaration: ValueDeclarationInput,
    /// Declared draw-weight semantics (`equal`, `probability`, `relative_weight`).
    pub weight_semantics: String,
    /// Producer draw namespace governing all outcome and parameter rows.
    pub draw_namespace: String,
    /// Paired outcome records covering every `(draw_key, strategy_id)`.
    pub outcomes: Vec<OutcomeRecordInput>,
    /// Optional draw-keyed parameter samples aligned by `draw_key`.
    #[serde(default)]
    pub parameter_samples: Option<Vec<ParameterSampleInput>>,
}

/// Draw-aligned parameter samples ordered canonically by `draw_key` and `parameter_name`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AlignedParameterSamples {
    parameter_names: Vec<Identifier>,
    draw_keys: Vec<Identifier>,
    matrix: Vec<Vec<f64>>,
}

impl AlignedParameterSamples {
    /// Ordered validated parameter names.
    #[must_use]
    pub fn parameter_names(&self) -> &[Identifier] {
        &self.parameter_names
    }

    /// Canonical draw keys matching the parent outcome profile.
    #[must_use]
    pub fn draw_keys(&self) -> &[Identifier] {
        &self.draw_keys
    }

    /// Draw-by-parameter matrix (`n_draws x n_parameters`).
    #[must_use]
    pub fn matrix(&self) -> &[Vec<f64>] {
        &self.matrix
    }
}

/// Validated v1 simulation-outcome profile with canonical pairing and key-mapping receipt.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ValidatedSimulationOutcomeProfile {
    profile_version: String,
    provenance: ProducerProvenance,
    draw_namespace: Identifier,
    strategy_ids: Vec<StrategyId>,
    value_declaration: ValueDeclaration,
    weight_semantics: WeightSemantics,
    draw_keys: Vec<Identifier>,
    payoff_matrix: Vec<Vec<f64>>,
    draw_weights: Vec<f64>,
    parameter_samples: Option<AlignedParameterSamples>,
}

impl ValidatedSimulationOutcomeProfile {
    /// Validates a raw v1 simulation-outcome profile without implicit conversions.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationOutcomeError`] if the profile version is unsupported,
    /// declarations are missing or incompatible, strategy/draw pairing is
    /// incomplete or duplicated, weights violate declared semantics, or
    /// supplied parameter draws fail to align by `draw_key`.
    pub fn validate(input: &SimulationOutcomeProfileInput) -> Result<Self, SimulationOutcomeError> {
        validate_profile_version(&input.profile_version)?;
        let provenance = ProducerProvenance::from_input(&input.provenance)?;
        let draw_namespace = strict_non_empty_id(&input.draw_namespace, "draw_namespace")?;
        let strategy_ids = validate_strategy_ids(&input.strategy_ids)?;
        let value_declaration = ValueDeclaration::from_input(&input.value_declaration)?;
        let weight_semantics = WeightSemantics::parse(&input.weight_semantics)?;
        if input.outcomes.is_empty() {
            return Err(SimulationOutcomeError::EmptyOutcomes);
        }

        let strategy_index: BTreeMap<&str, usize> = strategy_ids
            .iter()
            .enumerate()
            .map(|(index, strategy)| (strategy.as_str(), index))
            .collect();

        let (draw_keys, payoff_matrix, draw_weights) = pair_outcomes(
            &input.outcomes,
            &draw_namespace,
            &strategy_ids,
            &strategy_index,
            &value_declaration,
            weight_semantics,
        )?;

        let parameter_samples = match input.parameter_samples.as_deref() {
            Some(samples) => Some(align_parameter_samples(
                samples,
                &draw_namespace,
                &draw_keys,
            )?),
            None => None,
        };

        Ok(Self {
            profile_version: SIMULATION_OUTCOME_PROFILE_VERSION_V1.to_owned(),
            provenance,
            draw_namespace,
            strategy_ids,
            value_declaration,
            weight_semantics,
            draw_keys,
            payoff_matrix,
            draw_weights,
            parameter_samples,
        })
    }

    /// Declared profile version string.
    #[must_use]
    pub fn profile_version(&self) -> &str {
        &self.profile_version
    }

    /// Validated producer provenance.
    #[must_use]
    pub const fn provenance(&self) -> &ProducerProvenance {
        &self.provenance
    }

    /// Validated draw namespace.
    #[must_use]
    pub const fn draw_namespace(&self) -> &Identifier {
        &self.draw_namespace
    }

    /// Declared strategy identifiers in order.
    #[must_use]
    pub fn strategy_ids(&self) -> &[StrategyId] {
        &self.strategy_ids
    }

    /// Validated value, unit, perspective, and objective declaration.
    #[must_use]
    pub const fn value_declaration(&self) -> &ValueDeclaration {
        &self.value_declaration
    }

    /// Validated weight semantics.
    #[must_use]
    pub const fn weight_semantics(&self) -> WeightSemantics {
        self.weight_semantics
    }

    /// Canonical sorted draw keys forming the deterministic key-mapping receipt.
    #[must_use]
    pub fn draw_keys(&self) -> &[Identifier] {
        &self.draw_keys
    }

    /// Qualified draw identities combining `draw_namespace` with each canonical `draw_key`.
    #[must_use]
    pub fn draw_identities(&self) -> Vec<DrawIdentity> {
        self.draw_keys
            .iter()
            .map(|draw_key| DrawIdentity {
                namespace: self.draw_namespace.clone(),
                draw_key: draw_key.clone(),
            })
            .collect()
    }

    /// Canonical `n_draws x n_strategies` payoff matrix aligned to `draw_keys` and `strategy_ids`.
    #[must_use]
    pub fn payoff_matrix(&self) -> &[Vec<f64>] {
        &self.payoff_matrix
    }

    /// Canonical draw weights aligned to `draw_keys`.
    #[must_use]
    pub fn draw_weights(&self) -> &[f64] {
        &self.draw_weights
    }

    /// Optional validated parameter samples aligned by `draw_key`.
    #[must_use]
    pub const fn parameter_samples(&self) -> Option<&AlignedParameterSamples> {
        self.parameter_samples.as_ref()
    }
}

/// Validates that the profile version matches [`SIMULATION_OUTCOME_PROFILE_VERSION_V1`].
///
/// # Errors
///
/// Returns [`SimulationOutcomeError::UnsupportedProfileVersion`] for any
/// mismatching or empty version string.
pub fn validate_profile_version(version: &str) -> Result<(), SimulationOutcomeError> {
    if version != SIMULATION_OUTCOME_PROFILE_VERSION_V1 {
        return Err(SimulationOutcomeError::UnsupportedProfileVersion(
            version.to_owned(),
        ));
    }
    Ok(())
}

fn strict_non_empty_id(
    raw: &str,
    field: &'static str,
) -> Result<Identifier, SimulationOutcomeError> {
    if raw.trim().is_empty() {
        return Err(SimulationOutcomeError::EmptyIdentifier(field));
    }
    if raw != raw.trim() {
        return Err(SimulationOutcomeError::UntrimmedIdentifier(field));
    }
    Identifier::new(raw).map_err(|_| SimulationOutcomeError::EmptyIdentifier(field))
}

fn optional_id(
    raw: Option<&str>,
    field: &'static str,
) -> Result<Option<Identifier>, SimulationOutcomeError> {
    match raw {
        Some(text) => strict_non_empty_id(text, field).map(Some),
        None => Ok(None),
    }
}

fn validate_strategy_ids(raw: &[String]) -> Result<Vec<StrategyId>, SimulationOutcomeError> {
    if raw.is_empty() {
        return Err(SimulationOutcomeError::InvalidStrategies);
    }
    let mut seen = BTreeSet::new();
    let mut ids = Vec::with_capacity(raw.len());
    for entry in raw {
        if entry.trim().is_empty() {
            return Err(SimulationOutcomeError::InvalidStrategies);
        }
        if entry != entry.trim() {
            return Err(SimulationOutcomeError::UntrimmedIdentifier("strategy_ids"));
        }
        let id = StrategyId::new(entry).map_err(|_| SimulationOutcomeError::InvalidStrategies)?;
        if !seen.insert(id.as_str().to_owned()) {
            return Err(SimulationOutcomeError::InvalidStrategies);
        }
        ids.push(id);
    }
    Ok(ids)
}

fn normalize_signed_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn normalize_nonnegative_finite(value: f64) -> Option<f64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    Some(normalize_signed_zero(value))
}

type PairedOutcomes = (Vec<Identifier>, Vec<Vec<f64>>, Vec<f64>);

struct DrawAccumulator {
    key: Identifier,
    weight: f64,
    values: Vec<Option<f64>>,
}

fn pair_outcomes(
    outcomes: &[OutcomeRecordInput],
    expected_namespace: &Identifier,
    strategy_ids: &[StrategyId],
    strategy_index: &BTreeMap<&str, usize>,
    profile_declaration: &ValueDeclaration,
    weight_semantics: WeightSemantics,
) -> Result<PairedOutcomes, SimulationOutcomeError> {
    let mut draws: BTreeMap<String, DrawAccumulator> = BTreeMap::new();

    for record in outcomes {
        let record_ns = strict_non_empty_id(&record.draw_namespace, "draw_namespace")?;
        if &record_ns != expected_namespace {
            return Err(SimulationOutcomeError::MismatchedDrawNamespace {
                expected: expected_namespace.as_str().to_owned(),
                actual: record_ns.as_str().to_owned(),
            });
        }
        let draw_key = strict_non_empty_id(&record.draw_key, "draw_key")?;
        if record.strategy_id.trim().is_empty() {
            return Err(SimulationOutcomeError::EmptyIdentifier("strategy_id"));
        }
        if record.strategy_id != record.strategy_id.trim() {
            return Err(SimulationOutcomeError::UntrimmedIdentifier("strategy_id"));
        }
        let strategy = StrategyId::new(&record.strategy_id)
            .map_err(|_| SimulationOutcomeError::EmptyIdentifier("strategy_id"))?;
        let Some(&col_idx) = strategy_index.get(strategy.as_str()) else {
            return Err(SimulationOutcomeError::UnknownStrategy(
                strategy.as_str().to_owned(),
            ));
        };
        if !record.value.is_finite() {
            return Err(SimulationOutcomeError::NonFiniteValue {
                draw_key: draw_key.as_str().to_owned(),
                strategy_id: strategy.as_str().to_owned(),
            });
        }
        let weight = normalize_nonnegative_finite(record.weight)
            .ok_or_else(|| SimulationOutcomeError::InvalidWeight(draw_key.as_str().to_owned()))?;

        if let Some(record_decl_input) = &record.declaration {
            let record_decl = ValueDeclaration::from_input(record_decl_input)?;
            profile_declaration.validate_compatible(&record_decl)?;
        }

        let entry = draws
            .entry(draw_key.as_str().to_owned())
            .or_insert_with(|| DrawAccumulator {
                key: draw_key.clone(),
                weight,
                values: vec![None; strategy_ids.len()],
            });
        if entry.weight.to_bits() != weight.to_bits() {
            return Err(SimulationOutcomeError::ConflictingDrawWeight(
                draw_key.as_str().to_owned(),
            ));
        }
        if entry.values[col_idx].is_some() {
            return Err(SimulationOutcomeError::DuplicateOutcomeRecord {
                draw_key: draw_key.as_str().to_owned(),
                strategy_id: strategy.as_str().to_owned(),
            });
        }
        entry.values[col_idx] = Some(normalize_signed_zero(record.value));
    }

    let mut draw_keys = Vec::with_capacity(draws.len());
    let mut payoff_matrix = Vec::with_capacity(draws.len());
    let mut draw_weights = Vec::with_capacity(draws.len());

    for acc in draws.into_values() {
        let mut row = Vec::with_capacity(strategy_ids.len());
        for (idx, cell) in acc.values.into_iter().enumerate() {
            let Some(val) = cell else {
                return Err(SimulationOutcomeError::IncompleteDraw {
                    draw_key: acc.key.as_str().to_owned(),
                    missing_strategy: strategy_ids[idx].as_str().to_owned(),
                });
            };
            row.push(val);
        }
        draw_keys.push(acc.key);
        payoff_matrix.push(row);
        draw_weights.push(acc.weight);
    }

    validate_draw_weights(&draw_weights, weight_semantics)?;
    Ok((draw_keys, payoff_matrix, draw_weights))
}

fn validate_draw_weights(
    weights: &[f64],
    semantics: WeightSemantics,
) -> Result<(), SimulationOutcomeError> {
    if weights.is_empty() {
        return Err(SimulationOutcomeError::EmptyOutcomes);
    }
    let total: f64 = weights.iter().sum();
    match semantics {
        WeightSemantics::Equal => {
            let first = weights[0];
            if first <= 0.0
                || weights
                    .iter()
                    .any(|&w| w <= 0.0 || (w - first).abs() > 1e-12 * first)
            {
                return Err(SimulationOutcomeError::IncompatibleWeightSemantics(
                    "equal weight semantics requires identical positive weights for every draw"
                        .to_owned(),
                ));
            }
        }
        WeightSemantics::Probability => {
            if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
                return Err(SimulationOutcomeError::IncompatibleWeightSemantics(
                    format!("probability weights must sum to 1.0 (got {total})"),
                ));
            }
        }
        WeightSemantics::RelativeWeight => {
            if !total.is_finite() || total <= 0.0 {
                return Err(SimulationOutcomeError::IncompatibleWeightSemantics(
                    "relative weights must have a finite positive sum".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn align_parameter_samples(
    samples: &[ParameterSampleInput],
    expected_namespace: &Identifier,
    expected_draw_keys: &[Identifier],
) -> Result<AlignedParameterSamples, SimulationOutcomeError> {
    let expected_draws: BTreeSet<&str> =
        expected_draw_keys.iter().map(Identifier::as_str).collect();
    let mut by_draw: BTreeMap<String, BTreeMap<Identifier, f64>> = BTreeMap::new();
    let mut expected_param_names: Option<Vec<Identifier>> = None;

    for sample in samples {
        let sample_ns = strict_non_empty_id(&sample.draw_namespace, "draw_namespace")?;
        if &sample_ns != expected_namespace {
            return Err(SimulationOutcomeError::MismatchedDrawNamespace {
                expected: expected_namespace.as_str().to_owned(),
                actual: sample_ns.as_str().to_owned(),
            });
        }
        let draw_key = strict_non_empty_id(&sample.draw_key, "draw_key")?;
        if by_draw.contains_key(draw_key.as_str()) {
            return Err(SimulationOutcomeError::DuplicateParameterDraw(
                draw_key.as_str().to_owned(),
            ));
        }
        if !expected_draws.contains(draw_key.as_str()) {
            return Err(SimulationOutcomeError::UnmatchedParameterDraw(
                draw_key.as_str().to_owned(),
            ));
        }
        if sample.values.is_empty() {
            return Err(SimulationOutcomeError::EmptyParameterValues(
                draw_key.as_str().to_owned(),
            ));
        }

        let mut normalized_values = BTreeMap::new();
        for (raw_name, &value) in &sample.values {
            let param_id = strict_non_empty_id(raw_name, "parameter_id")?;
            if !value.is_finite() {
                return Err(SimulationOutcomeError::NonFiniteParameterValue {
                    draw_key: draw_key.as_str().to_owned(),
                    parameter_id: param_id.as_str().to_owned(),
                });
            }
            normalized_values.insert(param_id, normalize_signed_zero(value));
        }

        let names: Vec<Identifier> = normalized_values.keys().cloned().collect();
        match &expected_param_names {
            Some(expected) if expected != &names => {
                return Err(SimulationOutcomeError::InconsistentParameterSchema(
                    draw_key.as_str().to_owned(),
                ));
            }
            None => expected_param_names = Some(names),
            Some(_) => {}
        }

        by_draw.insert(draw_key.as_str().to_owned(), normalized_values);
    }

    let mut matrix = Vec::with_capacity(expected_draw_keys.len());
    let param_names = expected_param_names.unwrap_or_default();
    for draw_key in expected_draw_keys {
        let Some(row_map) = by_draw.remove(draw_key.as_str()) else {
            return Err(SimulationOutcomeError::MissingParameterDraw(
                draw_key.as_str().to_owned(),
            ));
        };
        let row = param_names.iter().map(|param| row_map[param]).collect();
        matrix.push(row);
    }

    Ok(AlignedParameterSamples {
        parameter_names: param_names,
        draw_keys: expected_draw_keys.to_vec(),
        matrix,
    })
}
