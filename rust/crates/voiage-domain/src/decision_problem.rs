//! Validated decision-problem and reproducibility aggregates.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use serde::{de, Deserialize, Deserializer, Serialize};

use crate::{Identifier, Seed, Threshold};

/// Validation failures for decision-problem aggregates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionProblemError {
    /// The currency code or symbol is shorter than the canonical schema permits.
    CurrencyTooShort,
    /// Outcome names were supplied as an empty collection.
    EmptyOutcomeNames,
    /// More than one outcome uses the same name.
    DuplicateOutcomeName,
    /// A decision problem contains no interventions.
    EmptyInterventions,
    /// More than one intervention uses the same stable identifier.
    DuplicateInterventionId,
    /// Fixture execution was declared without a fixture identity.
    MissingFixtureIdentity,
    /// Stochastic execution was declared without a seed.
    MissingStochasticSeed,
}

impl fmt::Display for DecisionProblemError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::CurrencyTooShort => "currency must contain at least three characters",
            Self::EmptyOutcomeNames => "supplied outcome names must not be empty",
            Self::DuplicateOutcomeName => "outcome names must be unique",
            Self::EmptyInterventions => "at least one intervention is required",
            Self::DuplicateInterventionId => "intervention identifiers must be unique",
            Self::MissingFixtureIdentity => {
                "deterministic fixture mode requires a provenance fixture identifier"
            }
            Self::MissingStochasticSeed => "stochastic execution requires a seed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DecisionProblemError {}

/// The normative v1 decision-analysis frame.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AnalysisType {
    /// Net benefit is the primary comparison representation.
    #[serde(rename = "net-benefit-first")]
    NetBenefitFirst,
}

/// One candidate intervention in a decision problem.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Intervention {
    #[serde(rename = "intervention_id")]
    id: Identifier,
    name: Identifier,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    is_reference: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    category: Option<String>,
}

impl Intervention {
    /// Creates an intervention from validated identity fields.
    #[must_use]
    pub const fn new(
        intervention_id: Identifier,
        name: Identifier,
        description: Option<String>,
        is_reference: bool,
        category: Option<String>,
    ) -> Self {
        Self {
            id: intervention_id,
            name,
            description,
            is_reference,
            category,
        }
    }

    /// Returns the stable intervention identifier.
    #[must_use]
    pub const fn intervention_id(&self) -> &Identifier {
        &self.id
    }

    /// Returns the human-readable intervention name.
    #[must_use]
    pub const fn name(&self) -> &Identifier {
        &self.name
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIntervention {
    intervention_id: Identifier,
    name: Identifier,
    description: Option<String>,
    #[serde(default)]
    is_reference: bool,
    category: Option<String>,
}

impl<'de> Deserialize<'de> for Intervention {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawIntervention::deserialize(deserializer)?;
        Ok(Self::new(
            raw.intervention_id,
            raw.name,
            raw.description,
            raw.is_reference,
            raw.category,
        ))
    }
}

/// A complete v1 decision problem with a non-empty intervention set.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionProblem {
    #[serde(rename = "decision_problem_id")]
    id: Identifier,
    title: Identifier,
    analysis_type: AnalysisType,
    currency: Identifier,
    willingness_to_pay: Threshold,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome_names: Option<Vec<Identifier>>,
    interventions: Vec<Intervention>,
}

impl DecisionProblem {
    /// Creates a validated decision problem.
    ///
    /// # Errors
    ///
    /// Returns an error when currency, outcome-name, or intervention invariants
    /// from the canonical v1 schema are violated.
    pub fn new(
        decision_problem_id: Identifier,
        title: Identifier,
        currency: Identifier,
        willingness_to_pay: Threshold,
        outcome_names: Option<Vec<Identifier>>,
        interventions: Vec<Intervention>,
    ) -> Result<Self, DecisionProblemError> {
        if currency.as_str().chars().count() < 3 {
            return Err(DecisionProblemError::CurrencyTooShort);
        }
        if let Some(outcomes) = outcome_names.as_ref() {
            if outcomes.is_empty() {
                return Err(DecisionProblemError::EmptyOutcomeNames);
            }
            let mut names = HashSet::with_capacity(outcomes.len());
            if outcomes
                .iter()
                .any(|outcome| !names.insert(outcome.as_str()))
            {
                return Err(DecisionProblemError::DuplicateOutcomeName);
            }
        }
        if interventions.is_empty() {
            return Err(DecisionProblemError::EmptyInterventions);
        }
        let mut identifiers = HashSet::with_capacity(interventions.len());
        if interventions
            .iter()
            .any(|item| !identifiers.insert(item.intervention_id().as_str()))
        {
            return Err(DecisionProblemError::DuplicateInterventionId);
        }
        Ok(Self {
            id: decision_problem_id,
            title,
            analysis_type: AnalysisType::NetBenefitFirst,
            currency,
            willingness_to_pay,
            outcome_names,
            interventions,
        })
    }

    /// Returns the stable decision-problem identifier.
    #[must_use]
    pub const fn decision_problem_id(&self) -> &Identifier {
        &self.id
    }

    /// Returns the positive willingness-to-pay threshold.
    #[must_use]
    pub const fn willingness_to_pay(&self) -> Threshold {
        self.willingness_to_pay
    }

    /// Borrows interventions in their declared order.
    #[must_use]
    pub fn interventions(&self) -> &[Intervention] {
        &self.interventions
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDecisionProblem {
    decision_problem_id: Identifier,
    title: Identifier,
    analysis_type: AnalysisType,
    currency: Identifier,
    willingness_to_pay: Threshold,
    outcome_names: Option<Vec<Identifier>>,
    interventions: Vec<Intervention>,
}

impl<'de> Deserialize<'de> for DecisionProblem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawDecisionProblem::deserialize(deserializer)?;
        if raw.analysis_type != AnalysisType::NetBenefitFirst {
            return Err(de::Error::custom("unsupported analysis type"));
        }
        Self::new(
            raw.decision_problem_id,
            raw.title,
            raw.currency,
            raw.willingness_to_pay,
            raw.outcome_names,
            raw.interventions,
        )
        .map_err(de::Error::custom)
    }
}

/// Whether an execution path is deterministic or stochastic.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Repeated execution has no random component.
    Deterministic,
    /// Execution depends on a declared random seed.
    Stochastic,
}

/// Typed provenance required for stable result families.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    voiage_version: Identifier,
    core_version: Identifier,
    method: Identifier,
    settings: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fixture_id: Option<Identifier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<Identifier>,
}

impl Provenance {
    /// Creates structured provenance with validated identity fields.
    #[must_use]
    pub const fn new(
        voiage_version: Identifier,
        core_version: Identifier,
        method: Identifier,
        settings: BTreeMap<String, String>,
        fixture_id: Option<Identifier>,
        backend: Option<Identifier>,
    ) -> Self {
        Self {
            voiage_version,
            core_version,
            method,
            settings,
            fixture_id,
            backend,
        }
    }

    /// Returns the committed fixture identity, when applicable.
    #[must_use]
    pub const fn fixture_id(&self) -> Option<&Identifier> {
        self.fixture_id.as_ref()
    }
}

/// Replay metadata for deterministic and stochastic result generation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReproducibilityMetadata {
    seed: Option<Seed>,
    execution_mode: ExecutionMode,
    deterministic_fixture_mode: bool,
    provenance: Provenance,
}

impl ReproducibilityMetadata {
    /// Creates replay metadata while enforcing mode-specific requirements.
    ///
    /// # Errors
    ///
    /// Fixture mode requires a fixture identity and stochastic mode requires
    /// a seed.
    pub fn new(
        seed: Option<Seed>,
        execution_mode: ExecutionMode,
        deterministic_fixture_mode: bool,
        provenance: Provenance,
    ) -> Result<Self, DecisionProblemError> {
        if deterministic_fixture_mode && provenance.fixture_id().is_none() {
            return Err(DecisionProblemError::MissingFixtureIdentity);
        }
        if execution_mode == ExecutionMode::Stochastic && seed.is_none() {
            return Err(DecisionProblemError::MissingStochasticSeed);
        }
        Ok(Self {
            seed,
            execution_mode,
            deterministic_fixture_mode,
            provenance,
        })
    }

    /// Returns the primary random seed when one was used.
    #[must_use]
    pub const fn seed(&self) -> Option<Seed> {
        self.seed
    }

    /// Returns whether this execution uses a committed deterministic fixture.
    #[must_use]
    pub const fn deterministic_fixture_mode(&self) -> bool {
        self.deterministic_fixture_mode
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReproducibilityMetadata {
    seed: Option<Seed>,
    execution_mode: ExecutionMode,
    deterministic_fixture_mode: bool,
    provenance: Provenance,
}

impl<'de> Deserialize<'de> for ReproducibilityMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawReproducibilityMetadata::deserialize(deserializer)?;
        Self::new(
            raw.seed,
            raw.execution_mode,
            raw.deterministic_fixture_mode,
            raw.provenance,
        )
        .map_err(de::Error::custom)
    }
}
