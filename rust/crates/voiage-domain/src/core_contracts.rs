use std::collections::{BTreeMap, HashSet};
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{Identifier, Probability, SampleMatrix, SampleVector, StrategyCollection};

/// Validation failures for stable aggregate domain contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreContractError {
    /// A required collection was empty.
    Empty(&'static str),
    /// A required label was blank after normalization.
    Blank {
        /// Name of the invalid field.
        field: &'static str,
        /// Position of the invalid value.
        index: usize,
    },
    /// A normalized identifier or label occurred more than once.
    Duplicate {
        /// Name of the invalid field.
        field: &'static str,
        /// Position of the repeated value.
        index: usize,
    },
    /// An observed dimension did not match its declared or related dimension.
    Dimension {
        /// Name of the invalid field.
        field: &'static str,
        /// Required dimension length.
        expected: usize,
        /// Observed dimension length.
        actual: usize,
    },
    /// A sample size was zero.
    ZeroSampleSize,
    /// A nested validated primitive rejected its input.
    Invalid(&'static str),
}

impl fmt::Display for CoreContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid stable domain contract: {self:?}")
    }
}

impl std::error::Error for CoreContractError {}

/// A validated sample-by-strategy net-benefit array.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ValueArray {
    #[serde(rename = "value_array_id")]
    id: Identifier,
    sample_count: u64,
    strategy_names: StrategyCollection,
    net_benefit: SampleMatrix,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ValueArrayWire {
    value_array_id: String,
    sample_count: u64,
    strategy_names: Vec<String>,
    net_benefit: Vec<Vec<f64>>,
}

impl ValueArray {
    /// Constructs a value array and derives its sample count from the matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid IDs, strategies, samples, or dimensions.
    pub fn new(
        value_array_id: impl AsRef<str>,
        strategy_names: Vec<String>,
        net_benefit: Vec<Vec<f64>>,
    ) -> Result<Self, CoreContractError> {
        let value_array_id = Identifier::new(value_array_id)
            .map_err(|_| CoreContractError::Invalid("value_array_id"))?;
        let strategy_names = StrategyCollection::try_from(strategy_names)
            .map_err(|_| CoreContractError::Invalid("strategy_names"))?;
        let net_benefit = SampleMatrix::try_from(net_benefit)
            .map_err(|_| CoreContractError::Invalid("net_benefit"))?;
        let [sample_count, strategy_count] = net_benefit.shape();
        if strategy_count != strategy_names.len() {
            return Err(CoreContractError::Dimension {
                field: "net_benefit",
                expected: strategy_names.len(),
                actual: strategy_count,
            });
        }
        Ok(Self {
            id: value_array_id,
            sample_count: u64::try_from(sample_count)
                .map_err(|_| CoreContractError::Invalid("sample_count"))?,
            strategy_names,
            net_benefit,
        })
    }

    /// Returns the stable value-array identifier.
    #[must_use]
    pub fn id(&self) -> &Identifier {
        &self.id
    }
    /// Returns the number of represented samples.
    #[must_use]
    pub const fn sample_count(&self) -> u64 {
        self.sample_count
    }
    /// Borrows the ordered strategies.
    #[must_use]
    pub const fn strategies(&self) -> &StrategyCollection {
        &self.strategy_names
    }
    /// Borrows the sample-by-strategy matrix.
    #[must_use]
    pub const fn net_benefit(&self) -> &SampleMatrix {
        &self.net_benefit
    }
}

impl<'de> Deserialize<'de> for ValueArray {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ValueArrayWire::deserialize(deserializer)?;
        let value = Self::new(wire.value_array_id, wire.strategy_names, wire.net_benefit)
            .map_err(serde::de::Error::custom)?;
        if wire.sample_count != value.sample_count {
            return Err(serde::de::Error::custom(
                "sample_count does not match net_benefit",
            ));
        }
        Ok(value)
    }
}

/// A validated set of equal-length parameter sample vectors.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ParameterSet {
    #[serde(rename = "parameter_set_id")]
    id: Identifier,
    sample_count: u64,
    parameters: BTreeMap<String, SampleVector>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterSetWire {
    parameter_set_id: String,
    sample_count: u64,
    parameters: BTreeMap<String, Vec<f64>>,
}

impl ParameterSet {
    /// Constructs a parameter set from named sample vectors.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid IDs, names, samples, or dimensions.
    pub fn new(
        parameter_set_id: impl AsRef<str>,
        parameters: BTreeMap<String, Vec<f64>>,
    ) -> Result<Self, CoreContractError> {
        let parameter_set_id = Identifier::new(parameter_set_id)
            .map_err(|_| CoreContractError::Invalid("parameter_set_id"))?;
        if parameters.is_empty() {
            return Err(CoreContractError::Empty("parameters"));
        }
        let mut normalized = BTreeMap::new();
        let mut sample_count = None;
        for (index, (name, values)) in parameters.into_iter().enumerate() {
            let name = name.trim().to_owned();
            if name.is_empty() {
                return Err(CoreContractError::Blank {
                    field: "parameters",
                    index,
                });
            }
            let values = SampleVector::try_from(values)
                .map_err(|_| CoreContractError::Invalid("parameters"))?;
            if let Some(expected) = sample_count {
                if values.len() != expected {
                    return Err(CoreContractError::Dimension {
                        field: "parameters",
                        expected,
                        actual: values.len(),
                    });
                }
            } else {
                sample_count = Some(values.len());
            }
            if normalized.insert(name, values).is_some() {
                return Err(CoreContractError::Duplicate {
                    field: "parameters",
                    index,
                });
            }
        }
        let sample_count =
            u64::try_from(sample_count.ok_or(CoreContractError::Empty("parameters"))?)
                .map_err(|_| CoreContractError::Invalid("sample_count"))?;
        Ok(Self {
            id: parameter_set_id,
            sample_count,
            parameters: normalized,
        })
    }

    /// Returns the parameter-set identifier.
    #[must_use]
    pub fn id(&self) -> &Identifier {
        &self.id
    }
    /// Returns the common number of samples.
    #[must_use]
    pub const fn sample_count(&self) -> u64 {
        self.sample_count
    }
    /// Borrows one parameter vector.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&SampleVector> {
        self.parameters.get(name)
    }
}

impl<'de> Deserialize<'de> for ParameterSet {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ParameterSetWire::deserialize(deserializer)?;
        let value =
            Self::new(wire.parameter_set_id, wire.parameters).map_err(serde::de::Error::custom)?;
        if wire.sample_count != value.sample_count {
            return Err(serde::de::Error::custom(
                "sample_count does not match parameters",
            ));
        }
        Ok(value)
    }
}

/// One validated arm in a trial design.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TrialArm {
    arm_id: Identifier,
    name: String,
    sample_size: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialArmWire {
    arm_id: String,
    name: String,
    sample_size: u64,
}

impl TrialArm {
    /// Constructs a non-empty arm with at least one participant.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid ID, blank name, or zero sample size.
    pub fn new(
        arm_id: impl AsRef<str>,
        name: impl AsRef<str>,
        sample_size: u64,
    ) -> Result<Self, CoreContractError> {
        let arm_id = Identifier::new(arm_id).map_err(|_| CoreContractError::Invalid("arm_id"))?;
        let name = name.as_ref().trim().to_owned();
        if name.is_empty() {
            return Err(CoreContractError::Blank {
                field: "name",
                index: 0,
            });
        }
        if sample_size == 0 {
            return Err(CoreContractError::ZeroSampleSize);
        }
        Ok(Self {
            arm_id,
            name,
            sample_size,
        })
    }
    /// Returns the arm identifier.
    #[must_use]
    pub fn id(&self) -> &Identifier {
        &self.arm_id
    }
    /// Returns the normalized display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns the participant count.
    #[must_use]
    pub const fn sample_size(&self) -> u64 {
        self.sample_size
    }
}

impl<'de> Deserialize<'de> for TrialArm {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = TrialArmWire::deserialize(deserializer)?;
        Self::new(wire.arm_id, wire.name, wire.sample_size).map_err(serde::de::Error::custom)
    }
}

/// A validated ordered trial design with unique arm IDs and names.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TrialDesign {
    trial_design_id: Identifier,
    arms: Vec<TrialArm>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialDesignWire {
    trial_design_id: String,
    arms: Vec<TrialArm>,
}

impl TrialDesign {
    /// Constructs a trial design.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid ID, no arms, or duplicate arm IDs or names.
    pub fn new(
        trial_design_id: impl AsRef<str>,
        arms: Vec<TrialArm>,
    ) -> Result<Self, CoreContractError> {
        let trial_design_id = Identifier::new(trial_design_id)
            .map_err(|_| CoreContractError::Invalid("trial_design_id"))?;
        if arms.is_empty() {
            return Err(CoreContractError::Empty("arms"));
        }
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        for (index, arm) in arms.iter().enumerate() {
            if !ids.insert(arm.id().as_str()) {
                return Err(CoreContractError::Duplicate {
                    field: "arm_id",
                    index,
                });
            }
            if !names.insert(arm.name()) {
                return Err(CoreContractError::Duplicate {
                    field: "name",
                    index,
                });
            }
        }
        Ok(Self {
            trial_design_id,
            arms,
        })
    }
    /// Returns the design identifier.
    #[must_use]
    pub fn id(&self) -> &Identifier {
        &self.trial_design_id
    }
    /// Borrows the ordered trial arms.
    #[must_use]
    pub fn arms(&self) -> &[TrialArm] {
        &self.arms
    }
}

impl<'de> Deserialize<'de> for TrialDesign {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = TrialDesignWire::deserialize(deserializer)?;
        Self::new(wire.trial_design_id, wire.arms).map_err(serde::de::Error::custom)
    }
}

/// A validated cost-effectiveness acceptability curve result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CeacResult {
    analysis_id: Identifier,
    decision_problem_id: Identifier,
    analysis_type: CeacDiscriminator,
    strategy_names: StrategyCollection,
    willingness_to_pay_values: Vec<f64>,
    cost_effectiveness_probabilities: Vec<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<BTreeMap<String, DiagnosticValue>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
enum DiagnosticValue {
    Null,
    Bool(bool),
    Integer(i64),
    Unsigned(u64),
    Number(f64),
    String(String),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum CeacDiscriminator {
    #[serde(rename = "ceac")]
    Ceac,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CeacWire {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: CeacDiscriminator,
    strategy_names: Vec<String>,
    willingness_to_pay_values: Vec<f64>,
    cost_effectiveness_probabilities: Vec<Vec<f64>>,
    method: Option<String>,
    diagnostics: Option<BTreeMap<String, DiagnosticValue>>,
}

impl CeacResult {
    /// Constructs an aligned strategy-by-threshold CEAC result.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identifiers, labels, finite values,
    /// probabilities, or dimensions.
    pub fn new(
        analysis_id: impl AsRef<str>,
        decision_problem_id: impl AsRef<str>,
        strategy_names: Vec<String>,
        willingness_to_pay_values: Vec<f64>,
        cost_effectiveness_probabilities: Vec<Vec<f64>>,
        method: Option<String>,
    ) -> Result<Self, CoreContractError> {
        Self::new_with_diagnostics(
            analysis_id,
            decision_problem_id,
            strategy_names,
            willingness_to_pay_values,
            cost_effectiveness_probabilities,
            method,
            None,
        )
    }

    fn new_with_diagnostics(
        analysis_id: impl AsRef<str>,
        decision_problem_id: impl AsRef<str>,
        strategy_names: Vec<String>,
        willingness_to_pay_values: Vec<f64>,
        cost_effectiveness_probabilities: Vec<Vec<f64>>,
        method: Option<String>,
        diagnostics: Option<BTreeMap<String, DiagnosticValue>>,
    ) -> Result<Self, CoreContractError> {
        let analysis_id =
            Identifier::new(analysis_id).map_err(|_| CoreContractError::Invalid("analysis_id"))?;
        let decision_problem_id = Identifier::new(decision_problem_id)
            .map_err(|_| CoreContractError::Invalid("decision_problem_id"))?;
        let strategy_names = StrategyCollection::try_from(strategy_names)
            .map_err(|_| CoreContractError::Invalid("strategy_names"))?;
        if willingness_to_pay_values.is_empty() {
            return Err(CoreContractError::Empty("willingness_to_pay_values"));
        }
        if willingness_to_pay_values
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(CoreContractError::Invalid("willingness_to_pay_values"));
        }
        if cost_effectiveness_probabilities.len() != strategy_names.len() {
            return Err(CoreContractError::Dimension {
                field: "cost_effectiveness_probabilities",
                expected: strategy_names.len(),
                actual: cost_effectiveness_probabilities.len(),
            });
        }
        for row in &cost_effectiveness_probabilities {
            if row.len() != willingness_to_pay_values.len() {
                return Err(CoreContractError::Dimension {
                    field: "cost_effectiveness_probabilities",
                    expected: willingness_to_pay_values.len(),
                    actual: row.len(),
                });
            }
            for value in row {
                Probability::new(*value)
                    .map_err(|_| CoreContractError::Invalid("cost_effectiveness_probabilities"))?;
            }
        }
        let method = method.map(|value| value.trim().to_owned());
        if method.as_deref() == Some("") {
            return Err(CoreContractError::Blank {
                field: "method",
                index: 0,
            });
        }
        Ok(Self {
            analysis_id,
            decision_problem_id,
            analysis_type: CeacDiscriminator::Ceac,
            strategy_names,
            willingness_to_pay_values,
            cost_effectiveness_probabilities,
            method,
            diagnostics,
        })
    }
    /// Returns the number of thresholds.
    #[must_use]
    pub fn threshold_count(&self) -> usize {
        self.willingness_to_pay_values.len()
    }
    /// Returns the number of strategies.
    #[must_use]
    pub fn strategy_count(&self) -> usize {
        self.strategy_names.len()
    }
}

impl<'de> Deserialize<'de> for CeacResult {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CeacWire::deserialize(deserializer)?;
        let _ = wire.analysis_type;
        Self::new_with_diagnostics(
            wire.analysis_id,
            wire.decision_problem_id,
            wire.strategy_names,
            wire.willingness_to_pay_values,
            wire.cost_effectiveness_probabilities,
            wire.method,
            wire.diagnostics,
        )
        .map_err(serde::de::Error::custom)
    }
}
