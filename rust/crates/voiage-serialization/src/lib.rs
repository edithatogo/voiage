//! Canonical, flat v1 wire contracts for stable voiage results.

#![forbid(unsafe_code)]

use core::{cmp::Ordering, fmt};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use std::collections::HashSet;

mod error_mapping;

/// Identifies this crate.
pub const CRATE_NAME: &str = "voiage-serialization";

/// A validation failure in a stable result payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError(&'static str);

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for ValidationError {}

fn text(value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError("text fields must not be blank"))
    } else {
        Ok(())
    }
}

fn texts(values: &[String]) -> Result<(), ValidationError> {
    if values.is_empty() {
        return Err(ValidationError("text arrays must not be empty"));
    }
    for value in values {
        text(value)?;
    }
    Ok(())
}

fn finite(value: f64) -> Result<(), ValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ValidationError("numeric values must be finite"))
    }
}

fn finite_values(values: &[f64]) -> Result<(), ValidationError> {
    if values.is_empty() {
        return Err(ValidationError("numeric arrays must not be empty"));
    }
    for &value in values {
        finite(value)?;
    }
    Ok(())
}

fn nonnegative(value: f64) -> Result<(), ValidationError> {
    finite(value)?;
    if value < 0.0 {
        Err(ValidationError("value must be nonnegative"))
    } else {
        Ok(())
    }
}

fn ids(analysis_id: &str, decision_problem_id: &str) -> Result<(), ValidationError> {
    text(analysis_id)?;
    text(decision_problem_id)
}

macro_rules! deserialize_validated {
    ($name:ty, $raw:ty) => {
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                <$raw>::deserialize(deserializer)?
                    .try_into()
                    .map_err(de::Error::custom)
            }
        }
    };
}

/// Flat EVPI v1 result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvpiResultV1 {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: EvpiType,
    willingness_to_pay: f64,
    expected_current_value: f64,
    expected_perfect_information: f64,
    evpi: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    strategy_names: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_net_benefit_by_strategy: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<String>,
}

/// Validated construction input for an EVPI result produced by a Rust kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct EvpiResultV1Input {
    /// Stable analysis identifier.
    pub analysis_id: String,
    /// Stable decision-problem identifier.
    pub decision_problem_id: String,
    /// Positive willingness-to-pay threshold.
    pub willingness_to_pay: f64,
    /// Expected value under current information.
    pub expected_current_value: f64,
    /// Expected value under perfect information.
    pub expected_perfect_information: f64,
    /// Expected value of perfect information.
    pub evpi: f64,
    /// Optional ordered strategy names.
    pub strategy_names: Option<Vec<String>>,
    /// Optional expected net benefits aligned with strategy names.
    pub expected_net_benefit_by_strategy: Option<Vec<f64>>,
    /// Optional method label.
    pub method: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum EvpiType {
    Evpi,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvpiRaw {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: EvpiType,
    willingness_to_pay: f64,
    expected_current_value: f64,
    expected_perfect_information: f64,
    evpi: f64,
    strategy_names: Option<Vec<String>>,
    expected_net_benefit_by_strategy: Option<Vec<f64>>,
    method: Option<String>,
}

impl TryFrom<EvpiRaw> for EvpiResultV1 {
    type Error = ValidationError;
    fn try_from(raw: EvpiRaw) -> Result<Self, Self::Error> {
        ids(&raw.analysis_id, &raw.decision_problem_id)?;
        finite(raw.willingness_to_pay)?;
        if raw.willingness_to_pay <= 0.0 {
            return Err(ValidationError("willingness_to_pay must be positive"));
        }
        finite(raw.expected_current_value)?;
        finite(raw.expected_perfect_information)?;
        nonnegative(raw.evpi)?;
        match (&raw.strategy_names, &raw.expected_net_benefit_by_strategy) {
            (None, None) => {}
            (Some(names), Some(values)) => {
                texts(names)?;
                finite_values(values)?;
                if names.len() != values.len() {
                    return Err(ValidationError("strategy arrays must be aligned"));
                }
            }
            _ => return Err(ValidationError("strategy arrays must be supplied together")),
        }
        if let Some(method) = &raw.method {
            text(method)?;
        }
        Ok(Self {
            analysis_id: raw.analysis_id,
            decision_problem_id: raw.decision_problem_id,
            analysis_type: raw.analysis_type,
            willingness_to_pay: raw.willingness_to_pay,
            expected_current_value: raw.expected_current_value,
            expected_perfect_information: raw.expected_perfect_information,
            evpi: raw.evpi,
            strategy_names: raw.strategy_names,
            expected_net_benefit_by_strategy: raw.expected_net_benefit_by_strategy,
            method: raw.method,
        })
    }
}

impl TryFrom<EvpiResultV1Input> for EvpiResultV1 {
    type Error = ValidationError;

    fn try_from(input: EvpiResultV1Input) -> Result<Self, Self::Error> {
        Self::try_from(EvpiRaw {
            analysis_id: input.analysis_id,
            decision_problem_id: input.decision_problem_id,
            analysis_type: EvpiType::Evpi,
            willingness_to_pay: input.willingness_to_pay,
            expected_current_value: input.expected_current_value,
            expected_perfect_information: input.expected_perfect_information,
            evpi: input.evpi,
            strategy_names: input.strategy_names,
            expected_net_benefit_by_strategy: input.expected_net_benefit_by_strategy,
            method: input.method,
        })
    }
}
deserialize_validated!(EvpiResultV1, EvpiRaw);

macro_rules! scalar_result {
    ($name:ident, $input:ident, $raw:ident, $kind:ident, $literal:literal, $value:ident,
        $( $extra_name:ident : $extra_type:ty ),* $(,)?) => {
        #[doc = concat!("Flat ", $literal, " v1 result.")]
        #[derive(Clone, Debug, PartialEq, Serialize)]
        pub struct $name {
            analysis_id: String, decision_problem_id: String, analysis_type: $kind,
            $( $extra_name: $extra_type, )*
            $value: f64,
            #[serde(skip_serializing_if = "Option::is_none")] expected_current_value: Option<f64>,
            #[serde(skip_serializing_if = "Option::is_none")] expected_perfect_information: Option<f64>,
            #[serde(skip_serializing_if = "Option::is_none")] method: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")] diagnostics: Option<Map<String, Value>>,
        }
        #[doc = concat!("Validated construction input for a ", $literal, " result produced by a Rust kernel.")]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $input {
            /// Stable analysis identifier.
            pub analysis_id: String,
            /// Stable decision-problem identifier.
            pub decision_problem_id: String,
            $(
            /// Method-specific validated input.
            pub $extra_name: $extra_type,
            )*
            /// Calculated value for this result family.
            pub $value: f64,
            /// Optional expected value under current information.
            pub expected_current_value: Option<f64>,
            /// Optional expected value under perfect information.
            pub expected_perfect_information: Option<f64>,
            /// Optional method label.
            pub method: Option<String>,
            /// Optional structured diagnostics.
            pub diagnostics: Option<Map<String, Value>>,
        }
        #[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
        enum $kind {
            #[serde(rename = $literal)]
            Value,
        }
        #[derive(Deserialize)] #[serde(deny_unknown_fields)]
        struct $raw {
            analysis_id: String, decision_problem_id: String, analysis_type: $kind,
            $( $extra_name: $extra_type, )*
            $value: f64, expected_current_value: Option<f64>,
            expected_perfect_information: Option<f64>, method: Option<String>,
            diagnostics: Option<Map<String, Value>>,
        }
        deserialize_validated!($name, $raw);
    };
}

scalar_result!(EvppiResultV1, EvppiResultV1Input, EvppiRaw, EvppiType, "evppi", evppi,
    parameter_names: Vec<String>,);
scalar_result!(EvsiResultV1, EvsiResultV1Input, EvsiRaw, EvsiType, "evsi", evsi,
    trial_design_id: String, sample_size: u64, expected_sample_value: Option<f64>,);

/// Flat ENBS v1 result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnbsResultV1 {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: EnbsType,
    trial_design_id: String,
    sample_size: u64,
    design_cost: f64,
    enbs: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_sample_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_perfect_information: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<Map<String, Value>>,
}

/// Validated construction input for an ENBS result produced by a Rust kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct EnbsResultV1Input {
    /// Stable analysis identifier.
    pub analysis_id: String,
    /// Stable decision-problem identifier.
    pub decision_problem_id: String,
    /// Stable trial-design identifier.
    pub trial_design_id: String,
    /// Fixed-width total sample size.
    pub sample_size: u64,
    /// Cost of the proposed design.
    pub design_cost: f64,
    /// Expected net benefit of sampling.
    pub enbs: f64,
    /// Optional expected sample information value.
    pub expected_sample_value: Option<f64>,
    /// Optional expected perfect-information value.
    pub expected_perfect_information: Option<f64>,
    /// Optional method label.
    pub method: Option<String>,
    /// Optional structured diagnostics.
    pub diagnostics: Option<Map<String, Value>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
enum EnbsType {
    #[serde(rename = "enbs")]
    Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EnbsRaw {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: EnbsType,
    trial_design_id: String,
    sample_size: u64,
    design_cost: f64,
    enbs: f64,
    expected_sample_value: Option<f64>,
    expected_perfect_information: Option<f64>,
    method: Option<String>,
    diagnostics: Option<Map<String, Value>>,
}

deserialize_validated!(EnbsResultV1, EnbsRaw);

fn validate_optional_finite(values: &[Option<f64>]) -> Result<(), ValidationError> {
    for value in values.iter().flatten() {
        finite(*value)?;
    }
    Ok(())
}

impl TryFrom<EvppiRaw> for EvppiResultV1 {
    type Error = ValidationError;
    fn try_from(raw: EvppiRaw) -> Result<Self, Self::Error> {
        ids(&raw.analysis_id, &raw.decision_problem_id)?;
        texts(&raw.parameter_names)?;
        nonnegative(raw.evppi)?;
        validate_optional_finite(&[raw.expected_current_value, raw.expected_perfect_information])?;
        if let Some(method) = &raw.method {
            text(method)?;
        }
        Ok(Self {
            analysis_id: raw.analysis_id,
            decision_problem_id: raw.decision_problem_id,
            analysis_type: raw.analysis_type,
            parameter_names: raw.parameter_names,
            evppi: raw.evppi,
            expected_current_value: raw.expected_current_value,
            expected_perfect_information: raw.expected_perfect_information,
            method: raw.method,
            diagnostics: raw.diagnostics,
        })
    }
}

impl TryFrom<EvppiResultV1Input> for EvppiResultV1 {
    type Error = ValidationError;

    fn try_from(input: EvppiResultV1Input) -> Result<Self, Self::Error> {
        Self::try_from(EvppiRaw {
            analysis_id: input.analysis_id,
            decision_problem_id: input.decision_problem_id,
            analysis_type: EvppiType::Value,
            parameter_names: input.parameter_names,
            evppi: input.evppi,
            expected_current_value: input.expected_current_value,
            expected_perfect_information: input.expected_perfect_information,
            method: input.method,
            diagnostics: input.diagnostics,
        })
    }
}

impl TryFrom<EvsiRaw> for EvsiResultV1 {
    type Error = ValidationError;
    fn try_from(raw: EvsiRaw) -> Result<Self, Self::Error> {
        ids(&raw.analysis_id, &raw.decision_problem_id)?;
        text(&raw.trial_design_id)?;
        if raw.sample_size == 0 {
            return Err(ValidationError("sample_size must be positive"));
        }
        nonnegative(raw.evsi)?;
        validate_optional_finite(&[
            raw.expected_sample_value,
            raw.expected_current_value,
            raw.expected_perfect_information,
        ])?;
        if let Some(method) = &raw.method {
            text(method)?;
        }
        Ok(Self {
            analysis_id: raw.analysis_id,
            decision_problem_id: raw.decision_problem_id,
            analysis_type: raw.analysis_type,
            trial_design_id: raw.trial_design_id,
            sample_size: raw.sample_size,
            expected_sample_value: raw.expected_sample_value,
            evsi: raw.evsi,
            expected_current_value: raw.expected_current_value,
            expected_perfect_information: raw.expected_perfect_information,
            method: raw.method,
            diagnostics: raw.diagnostics,
        })
    }
}

impl TryFrom<EvsiResultV1Input> for EvsiResultV1 {
    type Error = ValidationError;

    fn try_from(input: EvsiResultV1Input) -> Result<Self, Self::Error> {
        Self::try_from(EvsiRaw {
            analysis_id: input.analysis_id,
            decision_problem_id: input.decision_problem_id,
            analysis_type: EvsiType::Value,
            trial_design_id: input.trial_design_id,
            sample_size: input.sample_size,
            expected_sample_value: input.expected_sample_value,
            evsi: input.evsi,
            expected_current_value: input.expected_current_value,
            expected_perfect_information: input.expected_perfect_information,
            method: input.method,
            diagnostics: input.diagnostics,
        })
    }
}

impl TryFrom<EnbsRaw> for EnbsResultV1 {
    type Error = ValidationError;
    fn try_from(raw: EnbsRaw) -> Result<Self, Self::Error> {
        ids(&raw.analysis_id, &raw.decision_problem_id)?;
        text(&raw.trial_design_id)?;
        if raw.sample_size == 0 {
            return Err(ValidationError("sample_size must be positive"));
        }
        nonnegative(raw.design_cost)?;
        finite(raw.enbs)?;
        validate_optional_finite(&[raw.expected_sample_value, raw.expected_perfect_information])?;
        if let Some(method) = &raw.method {
            text(method)?;
        }
        Ok(Self {
            analysis_id: raw.analysis_id,
            decision_problem_id: raw.decision_problem_id,
            analysis_type: raw.analysis_type,
            trial_design_id: raw.trial_design_id,
            sample_size: raw.sample_size,
            design_cost: raw.design_cost,
            enbs: raw.enbs,
            expected_sample_value: raw.expected_sample_value,
            expected_perfect_information: raw.expected_perfect_information,
            method: raw.method,
            diagnostics: raw.diagnostics,
        })
    }
}

impl TryFrom<EnbsResultV1Input> for EnbsResultV1 {
    type Error = ValidationError;

    fn try_from(input: EnbsResultV1Input) -> Result<Self, Self::Error> {
        Self::try_from(EnbsRaw {
            analysis_id: input.analysis_id,
            decision_problem_id: input.decision_problem_id,
            analysis_type: EnbsType::Value,
            trial_design_id: input.trial_design_id,
            sample_size: input.sample_size,
            design_cost: input.design_cost,
            enbs: input.enbs,
            expected_sample_value: input.expected_sample_value,
            expected_perfect_information: input.expected_perfect_information,
            method: input.method,
            diagnostics: input.diagnostics,
        })
    }
}

/// Flat expected opportunity-loss v1 result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExpectedLossResultV1 {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: ExpectedLossType,
    strategy_names: Vec<String>,
    expected_net_benefit_by_strategy: Vec<f64>,
    expected_opportunity_loss_by_strategy: Vec<f64>,
    optimal_strategy_index: u64,
    minimum_expected_opportunity_loss: f64,
    sample_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reporting: Option<Map<String, Value>>,
}

/// Validated construction input for an expected-loss result.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedLossResultV1Input {
    /// Stable analysis identifier.
    pub analysis_id: String,
    /// Stable decision-problem identifier.
    pub decision_problem_id: String,
    /// Ordered strategy names.
    pub strategy_names: Vec<String>,
    /// Mean net or generalized benefit aligned with strategy names.
    pub expected_net_benefit_by_strategy: Vec<f64>,
    /// Mean opportunity loss aligned with strategy names.
    pub expected_opportunity_loss_by_strategy: Vec<f64>,
    /// Lowest-index strategy with greatest expected benefit.
    pub optimal_strategy_index: u64,
    /// Expected opportunity loss of the selected strategy.
    pub minimum_expected_opportunity_loss: f64,
    /// Number of uncertainty samples.
    pub sample_count: u64,
    /// Optional estimator or method label.
    pub method: Option<String>,
    /// Optional reporting extensions.
    pub reporting: Option<Map<String, Value>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExpectedLossType {
    ExpectedLoss,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedLossRaw {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: ExpectedLossType,
    strategy_names: Vec<String>,
    expected_net_benefit_by_strategy: Vec<f64>,
    expected_opportunity_loss_by_strategy: Vec<f64>,
    optimal_strategy_index: u64,
    minimum_expected_opportunity_loss: f64,
    sample_count: u64,
    method: Option<String>,
    reporting: Option<Map<String, Value>>,
}

deserialize_validated!(ExpectedLossResultV1, ExpectedLossRaw);

impl TryFrom<ExpectedLossRaw> for ExpectedLossResultV1 {
    type Error = ValidationError;

    fn try_from(raw: ExpectedLossRaw) -> Result<Self, Self::Error> {
        ids(&raw.analysis_id, &raw.decision_problem_id)?;
        texts(&raw.strategy_names)?;
        finite_values(&raw.expected_net_benefit_by_strategy)?;
        finite_values(&raw.expected_opportunity_loss_by_strategy)?;
        if raw.strategy_names.len() != raw.expected_net_benefit_by_strategy.len()
            || raw.strategy_names.len() != raw.expected_opportunity_loss_by_strategy.len()
        {
            return Err(ValidationError(
                "expected-loss strategy arrays must be aligned",
            ));
        }
        for &loss in &raw.expected_opportunity_loss_by_strategy {
            nonnegative(loss)?;
        }
        nonnegative(raw.minimum_expected_opportunity_loss)?;
        let optimal_index = usize::try_from(raw.optimal_strategy_index)
            .map_err(|_| ValidationError("optimal strategy index is out of range"))?;
        if optimal_index >= raw.strategy_names.len() {
            return Err(ValidationError("optimal strategy index is out of range"));
        }
        let expected_optimal = raw
            .expected_net_benefit_by_strategy
            .iter()
            .enumerate()
            .fold(0, |best, (index, value)| {
                if *value > raw.expected_net_benefit_by_strategy[best] {
                    index
                } else {
                    best
                }
            });
        if optimal_index != expected_optimal {
            return Err(ValidationError(
                "optimal strategy index must select the first greatest expected benefit",
            ));
        }
        if raw
            .minimum_expected_opportunity_loss
            .partial_cmp(&raw.expected_opportunity_loss_by_strategy[optimal_index])
            != Some(Ordering::Equal)
        {
            return Err(ValidationError(
                "minimum expected opportunity loss must match the selected strategy",
            ));
        }
        if raw.sample_count == 0 {
            return Err(ValidationError("sample_count must be positive"));
        }
        if let Some(method) = &raw.method {
            text(method)?;
        }
        Ok(Self {
            analysis_id: raw.analysis_id,
            decision_problem_id: raw.decision_problem_id,
            analysis_type: raw.analysis_type,
            strategy_names: raw.strategy_names,
            expected_net_benefit_by_strategy: raw.expected_net_benefit_by_strategy,
            expected_opportunity_loss_by_strategy: raw.expected_opportunity_loss_by_strategy,
            optimal_strategy_index: raw.optimal_strategy_index,
            minimum_expected_opportunity_loss: raw.minimum_expected_opportunity_loss,
            sample_count: raw.sample_count,
            method: raw.method,
            reporting: raw.reporting,
        })
    }
}

impl TryFrom<ExpectedLossResultV1Input> for ExpectedLossResultV1 {
    type Error = ValidationError;

    fn try_from(input: ExpectedLossResultV1Input) -> Result<Self, Self::Error> {
        Self::try_from(ExpectedLossRaw {
            analysis_id: input.analysis_id,
            decision_problem_id: input.decision_problem_id,
            analysis_type: ExpectedLossType::ExpectedLoss,
            strategy_names: input.strategy_names,
            expected_net_benefit_by_strategy: input.expected_net_benefit_by_strategy,
            expected_opportunity_loss_by_strategy: input.expected_opportunity_loss_by_strategy,
            optimal_strategy_index: input.optimal_strategy_index,
            minimum_expected_opportunity_loss: input.minimum_expected_opportunity_loss,
            sample_count: input.sample_count,
            method: input.method,
            reporting: input.reporting,
        })
    }
}

/// Flat CEAF v1 result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CeafResultV1 {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: CeafType,
    wtp_thresholds: Vec<f64>,
    optimal_strategy_indices: Vec<u64>,
    optimal_strategy_names: Vec<String>,
    acceptability_probabilities: Vec<f64>,
    probability_lower: Vec<f64>,
    probability_upper: Vec<f64>,
    expected_net_benefit: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reporting: Option<Map<String, Value>>,
}
/// Validated construction input for a CEAF result produced by a Rust kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct CeafResultV1Input {
    /// Stable analysis identifier.
    pub analysis_id: String,
    /// Stable decision-problem identifier.
    pub decision_problem_id: String,
    /// Willingness-to-pay thresholds.
    pub wtp_thresholds: Vec<f64>,
    /// Optimal strategy index at each threshold.
    pub optimal_strategy_indices: Vec<u64>,
    /// Optimal strategy name at each threshold.
    pub optimal_strategy_names: Vec<String>,
    /// Point estimates for acceptability probability.
    pub acceptability_probabilities: Vec<f64>,
    /// Lower probability bounds.
    pub probability_lower: Vec<f64>,
    /// Upper probability bounds.
    pub probability_upper: Vec<f64>,
    /// Expected net benefit at each threshold.
    pub expected_net_benefit: Vec<f64>,
    /// Optional reporting extensions.
    pub reporting: Option<Map<String, Value>>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum CeafType {
    Ceaf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CeafRaw {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: CeafType,
    wtp_thresholds: Vec<f64>,
    optimal_strategy_indices: Vec<u64>,
    optimal_strategy_names: Vec<String>,
    acceptability_probabilities: Vec<f64>,
    probability_lower: Vec<f64>,
    probability_upper: Vec<f64>,
    expected_net_benefit: Vec<f64>,
    reporting: Option<Map<String, Value>>,
}
deserialize_validated!(CeafResultV1, CeafRaw);
impl TryFrom<CeafRaw> for CeafResultV1 {
    type Error = ValidationError;
    fn try_from(raw: CeafRaw) -> Result<Self, Self::Error> {
        ids(&raw.analysis_id, &raw.decision_problem_id)?;
        texts(&raw.optimal_strategy_names)?;
        finite_values(&raw.wtp_thresholds)?;
        finite_values(&raw.expected_net_benefit)?;
        let length = raw.wtp_thresholds.len();
        if [
            raw.optimal_strategy_indices.len(),
            raw.optimal_strategy_names.len(),
            raw.acceptability_probabilities.len(),
            raw.probability_lower.len(),
            raw.probability_upper.len(),
            raw.expected_net_benefit.len(),
        ]
        .iter()
        .any(|&n| n != length)
        {
            return Err(ValidationError("CEAF arrays must be aligned"));
        }
        for ((&lower, &estimate), &upper) in raw
            .probability_lower
            .iter()
            .zip(&raw.acceptability_probabilities)
            .zip(&raw.probability_upper)
        {
            nonnegative(lower)?;
            nonnegative(estimate)?;
            nonnegative(upper)?;
            if upper > 1.0 || estimate > 1.0 || lower > estimate || estimate > upper {
                return Err(ValidationError("CEAF probabilities or bounds are invalid"));
            }
        }
        Ok(Self {
            analysis_id: raw.analysis_id,
            decision_problem_id: raw.decision_problem_id,
            analysis_type: raw.analysis_type,
            wtp_thresholds: raw.wtp_thresholds,
            optimal_strategy_indices: raw.optimal_strategy_indices,
            optimal_strategy_names: raw.optimal_strategy_names,
            acceptability_probabilities: raw.acceptability_probabilities,
            probability_lower: raw.probability_lower,
            probability_upper: raw.probability_upper,
            expected_net_benefit: raw.expected_net_benefit,
            reporting: raw.reporting,
        })
    }
}
impl TryFrom<CeafResultV1Input> for CeafResultV1 {
    type Error = ValidationError;

    fn try_from(input: CeafResultV1Input) -> Result<Self, Self::Error> {
        Self::try_from(CeafRaw {
            analysis_id: input.analysis_id,
            decision_problem_id: input.decision_problem_id,
            analysis_type: CeafType::Ceaf,
            wtp_thresholds: input.wtp_thresholds,
            optimal_strategy_indices: input.optimal_strategy_indices,
            optimal_strategy_names: input.optimal_strategy_names,
            acceptability_probabilities: input.acceptability_probabilities,
            probability_lower: input.probability_lower,
            probability_upper: input.probability_upper,
            expected_net_benefit: input.expected_net_benefit,
            reporting: input.reporting,
        })
    }
}

/// Classification assigned to a strategy in a dominance result.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DominanceStatus {
    /// The strategy lies on the cost-effectiveness frontier.
    Frontier,
    /// Another strategy is no more costly and at least as effective.
    StronglyDominated,
    /// A convex combination of frontier strategies dominates this strategy.
    ExtendedDominated,
}

/// Flat dominance v1 result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DominanceResultV1 {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: DominanceType,
    strategy_names: Vec<String>,
    costs: Vec<f64>,
    effects: Vec<f64>,
    frontier_indices: Vec<u64>,
    strongly_dominated_indices: Vec<u64>,
    extended_dominated_indices: Vec<u64>,
    status: Vec<DominanceStatus>,
    incremental_costs: Vec<f64>,
    incremental_effects: Vec<f64>,
    icers: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reporting: Option<Map<String, Value>>,
}
/// Validated construction input for a dominance result produced by a Rust kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct DominanceResultV1Input {
    /// Stable analysis identifier.
    pub analysis_id: String,
    /// Stable decision-problem identifier.
    pub decision_problem_id: String,
    /// Ordered strategy names.
    pub strategy_names: Vec<String>,
    /// Strategy costs aligned with names.
    pub costs: Vec<f64>,
    /// Strategy effects aligned with names.
    pub effects: Vec<f64>,
    /// Indices on the efficiency frontier.
    pub frontier_indices: Vec<u64>,
    /// Strongly dominated strategy indices.
    pub strongly_dominated_indices: Vec<u64>,
    /// Extended-dominated strategy indices.
    pub extended_dominated_indices: Vec<u64>,
    /// Classification aligned with strategy names.
    pub status: Vec<DominanceStatus>,
    /// Incremental costs between frontier strategies.
    pub incremental_costs: Vec<f64>,
    /// Incremental effects between frontier strategies.
    pub incremental_effects: Vec<f64>,
    /// Incremental cost-effectiveness ratios.
    pub icers: Vec<f64>,
    /// Optional reporting extensions.
    pub reporting: Option<Map<String, Value>>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum DominanceType {
    Dominance,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DominanceRaw {
    analysis_id: String,
    decision_problem_id: String,
    analysis_type: DominanceType,
    strategy_names: Vec<String>,
    costs: Vec<f64>,
    effects: Vec<f64>,
    frontier_indices: Vec<u64>,
    strongly_dominated_indices: Vec<u64>,
    extended_dominated_indices: Vec<u64>,
    status: Vec<DominanceStatus>,
    incremental_costs: Vec<f64>,
    incremental_effects: Vec<f64>,
    icers: Vec<f64>,
    reporting: Option<Map<String, Value>>,
}
deserialize_validated!(DominanceResultV1, DominanceRaw);
impl TryFrom<DominanceRaw> for DominanceResultV1 {
    type Error = ValidationError;
    fn try_from(raw: DominanceRaw) -> Result<Self, Self::Error> {
        ids(&raw.analysis_id, &raw.decision_problem_id)?;
        texts(&raw.strategy_names)?;
        if raw.strategy_names.len() < 2 {
            return Err(ValidationError(
                "dominance requires at least two strategies",
            ));
        }
        finite_values(&raw.costs)?;
        finite_values(&raw.effects)?;
        let count = raw.strategy_names.len();
        if raw.costs.len() != count || raw.effects.len() != count || raw.status.len() != count {
            return Err(ValidationError("dominance strategy arrays must be aligned"));
        }
        let frontier_len = raw.frontier_indices.len();
        if frontier_len == 0 {
            return Err(ValidationError("frontier must not be empty"));
        }
        let transitions = frontier_len - 1;
        if raw.incremental_costs.len() != transitions
            || raw.incremental_effects.len() != transitions
            || raw.icers.len() != transitions
        {
            return Err(ValidationError(
                "incremental arrays must align with frontier transitions",
            ));
        }
        for values in [&raw.incremental_costs, &raw.incremental_effects, &raw.icers] {
            for &value in values {
                finite(value)?;
            }
        }
        let groups = [
            (&raw.frontier_indices, DominanceStatus::Frontier),
            (
                &raw.strongly_dominated_indices,
                DominanceStatus::StronglyDominated,
            ),
            (
                &raw.extended_dominated_indices,
                DominanceStatus::ExtendedDominated,
            ),
        ];
        let mut seen = HashSet::new();
        for (indices, expected) in groups {
            for &index in indices {
                let index = usize::try_from(index)
                    .map_err(|_| ValidationError("strategy index is out of range"))?;
                if index >= count || !seen.insert(index) {
                    return Err(ValidationError(
                        "dominance indices must be unique and in range",
                    ));
                }
                if raw.status[index] != expected {
                    return Err(ValidationError(
                        "dominance status disagrees with index classification",
                    ));
                }
            }
        }
        if seen.len() != count {
            return Err(ValidationError(
                "every strategy must have exactly one classification",
            ));
        }
        Ok(Self {
            analysis_id: raw.analysis_id,
            decision_problem_id: raw.decision_problem_id,
            analysis_type: raw.analysis_type,
            strategy_names: raw.strategy_names,
            costs: raw.costs,
            effects: raw.effects,
            frontier_indices: raw.frontier_indices,
            strongly_dominated_indices: raw.strongly_dominated_indices,
            extended_dominated_indices: raw.extended_dominated_indices,
            status: raw.status,
            incremental_costs: raw.incremental_costs,
            incremental_effects: raw.incremental_effects,
            icers: raw.icers,
            reporting: raw.reporting,
        })
    }
}
impl TryFrom<DominanceResultV1Input> for DominanceResultV1 {
    type Error = ValidationError;

    fn try_from(input: DominanceResultV1Input) -> Result<Self, Self::Error> {
        Self::try_from(DominanceRaw {
            analysis_id: input.analysis_id,
            decision_problem_id: input.decision_problem_id,
            analysis_type: DominanceType::Dominance,
            strategy_names: input.strategy_names,
            costs: input.costs,
            effects: input.effects,
            frontier_indices: input.frontier_indices,
            strongly_dominated_indices: input.strongly_dominated_indices,
            extended_dominated_indices: input.extended_dominated_indices,
            status: input.status,
            incremental_costs: input.incremental_costs,
            incremental_effects: input.incremental_effects,
            icers: input.icers,
            reporting: input.reporting,
        })
    }
}
