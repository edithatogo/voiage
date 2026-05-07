use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainError {
    EmptyField(&'static str),
    EmptyCollection(&'static str),
    RaggedMatrix(&'static str),
    SampleCountMismatch { expected: usize, actual: usize },
    WidthMismatch { expected: usize, actual: usize },
    DuplicateValue(&'static str),
    NonFinite(&'static str),
}

impl core::fmt::Display for DomainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyField(name) => write!(f, "{name} must be a non-empty string"),
            Self::EmptyCollection(name) => write!(f, "{name} must not be empty"),
            Self::RaggedMatrix(name) => write!(f, "{name} rows must have a consistent width"),
            Self::SampleCountMismatch { expected, actual } => {
                write!(
                    f,
                    "sample_count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::WidthMismatch { expected, actual } => {
                write!(f, "width mismatch: expected {expected}, got {actual}")
            }
            Self::DuplicateValue(name) => write!(f, "{name} entries must be unique"),
            Self::NonFinite(name) => write!(f, "{name} values must be finite"),
        }
    }
}

impl std::error::Error for DomainError {}

fn ensure_non_empty(value: &str, name: &'static str) -> Result<(), DomainError> {
    if value.trim().is_empty() {
        return Err(DomainError::EmptyField(name));
    }
    Ok(())
}

fn ensure_matrix_shape(matrix: &[Vec<f64>], name: &'static str) -> Result<usize, DomainError> {
    if matrix.is_empty() {
        return Err(DomainError::EmptyCollection(name));
    }
    let width = matrix[0].len();
    if width == 0 {
        return Err(DomainError::EmptyCollection(name));
    }
    for row in matrix {
        if row.len() != width {
            return Err(DomainError::RaggedMatrix(name));
        }
        for value in row {
            if !value.is_finite() {
                return Err(DomainError::NonFinite(name));
            }
        }
    }
    Ok(width)
}

fn ensure_vector_shape(values: &[f64], name: &'static str) -> Result<(), DomainError> {
    if values.is_empty() {
        return Err(DomainError::EmptyCollection(name));
    }
    for value in values {
        if !value.is_finite() {
            return Err(DomainError::NonFinite(name));
        }
    }
    Ok(())
}

fn normalize_string_map(
    map: &BTreeMap<String, String>,
    name: &'static str,
) -> Result<(), DomainError> {
    for key in map.keys() {
        ensure_non_empty(key, name)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueArray {
    pub value_array_id: String,
    pub sample_count: usize,
    pub strategy_names: Vec<String>,
    pub net_benefit: Vec<Vec<f64>>,
}

impl ValueArray {
    pub fn new(
        value_array_id: impl Into<String>,
        strategy_names: Vec<String>,
        net_benefit: Vec<Vec<f64>>,
    ) -> Result<Self, DomainError> {
        let value_array_id = value_array_id.into();
        ensure_non_empty(&value_array_id, "value_array_id")?;
        if strategy_names.is_empty() {
            return Err(DomainError::EmptyCollection("strategy_names"));
        }
        if strategy_names.iter().any(|item| item.trim().is_empty()) {
            return Err(DomainError::EmptyField("strategy_names"));
        }
        let width = ensure_matrix_shape(&net_benefit, "net_benefit")?;
        if width != strategy_names.len() {
            return Err(DomainError::WidthMismatch {
                expected: strategy_names.len(),
                actual: width,
            });
        }
        Ok(Self {
            value_array_id,
            sample_count: net_benefit.len(),
            strategy_names,
            net_benefit,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterSet {
    pub parameter_set_id: String,
    pub sample_count: usize,
    pub parameters: BTreeMap<String, Vec<f64>>,
}

impl ParameterSet {
    pub fn new(
        parameter_set_id: impl Into<String>,
        parameters: BTreeMap<String, Vec<f64>>,
    ) -> Result<Self, DomainError> {
        let parameter_set_id = parameter_set_id.into();
        ensure_non_empty(&parameter_set_id, "parameter_set_id")?;
        if parameters.is_empty() {
            return Err(DomainError::EmptyCollection("parameters"));
        }

        let mut expected = None;
        for (name, values) in &parameters {
            ensure_non_empty(name, "parameters")?;
            ensure_vector_shape(values, "parameters")?;
            match expected {
                Some(count) if count != values.len() => {
                    return Err(DomainError::WidthMismatch {
                        expected: count,
                        actual: values.len(),
                    });
                }
                None => expected = Some(values.len()),
                _ => {}
            }
        }

        Ok(Self {
            parameter_set_id,
            sample_count: expected.unwrap_or(0),
            parameters,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialArm {
    pub arm_id: String,
    pub name: String,
    pub sample_size: usize,
}

impl TrialArm {
    pub fn new(
        arm_id: impl Into<String>,
        name: impl Into<String>,
        sample_size: usize,
    ) -> Result<Self, DomainError> {
        let arm_id = arm_id.into();
        let name = name.into();
        ensure_non_empty(&arm_id, "arm_id")?;
        ensure_non_empty(&name, "name")?;
        if sample_size == 0 {
            return Err(DomainError::EmptyCollection("sample_size"));
        }
        Ok(Self {
            arm_id,
            name,
            sample_size,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialDesign {
    pub trial_design_id: String,
    pub arms: Vec<TrialArm>,
}

impl TrialDesign {
    pub fn new(
        trial_design_id: impl Into<String>,
        arms: Vec<TrialArm>,
    ) -> Result<Self, DomainError> {
        let trial_design_id = trial_design_id.into();
        ensure_non_empty(&trial_design_id, "trial_design_id")?;
        if arms.is_empty() {
            return Err(DomainError::EmptyCollection("arms"));
        }
        let mut seen = BTreeMap::new();
        for arm in &arms {
            if seen.insert(arm.arm_id.clone(), ()).is_some() {
                return Err(DomainError::DuplicateValue("arm_id"));
            }
        }
        Ok(Self {
            trial_design_id,
            arms,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticStatus {
    Ok,
    Degraded,
    Unsupported,
    Approximate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticWarning {
    pub severity: DiagnosticSeverity,
    pub code: String,
    pub message: String,
    pub capability: Option<String>,
    pub degraded_path: Option<String>,
    pub approximation: Option<bool>,
    pub backend: Option<String>,
    pub fallback: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostics {
    pub analysis_id: String,
    pub status: DiagnosticStatus,
    pub backend: Option<String>,
    pub warnings: Vec<DiagnosticWarning>,
    pub unsupported_capabilities: Vec<String>,
    pub degraded_paths: Vec<String>,
    pub approximation_caveats: Vec<String>,
}

impl Diagnostics {
    pub fn new(
        analysis_id: impl Into<String>,
        status: DiagnosticStatus,
    ) -> Result<Self, DomainError> {
        let analysis_id = analysis_id.into();
        ensure_non_empty(&analysis_id, "analysis_id")?;
        Ok(Self {
            analysis_id,
            status,
            backend: None,
            warnings: Vec::new(),
            unsupported_capabilities: Vec::new(),
            degraded_paths: Vec::new(),
            approximation_caveats: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MethodMaturity {
    Stable,
    Approximate,
    Experimental,
    BackendDependent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApproximationStatus {
    Exact,
    Approximate,
    Surrogate,
    BackendDependent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodMetadata {
    pub analysis_type: String,
    pub method_family: String,
    pub method_maturity: MethodMaturity,
    pub approximation_status: ApproximationStatus,
    pub capability_labels: Vec<String>,
    pub analysis_id: Option<String>,
    pub decision_problem_id: Option<String>,
    pub decision_context: Option<String>,
    pub backend: Option<String>,
    pub notes: Vec<String>,
}

impl MethodMetadata {
    pub fn new(
        analysis_type: impl Into<String>,
        method_family: impl Into<String>,
        method_maturity: MethodMaturity,
        approximation_status: ApproximationStatus,
    ) -> Result<Self, DomainError> {
        let analysis_type = analysis_type.into();
        let method_family = method_family.into();
        ensure_non_empty(&analysis_type, "analysis_type")?;
        ensure_non_empty(&method_family, "method_family")?;
        Ok(Self {
            analysis_type,
            method_family,
            method_maturity,
            approximation_status,
            capability_labels: Vec::new(),
            analysis_id: None,
            decision_problem_id: None,
            decision_context: None,
            backend: None,
            notes: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reporting {
    pub reporting_standard: String,
    pub analysis_type: String,
    pub method_family: String,
    pub method_maturity: MethodMaturity,
    pub analysis_id: Option<String>,
    pub decision_problem_id: Option<String>,
    pub decision_context: Option<String>,
    pub perspective_ids: Vec<String>,
    pub perspective_labels: Vec<String>,
    pub population: Option<f64>,
    pub estimator: Option<String>,
    pub seed: Option<u64>,
    pub provenance: BTreeMap<String, String>,
    pub reproducibility: BTreeMap<String, String>,
    pub diagnostics: BTreeMap<String, String>,
}

impl Reporting {
    pub fn cheers_voi(
        analysis_type: impl Into<String>,
        method_family: impl Into<String>,
        method_maturity: MethodMaturity,
    ) -> Result<Self, DomainError> {
        let analysis_type = analysis_type.into();
        let method_family = method_family.into();
        ensure_non_empty(&analysis_type, "analysis_type")?;
        ensure_non_empty(&method_family, "method_family")?;
        Ok(Self {
            reporting_standard: "CHEERS-VOI".to_string(),
            analysis_type,
            method_family,
            method_maturity,
            analysis_id: None,
            decision_problem_id: None,
            decision_context: None,
            perspective_ids: Vec::new(),
            perspective_labels: Vec::new(),
            population: None,
            estimator: None,
            seed: None,
            provenance: BTreeMap::new(),
            reproducibility: BTreeMap::new(),
            diagnostics: BTreeMap::new(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisEnvelope<T> {
    pub analysis_id: String,
    pub decision_problem_id: Option<String>,
    pub analysis_type: String,
    pub method: Option<String>,
    pub method_metadata: MethodMetadata,
    pub diagnostics: Diagnostics,
    pub reporting: Reporting,
    pub result: T,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvpiSummary {
    pub evpi: f64,
    pub expected_current_value: f64,
    pub expected_perfect_information: f64,
    pub strategy_names: Vec<String>,
    pub expected_net_benefit_by_strategy: Vec<f64>,
    pub method: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvppiSummary {
    pub evppi: f64,
    pub parameter_names: Vec<String>,
    pub expected_current_value: f64,
    pub expected_partial_information_value: f64,
    pub expected_perfect_information: f64,
    pub method: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvsiSummary {
    pub evsi: f64,
    pub trial_design_id: String,
    pub sample_size: usize,
    pub expected_current_value: f64,
    pub expected_sample_value: f64,
    pub expected_perfect_information: f64,
    pub method: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnbsSummary {
    pub enbs: f64,
    pub trial_design_id: String,
    pub sample_size: usize,
    pub design_cost: f64,
    pub expected_sample_value: f64,
    pub expected_perfect_information: f64,
    pub method: Option<String>,
}

impl<T> AnalysisEnvelope<T> {
    pub fn new(
        analysis_id: impl Into<String>,
        analysis_type: impl Into<String>,
        method_metadata: MethodMetadata,
        diagnostics: Diagnostics,
        reporting: Reporting,
        result: T,
    ) -> Result<Self, DomainError> {
        let analysis_id = analysis_id.into();
        let analysis_type = analysis_type.into();
        ensure_non_empty(&analysis_id, "analysis_id")?;
        ensure_non_empty(&analysis_type, "analysis_type")?;
        Ok(Self {
            analysis_id,
            decision_problem_id: None,
            analysis_type,
            method: None,
            method_metadata,
            diagnostics,
            reporting,
            result,
        })
    }
}

pub fn validate_reporting_payload(reporting: &Reporting) -> Result<(), DomainError> {
    ensure_non_empty(&reporting.reporting_standard, "reporting_standard")?;
    ensure_non_empty(&reporting.analysis_type, "analysis_type")?;
    ensure_non_empty(&reporting.method_family, "method_family")?;
    normalize_string_map(&reporting.provenance, "provenance")?;
    normalize_string_map(&reporting.reproducibility, "reproducibility")?;
    normalize_string_map(&reporting.diagnostics, "diagnostics")?;
    Ok(())
}
