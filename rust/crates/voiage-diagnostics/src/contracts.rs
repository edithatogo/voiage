use std::collections::BTreeSet;

use crate::{ErrorCode, ErrorDetails, ErrorRecord};
use serde::{Deserialize, Deserializer, Serialize};

type ContractResult<T> = Result<T, Box<ErrorRecord>>;

/// Overall status of a stable diagnostics envelope.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticStatus {
    /// No fallback, unsupported capability, or approximation affects the result.
    Ok,
    /// The result is usable but relies on a reduced or fallback path.
    Degraded,
    /// A requested capability could not be executed.
    Unsupported,
    /// The result requires an explicit approximation caveat.
    Approximate,
}

/// Stable warning severity.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WarningSeverity {
    /// Informational notice.
    Info,
    /// User-actionable warning.
    Warning,
    /// Critical warning affecting interpretation.
    Critical,
}

/// Validated user-facing warning record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WarningRecord {
    severity: WarningSeverity,
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    capability: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    degraded_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    approximation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallback: Option<String>,
}

impl WarningRecord {
    /// Creates a warning with a stable code and nonempty explanatory message.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input record when the code or message is blank.
    pub fn new(
        severity: WarningSeverity,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> ContractResult<Self> {
        Ok(Self {
            severity,
            code: checked_text(code, "warnings.code")?,
            message: checked_text(message, "warnings.message")?,
            capability: None,
            degraded_path: None,
            approximation: None,
            backend: None,
            fallback: None,
        })
    }

    /// Returns the warning severity.
    #[must_use]
    pub const fn severity(&self) -> WarningSeverity {
        self.severity
    }

    /// Returns the stable warning code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Returns the explanatory message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WarningWire {
    severity: WarningSeverity,
    code: String,
    message: String,
    capability: Option<String>,
    degraded_path: Option<String>,
    approximation: Option<bool>,
    backend: Option<String>,
    fallback: Option<String>,
}

impl TryFrom<WarningWire> for WarningRecord {
    type Error = Box<ErrorRecord>;

    fn try_from(wire: WarningWire) -> Result<Self, Self::Error> {
        let mut warning = Self::new(wire.severity, wire.code, wire.message)?;
        warning.capability = checked_optional_text(wire.capability, "warnings.capability")?;
        warning.degraded_path =
            checked_optional_text(wire.degraded_path, "warnings.degraded_path")?;
        warning.approximation = wire.approximation;
        warning.backend = checked_optional_text(wire.backend, "warnings.backend")?;
        warning.fallback = checked_optional_text(wire.fallback, "warnings.fallback")?;
        Ok(warning)
    }
}

/// Validated stable diagnostics envelope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Diagnostics {
    analysis_id: String,
    status: DiagnosticStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    warnings: Vec<WarningRecord>,
    unsupported_capabilities: Vec<String>,
    degraded_paths: Vec<String>,
    approximation_caveats: Vec<String>,
}

impl Diagnostics {
    /// Creates diagnostics while enforcing status/detail consistency.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input record for blank or duplicate labels, or when
    /// the status conflicts with its warning and detail lists.
    pub fn new(
        analysis_id: impl Into<String>,
        status: DiagnosticStatus,
        warnings: Vec<WarningRecord>,
        unsupported_capabilities: Vec<String>,
        degraded_paths: Vec<String>,
        approximation_caveats: Vec<String>,
    ) -> ContractResult<Self> {
        let analysis_id = checked_text(analysis_id, "analysis_id")?;
        validate_labels(&unsupported_capabilities, "unsupported_capabilities")?;
        validate_labels(&degraded_paths, "degraded_paths")?;
        validate_labels(&approximation_caveats, "approximation_caveats")?;

        let has_warning = !warnings.is_empty();
        let consistent = match status {
            DiagnosticStatus::Ok => {
                unsupported_capabilities.is_empty()
                    && degraded_paths.is_empty()
                    && approximation_caveats.is_empty()
            }
            DiagnosticStatus::Degraded => !degraded_paths.is_empty() && has_warning,
            DiagnosticStatus::Unsupported => !unsupported_capabilities.is_empty() && has_warning,
            DiagnosticStatus::Approximate => !approximation_caveats.is_empty() && has_warning,
        };
        if !consistent {
            return Err(invalid(
                "status",
                "diagnostic status is inconsistent with warnings and detail lists",
            ));
        }

        Ok(Self {
            analysis_id,
            status,
            backend: None,
            warnings,
            unsupported_capabilities,
            degraded_paths,
            approximation_caveats,
        })
    }

    /// Returns the analysis identifier.
    #[must_use]
    pub fn analysis_id(&self) -> &str {
        &self.analysis_id
    }

    /// Returns the diagnostic status.
    #[must_use]
    pub const fn status(&self) -> DiagnosticStatus {
        self.status
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticsWire {
    analysis_id: String,
    status: DiagnosticStatus,
    backend: Option<String>,
    warnings: Vec<WarningWire>,
    unsupported_capabilities: Vec<String>,
    degraded_paths: Vec<String>,
    approximation_caveats: Vec<String>,
}

impl<'de> Deserialize<'de> for Diagnostics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DiagnosticsWire::deserialize(deserializer)?;
        let warnings = wire
            .warnings
            .into_iter()
            .map(WarningRecord::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| serde::de::Error::custom(error.message()))?;
        let mut value = Self::new(
            wire.analysis_id,
            wire.status,
            warnings,
            wire.unsupported_capabilities,
            wire.degraded_paths,
            wire.approximation_caveats,
        )
        .map_err(|error| serde::de::Error::custom(error.message()))?;
        value.backend = checked_optional_text(wire.backend, "backend")
            .map_err(|error| serde::de::Error::custom(error.message()))?;
        Ok(value)
    }
}

/// Published maturity of a method family.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MethodMaturity {
    /// Stable contract method.
    Stable,
    /// Intentionally approximate method.
    Approximate,
    /// Experimental method.
    Experimental,
    /// Behavior depends on an explicit backend.
    BackendDependent,
}

/// Published approximation behavior.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApproximationStatus {
    /// Exact under the stated numerical contract.
    Exact,
    /// Explicitly approximate.
    Approximate,
    /// Uses a surrogate model.
    Surrogate,
    /// Depends on an explicit backend path.
    BackendDependent,
}

/// Validated capability, maturity, and approximation metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MethodMetadata {
    analysis_type: MethodMetadataDiscriminator,
    method_family: String,
    method_maturity: MethodMaturity,
    approximation_status: ApproximationStatus,
    capability_labels: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    analysis_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decision_problem_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decision_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<Vec<String>>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
enum MethodMetadataDiscriminator {
    #[serde(rename = "method_metadata")]
    MethodMetadata,
}

impl MethodMetadata {
    /// Creates method metadata with explicit, consistent approximation semantics.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input record for invalid labels or contradictory
    /// maturity and approximation values.
    pub fn new(
        method_family: impl Into<String>,
        method_maturity: MethodMaturity,
        approximation_status: ApproximationStatus,
        capability_labels: Vec<String>,
    ) -> ContractResult<Self> {
        let method_family = checked_text(method_family, "method_family")?;
        validate_labels(&capability_labels, "capability_labels")?;
        if matches!(
            method_maturity,
            MethodMaturity::Approximate | MethodMaturity::BackendDependent
        ) && approximation_status == ApproximationStatus::Exact
        {
            return Err(invalid(
                "approximation_status",
                "approximate or backend-dependent methods cannot report exact",
            ));
        }
        Ok(Self {
            analysis_type: MethodMetadataDiscriminator::MethodMetadata,
            method_family,
            method_maturity,
            approximation_status,
            capability_labels,
            analysis_id: None,
            decision_problem_id: None,
            decision_context: None,
            backend: None,
            notes: None,
        })
    }

    /// Sets the optional analysis identifier.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input record when the identifier is blank.
    pub fn with_analysis_id(mut self, value: impl Into<String>) -> ContractResult<Self> {
        self.analysis_id = Some(checked_text(value, "analysis_id")?);
        Ok(self)
    }

    /// Sets the optional backend identifier.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input record when the backend identifier is blank.
    pub fn with_backend(mut self, value: impl Into<String>) -> ContractResult<Self> {
        self.backend = Some(checked_text(value, "backend")?);
        Ok(self)
    }

    /// Returns the stable payload discriminator.
    #[must_use]
    pub const fn analysis_type(&self) -> &'static str {
        "method_metadata"
    }

    /// Returns the method family.
    #[must_use]
    pub fn method_family(&self) -> &str {
        &self.method_family
    }

    /// Returns the optional analysis identifier.
    #[must_use]
    pub fn analysis_id(&self) -> Option<&str> {
        self.analysis_id.as_deref()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodMetadataWire {
    analysis_type: MethodMetadataDiscriminator,
    method_family: String,
    method_maturity: MethodMaturity,
    approximation_status: ApproximationStatus,
    capability_labels: Vec<String>,
    analysis_id: Option<String>,
    decision_problem_id: Option<String>,
    decision_context: Option<String>,
    backend: Option<String>,
    notes: Option<Vec<String>>,
}

impl<'de> Deserialize<'de> for MethodMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MethodMetadataWire::deserialize(deserializer)?;
        let mut value = Self::new(
            wire.method_family,
            wire.method_maturity,
            wire.approximation_status,
            wire.capability_labels,
        )
        .map_err(|error| serde::de::Error::custom(error.message()))?;
        value.analysis_type = wire.analysis_type;
        value.analysis_id = checked_optional_text(wire.analysis_id, "analysis_id")
            .map_err(|error| serde::de::Error::custom(error.message()))?;
        value.decision_problem_id =
            checked_optional_text(wire.decision_problem_id, "decision_problem_id")
                .map_err(|error| serde::de::Error::custom(error.message()))?;
        value.decision_context = checked_optional_text(wire.decision_context, "decision_context")
            .map_err(|error| serde::de::Error::custom(error.message()))?;
        value.backend = checked_optional_text(wire.backend, "backend")
            .map_err(|error| serde::de::Error::custom(error.message()))?;
        if let Some(notes) = wire.notes {
            validate_labels(&notes, "notes")
                .map_err(|error| serde::de::Error::custom(error.message()))?;
            value.notes = Some(notes);
        }
        Ok(value)
    }
}

fn checked_text(value: impl Into<String>, field: &'static str) -> ContractResult<String> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(invalid(field, "value must not be blank"));
    }
    Ok(value)
}

fn checked_optional_text(
    value: Option<String>,
    field: &'static str,
) -> ContractResult<Option<String>> {
    value.map(|value| checked_text(value, field)).transpose()
}

fn validate_labels(values: &[String], field: &'static str) -> ContractResult<()> {
    let mut unique = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(invalid(field, "labels must not be blank"));
        }
        if !unique.insert(value) {
            return Err(invalid(field, "labels must be unique"));
        }
    }
    Ok(())
}

fn invalid(field: &'static str, message: &'static str) -> Box<ErrorRecord> {
    Box::new(
        ErrorRecord::new(ErrorCode::InvalidInput, message)
            .with_details(ErrorDetails::new().with_field(field)),
    )
}
