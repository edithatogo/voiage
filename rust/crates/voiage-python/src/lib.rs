//! Private, non-numerical Python adapter for the voiage Rust core.
//!
//! Run interpreter-backed native tests reproducibly with:
//! `PYO3_PYTHON="$(command -v python3)" cargo test -p voiage-python`.

#![forbid(unsafe_code)]

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use voiage_diagnostics::ErrorCategory;
use voiage_domain::{SampleCube, SampleMatrix, SampleVector};
use voiage_numerics::{
    ceaf, dominance, enbs, evpi, evppi, evsi_efficient_linear, evsi_moment_based, evsi_regression,
    evsi_stochastic, heterogeneity, structural_evpi, structural_evppi,
    DominanceStatus as KernelDominanceStatus,
};
use voiage_serialization::{
    CeafResultV1, CeafResultV1Input, DominanceResultV1, DominanceResultV1Input, DominanceStatus,
};

create_exception!(
    _core,
    InputError,
    PyException,
    "Native invalid-input exception carrying the stable invalid_input code."
);
create_exception!(
    _core,
    DimensionMismatchError,
    PyException,
    "Native shape-mismatch exception carrying the stable dimension_mismatch code."
);
create_exception!(
    _core,
    SerializationError,
    PyException,
    "Native encoding exception carrying the stable serialization_failure code."
);

static CEAF_TELEMETRY: OperationTelemetry = OperationTelemetry::new();
static DOMINANCE_TELEMETRY: OperationTelemetry = OperationTelemetry::new();
const DIGEST_ALGORITHM: &str = "rfc8785-sha256-v1";
const BUILD_ID_ALGORITHM: &str = env!("VOIAGE_BUILD_ID_ALGORITHM");

#[derive(Debug)]
struct BuildMetadata {
    runtime_info_schema: u32,
    source_revision: &'static str,
    source_dirty: bool,
    target_triple: &'static str,
    rustc_version: &'static str,
    build_profile: &'static str,
    cargo_lock_sha256: &'static str,
    source_tree_git_oid: &'static str,
    source_state_sha256: &'static str,
    source_state_algorithm: &'static str,
    build_id: &'static str,
    source_date_epoch: Option<i64>,
}

fn build_metadata() -> BuildMetadata {
    BuildMetadata {
        runtime_info_schema: 3,
        source_revision: env!("VOIAGE_SOURCE_REVISION"),
        source_dirty: env!("VOIAGE_SOURCE_DIRTY") == "true",
        target_triple: env!("VOIAGE_TARGET_TRIPLE"),
        rustc_version: env!("VOIAGE_RUSTC_VERSION"),
        build_profile: env!("VOIAGE_BUILD_PROFILE"),
        cargo_lock_sha256: env!("VOIAGE_CARGO_LOCK_SHA256"),
        source_tree_git_oid: env!("VOIAGE_SOURCE_TREE_GIT_OID"),
        source_state_sha256: env!("VOIAGE_SOURCE_STATE_SHA256"),
        source_state_algorithm: env!("VOIAGE_SOURCE_STATE_ALGORITHM"),
        build_id: env!("VOIAGE_BUILD_ID"),
        source_date_epoch: match env!("VOIAGE_SOURCE_DATE_EPOCH") {
            "" => None,
            value => Some(match value.parse() {
                Ok(epoch) => epoch,
                Err(_) => unreachable!("build.rs validates SOURCE_DATE_EPOCH"),
            }),
        },
    }
}

#[cfg(test)]
fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
fn build_id_for_metadata(metadata: &BuildMetadata) -> String {
    let dirty = if metadata.source_dirty {
        "true"
    } else {
        "false"
    };
    let epoch = metadata
        .source_date_epoch
        .map(|value| value.to_string())
        .unwrap_or_default();
    let identity = [
        BUILD_ID_ALGORITHM,
        metadata.source_revision,
        metadata.source_tree_git_oid,
        dirty,
        metadata.source_state_sha256,
        metadata.target_triple,
        metadata.rustc_version,
        metadata.build_profile,
        metadata.cargo_lock_sha256,
        &epoch,
    ]
    .into_iter()
    .fold(String::new(), |mut identity, value| {
        write!(identity, "{}:{value}", value.len()).expect("writing to String is infallible");
        identity
    });
    sha256_hex(identity.as_bytes())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for &byte in &digest {
        hex.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        hex.push(char::from(b"0123456789abcdef"[(byte & 0xf) as usize]));
    }
    hex
}

fn canonical_payload_bytes<T: serde::Serialize>(value: &T) -> serde_json::Result<Vec<u8>> {
    serde_json_canonicalizer::to_vec(value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OperationSnapshot {
    attempts: u64,
    successes: u64,
    failures: u64,
    calls: u64,
    last_payload_sha256: Option<String>,
}

#[derive(Debug)]
struct OperationTelemetry {
    attempts: AtomicU64,
    successes: AtomicU64,
    failures: AtomicU64,
    snapshot_gate: Mutex<()>,
    last_payload_sha256: Mutex<Option<String>>,
}

impl OperationTelemetry {
    const fn new() -> Self {
        Self {
            attempts: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            snapshot_gate: Mutex::new(()),
            last_payload_sha256: Mutex::new(None),
        }
    }

    #[cfg(test)]
    fn with_counts_for_test(attempts: u64, successes: u64, failures: u64) -> Self {
        Self {
            attempts: AtomicU64::new(attempts),
            successes: AtomicU64::new(successes),
            failures: AtomicU64::new(failures),
            snapshot_gate: Mutex::new(()),
            last_payload_sha256: Mutex::new(None),
        }
    }

    fn saturating_increment(counter: &AtomicU64) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            Some(value.saturating_add(1))
        });
    }

    fn record_attempt(&self) {
        let _gate = self
            .snapshot_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::saturating_increment(&self.attempts);
    }

    fn record_success(&self, canonical_payload: &[u8]) {
        let _gate = self
            .snapshot_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut digest = self
            .last_payload_sha256
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *digest = Some(sha256_hex(canonical_payload));
        Self::saturating_increment(&self.successes);
    }

    fn record_failure(&self) {
        let _gate = self
            .snapshot_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::saturating_increment(&self.failures);
    }

    fn snapshot(&self) -> OperationSnapshot {
        let _gate = self
            .snapshot_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let digest = self
            .last_payload_sha256
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let successes = self.successes.load(Ordering::Relaxed);
        OperationSnapshot {
            attempts: self.attempts.load(Ordering::Relaxed),
            successes,
            failures: self.failures.load(Ordering::Relaxed),
            calls: successes,
            last_payload_sha256: digest.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum BoundaryError {
    Input(String),
    DimensionMismatch(String),
}

impl BoundaryError {
    fn into_pyerr(self) -> PyErr {
        match self {
            Self::Input(message) => InputError::new_err(("invalid_input", message)),
            Self::DimensionMismatch(message) => {
                DimensionMismatchError::new_err(("dimension_mismatch", message))
            }
        }
    }
}

fn validate_identifiers(analysis_id: &str, decision_problem_id: &str) -> Result<(), BoundaryError> {
    if analysis_id.trim().is_empty() {
        return Err(BoundaryError::Input(
            "`analysis_id` must be a non-empty string.".into(),
        ));
    }
    if decision_problem_id.trim().is_empty() {
        return Err(BoundaryError::Input(
            "`decision_problem_id` must be a non-empty string.".into(),
        ));
    }
    Ok(())
}

fn indices_from_python(values: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<u64>> {
    let invalid = || {
        InputError::new_err((
            "invalid_input",
            format!("`{field}` must contain integers in the range 0..=i64::MAX."),
        ))
    };
    values
        .try_iter()
        .map_err(|_| invalid())?
        .map(|item| {
            let item = item.map_err(|_| invalid())?;
            let value = item.extract::<i64>().map_err(|_| invalid())?;
            u64::try_from(value).map_err(|_| invalid())
        })
        .collect()
}

fn matrix_from_python(values: &Bound<'_, PyAny>, field: &str) -> PyResult<SampleMatrix> {
    let value = if values.hasattr("tolist")? {
        values.call_method0("tolist")?
    } else {
        values.clone()
    };
    let rows = value.extract::<Vec<Vec<f64>>>().map_err(|error| {
        InputError::new_err((
            "invalid_input",
            format!("{field} must be a finite rectangular numeric matrix: {error}"),
        ))
    })?;
    SampleMatrix::try_from(rows).map_err(|error| {
        InputError::new_err((
            "invalid_input",
            format!("{field} must be a finite rectangular numeric matrix: {error}"),
        ))
    })
}

fn cube_from_python(values: &Bound<'_, PyAny>, field: &str) -> PyResult<SampleCube> {
    let value = if values.hasattr("tolist")? {
        values.call_method0("tolist")?
    } else {
        values.clone()
    };
    let values = value.extract::<Vec<Vec<Vec<f64>>>>().map_err(|error| {
        InputError::new_err((
            "invalid_input",
            format!("{field} must be a finite rectangular numeric cube: {error}"),
        ))
    })?;
    SampleCube::try_from(values).map_err(|error| {
        InputError::new_err((
            "invalid_input",
            format!("{field} must be a finite rectangular numeric cube: {error}"),
        ))
    })
}

fn vector_from_python(values: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<f64>> {
    let value = if values.hasattr("tolist")? {
        values.call_method0("tolist")?
    } else {
        values.clone()
    };
    value.extract::<Vec<f64>>().map_err(|error| {
        InputError::new_err((
            "invalid_input",
            format!("{field} must be a finite numeric vector: {error}"),
        ))
    })
}

fn validate_ceaf_alignment(lengths: &[usize]) -> Result<(), BoundaryError> {
    let Some((&expected, remaining)) = lengths.split_first() else {
        return Err(BoundaryError::Input(
            "alignment validation requires at least one array.".into(),
        ));
    };
    if remaining.iter().any(|&length| length != expected) {
        return Err(BoundaryError::DimensionMismatch(
            "CEAF arrays must be aligned.".into(),
        ));
    }
    Ok(())
}

fn validate_dominance_alignment(
    strategy_count: usize,
    costs: usize,
    effects: usize,
    statuses: usize,
    frontier_count: usize,
    incremental_lengths: [usize; 3],
) -> Result<(), BoundaryError> {
    if [costs, effects, statuses]
        .into_iter()
        .any(|length| length != strategy_count)
    {
        return Err(BoundaryError::DimensionMismatch(
            "dominance strategy arrays must be aligned.".into(),
        ));
    }
    let transitions = frontier_count.saturating_sub(1);
    if incremental_lengths
        .into_iter()
        .any(|length| length != transitions)
    {
        return Err(BoundaryError::DimensionMismatch(
            "dominance incremental arrays must align with frontier transitions.".into(),
        ));
    }
    Ok(())
}

fn serialization_error(error: impl std::fmt::Display) -> PyErr {
    SerializationError::new_err(("serialization_failure", error.to_string()))
}

fn reporting_from_python(
    reporting: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Map<String, Value>>> {
    reporting
        .map(|value| {
            let py = value.py();
            let encoded: String = py
                .import("json")?
                .call_method1("dumps", (value,))
                .map_err(serialization_error)?
                .extract()
                .map_err(serialization_error)?;
            let decoded: Value = serde_json::from_str(&encoded).map_err(serialization_error)?;
            decoded.as_object().cloned().ok_or_else(|| {
                InputError::new_err(("invalid_input", "reporting must be a mapping"))
            })
        })
        .transpose()
}

fn result_to_dict<'py, T: serde::Serialize>(
    py: Python<'py>,
    result: &T,
) -> PyResult<(Bound<'py, PyDict>, Vec<u8>)> {
    // RFC 8785 makes the hashed representation independently reproducible in
    // Python and other runtimes, including ECMAScript-compatible float output.
    let canonical = canonical_payload_bytes(result).map_err(serialization_error)?;
    // Do not use JCS bytes to construct the Python object: JCS renders 1e20 as
    // an integer token, which Python would widen to int and thereby lose the
    // result DTO's floating-point type. Ordinary serde JSON preserves the
    // exponent token while the separate canonical bytes remain the lineage
    // input.
    let returned = serde_json::to_vec(result).map_err(serialization_error)?;
    let returned_text = std::str::from_utf8(&returned).map_err(serialization_error)?;
    py.import("json")
        .map_err(serialization_error)?
        .call_method1("loads", (returned_text,))
        .map_err(serialization_error)?
        .cast_into::<PyDict>()
        .map(|dictionary| (dictionary, canonical))
        .map_err(serialization_error)
}

/// Return build provenance and per-operation invocation counts.
#[pyfunction]
fn runtime_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let metadata = build_metadata();
    let result = PyDict::new(py);
    result.set_item("runtime_info_schema", metadata.runtime_info_schema)?;
    result.set_item("engine", "rust")?;
    result.set_item("bridge", "pyo3")?;
    result.set_item("core_version", env!("CARGO_PKG_VERSION"))?;
    result.set_item("abi_version", 1_u32)?;
    result.set_item("digest_algorithm", DIGEST_ALGORITHM)?;
    result.set_item("build_id_algorithm", BUILD_ID_ALGORITHM)?;
    result.set_item("source_revision", metadata.source_revision)?;
    result.set_item("source_dirty", metadata.source_dirty)?;
    result.set_item("target_triple", metadata.target_triple)?;
    result.set_item("target", metadata.target_triple)?;
    result.set_item("rustc_version", metadata.rustc_version)?;
    result.set_item("build_profile", metadata.build_profile)?;
    result.set_item("profile", metadata.build_profile)?;
    result.set_item("cargo_lock_sha256", metadata.cargo_lock_sha256)?;
    result.set_item("source_tree_git_oid", metadata.source_tree_git_oid)?;
    result.set_item("source_state_sha256", metadata.source_state_sha256)?;
    result.set_item("source_state_algorithm", metadata.source_state_algorithm)?;
    result.set_item("build_id", metadata.build_id)?;
    result.set_item("source_date_epoch", metadata.source_date_epoch)?;

    let operations = PyDict::new(py);
    add_operation_snapshot(py, &operations, "serialize_ceaf_result", &CEAF_TELEMETRY)?;
    add_operation_snapshot(
        py,
        &operations,
        "serialize_dominance_result",
        &DOMINANCE_TELEMETRY,
    )?;
    result.set_item("operations", operations)?;
    Ok(result)
}

/// Compute the stable EVPI kernel for Python callers.
#[pyfunction]
fn compute_evpi(net_benefit: &Bound<'_, PyAny>) -> PyResult<f64> {
    let net_benefit = matrix_from_python(net_benefit, "net_benefit")?;
    evpi(&net_benefit).map_err(|error| match error.category() {
        ErrorCategory::DimensionMismatch => {
            DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
        }
        _ => InputError::new_err(("invalid_input", error.to_string())),
    })
}

/// Compute the stable ENBS kernel for Python callers.
#[pyfunction]
fn compute_enbs(evsi_result: f64, research_cost: f64) -> PyResult<f64> {
    enbs(evsi_result, research_cost)
        .map_err(|error| InputError::new_err(("invalid_input", error.to_string())))
}

/// Compute the stable value-of-heterogeneity kernel for Python callers.
#[pyfunction]
fn compute_heterogeneity<'py>(
    py: Python<'py>,
    net_benefit: &Bound<'_, PyAny>,
    subgroups: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let net_benefit = matrix_from_python(net_benefit, "net_benefit")?;
    let subgroups = subgroups.extract::<Vec<String>>().map_err(|error| {
        InputError::new_err(("invalid_input", format!("invalid subgroups: {error}")))
    })?;
    let rows = net_benefit.rows().map(<[f64]>::to_vec).collect::<Vec<_>>();
    let result = heterogeneity(&rows, &subgroups).map_err(|error| match error.category() {
        ErrorCategory::DimensionMismatch => {
            DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
        }
        _ => InputError::new_err(("invalid_input", error.to_string())),
    })?;
    let output = PyDict::new(py);
    output.set_item("value", result.value)?;
    output.set_item("subgroup_labels", result.subgroup_labels)?;
    output.set_item("subgroup_weights", result.subgroup_weights)?;
    output.set_item(
        "subgroup_optimal_strategy_indices",
        result.subgroup_optimal_strategy_indices,
    )?;
    output.set_item(
        "subgroup_expected_net_benefits",
        result.subgroup_expected_net_benefits,
    )?;
    output.set_item(
        "overall_optimal_strategy_index",
        result.overall_optimal_strategy_index,
    )?;
    output.set_item(
        "overall_expected_net_benefit",
        result.overall_expected_net_benefit,
    )?;
    Ok(output)
}

/// Aggregate structural EVPI after Python model evaluators have run.
#[pyfunction]
fn compute_structural_evpi(
    net_benefit_by_structure: &Bound<'_, PyAny>,
    structure_probabilities: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let net_benefit_by_structure = cube_from_python(net_benefit_by_structure, "net_benefit")?;
    let structure_probabilities = SampleVector::try_from(vector_from_python(
        structure_probabilities,
        "structure_probabilities",
    )?)
    .map_err(|error| InputError::new_err(("invalid_input", error.to_string())))?;
    structural_evpi(&net_benefit_by_structure, &structure_probabilities).map_err(
        |error| match error.category() {
            ErrorCategory::DimensionMismatch => {
                DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
            }
            _ => InputError::new_err(("invalid_input", error.to_string())),
        },
    )
}

/// Aggregate structural EVPPI after Python model evaluators have run.
#[pyfunction]
fn compute_structural_evppi(
    net_benefit_by_structure: &Bound<'_, PyAny>,
    structure_probabilities: &Bound<'_, PyAny>,
    structures_of_interest: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let cube = cube_from_python(net_benefit_by_structure, "net_benefit")?;
    let probabilities = SampleVector::try_from(vector_from_python(
        structure_probabilities,
        "structure_probabilities",
    )?)
    .map_err(|error| InputError::new_err(("invalid_input", error.to_string())))?;
    let indices = indices_from_python(structures_of_interest, "structures_of_interest")?
        .into_iter()
        .map(|index| {
            usize::try_from(index).map_err(|_| {
                InputError::new_err((
                    "invalid_input",
                    "structure index exceeds the supported platform range",
                ))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    structural_evppi(&cube, &probabilities, &indices).map_err(|error| match error.category() {
        ErrorCategory::DimensionMismatch => {
            DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
        }
        _ => InputError::new_err(("invalid_input", error.to_string())),
    })
}

/// Compute the stable dominance kernel for Python callers.
#[pyfunction]
fn compute_dominance<'py>(
    py: Python<'py>,
    costs: &Bound<'_, PyAny>,
    effects: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let costs = vector_from_python(costs, "costs")?;
    let effects = vector_from_python(effects, "effects")?;
    let costs_domain = voiage_domain::SampleVector::try_from(costs).map_err(|error| {
        InputError::new_err(("invalid_input", format!("invalid costs vector: {error}")))
    })?;
    let effects_domain = voiage_domain::SampleVector::try_from(effects).map_err(|error| {
        InputError::new_err(("invalid_input", format!("invalid effects vector: {error}")))
    })?;
    let result =
        dominance(&costs_domain, &effects_domain).map_err(|error| match error.category() {
            ErrorCategory::DimensionMismatch => {
                DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
            }
            _ => InputError::new_err(("invalid_input", error.to_string())),
        })?;
    let status = result
        .status
        .into_iter()
        .map(|value| match value {
            KernelDominanceStatus::Frontier => "frontier",
            KernelDominanceStatus::StronglyDominated => "strongly_dominated",
            KernelDominanceStatus::ExtendedDominated => "extended_dominated",
        })
        .collect::<Vec<_>>();
    let output = PyDict::new(py);
    output.set_item("frontier_indices", result.frontier_indices)?;
    output.set_item(
        "strongly_dominated_indices",
        result.strongly_dominated_indices,
    )?;
    output.set_item(
        "extended_dominated_indices",
        result.extended_dominated_indices,
    )?;
    output.set_item("status", status)?;
    output.set_item("incremental_costs", result.incremental_costs)?;
    output.set_item("incremental_effects", result.incremental_effects)?;
    output.set_item("icers", result.icers)?;
    Ok(output)
}

/// Compute the stable CEAF kernel for Python callers.
#[pyfunction]
fn compute_ceaf<'py>(
    py: Python<'py>,
    net_benefit: &Bound<'_, PyAny>,
    wtp_thresholds: &Bound<'_, PyAny>,
    confidence_level: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let net_benefit = cube_from_python(net_benefit, "net_benefit")?;
    let wtp_thresholds =
        SampleVector::try_from(vector_from_python(wtp_thresholds, "wtp_thresholds")?)
            .map_err(|error| InputError::new_err(("invalid_input", error.to_string())))?;
    let result =
        ceaf(&net_benefit, &wtp_thresholds, confidence_level).map_err(|error| {
            match error.category() {
                ErrorCategory::DimensionMismatch => {
                    DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
                }
                _ => InputError::new_err(("invalid_input", error.to_string())),
            }
        })?;
    let output = PyDict::new(py);
    output.set_item("wtp_thresholds", result.wtp_thresholds)?;
    output.set_item("optimal_strategy_indices", result.optimal_strategy_indices)?;
    output.set_item(
        "acceptability_probabilities",
        result.acceptability_probabilities,
    )?;
    output.set_item("probability_lower", result.probability_lower)?;
    output.set_item("probability_upper", result.probability_upper)?;
    output.set_item("expected_net_benefit", result.expected_net_benefit)?;
    Ok(output)
}

/// Compute the stable regression-based EVPPI kernel for Python callers.
#[pyfunction]
fn compute_evppi(
    net_benefit: &Bound<'_, PyAny>,
    parameter_samples: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let net_benefit = matrix_from_python(net_benefit, "net_benefit")?;
    let parameter_samples = matrix_from_python(parameter_samples, "parameter_samples")?;
    evppi(&net_benefit, &parameter_samples).map_err(|error| match error.category() {
        ErrorCategory::DimensionMismatch => {
            DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
        }
        _ => InputError::new_err(("invalid_input", error.to_string())),
    })
}

/// Compute the explicit seeded-bootstrap EVSI kernel for Python callers.
#[pyfunction]
#[pyo3(signature = (net_benefit, trial_sample_size, resample_count, seed))]
fn compute_evsi<'py>(
    py: Python<'py>,
    net_benefit: &Bound<'_, PyAny>,
    trial_sample_size: usize,
    resample_count: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let net_benefit = matrix_from_python(net_benefit, "net_benefit")?;
    let result = evsi_stochastic(&net_benefit, trial_sample_size, resample_count, seed).map_err(
        |error| match error.category() {
            ErrorCategory::DimensionMismatch => {
                DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
            }
            _ => InputError::new_err(("invalid_input", error.to_string())),
        },
    )?;
    let output = PyDict::new(py);
    output.set_item("estimator", result.estimator)?;
    output.set_item("contract_version", result.contract_version)?;
    output.set_item("expected_current_value", result.expected_current_value)?;
    output.set_item("expected_sample_value", result.expected_sample_value)?;
    output.set_item(
        "expected_perfect_information",
        result.expected_perfect_information,
    )?;
    output.set_item("evsi", result.evsi)?;
    output.set_item("draw_count", result.draw_count)?;
    output.set_item("resample_count", result.resample_count)?;
    Ok(output)
}

/// Compute the explicit deterministic efficient-linear EVSI kernel.
#[pyfunction]
#[pyo3(signature = (net_benefit, parameter_samples, trial_sample_size))]
fn compute_evsi_efficient_linear<'py>(
    py: Python<'py>,
    net_benefit: &Bound<'_, PyAny>,
    parameter_samples: &Bound<'_, PyAny>,
    trial_sample_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let net_benefit = matrix_from_python(net_benefit, "net_benefit")?;
    let parameter_samples = matrix_from_python(parameter_samples, "parameter_samples")?;
    let result = evsi_efficient_linear(&net_benefit, &parameter_samples, trial_sample_size)
        .map_err(|error| match error.category() {
            ErrorCategory::DimensionMismatch => {
                DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
            }
            _ => InputError::new_err(("invalid_input", error.to_string())),
        })?;
    let output = PyDict::new(py);
    output.set_item("estimator", result.estimator)?;
    output.set_item("contract_version", result.contract_version)?;
    output.set_item("expected_current_value", result.expected_current_value)?;
    output.set_item("expected_sample_value", result.expected_sample_value)?;
    output.set_item(
        "expected_perfect_information",
        result.expected_perfect_information,
    )?;
    output.set_item("information_fraction", result.information_fraction)?;
    output.set_item("evsi", result.evsi)?;
    output.set_item("sample_count", result.sample_count)?;
    output.set_item("strategy_count", result.strategy_count)?;
    output.set_item("parameter_count", result.parameter_count)?;
    Ok(output)
}

/// Compute the explicit deterministic moment-based EVSI kernel.
#[pyfunction]
#[pyo3(signature = (net_benefit, parameter_samples, trial_sample_size))]
fn compute_evsi_moment_based<'py>(
    py: Python<'py>,
    net_benefit: &Bound<'_, PyAny>,
    parameter_samples: &Bound<'_, PyAny>,
    trial_sample_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let net_benefit = matrix_from_python(net_benefit, "net_benefit")?;
    let parameter_samples = matrix_from_python(parameter_samples, "parameter_samples")?;
    let result = evsi_moment_based(&net_benefit, &parameter_samples, trial_sample_size).map_err(
        |error| match error.category() {
            ErrorCategory::DimensionMismatch => {
                DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
            }
            _ => InputError::new_err(("invalid_input", error.to_string())),
        },
    )?;
    let output = PyDict::new(py);
    output.set_item("estimator", result.estimator)?;
    output.set_item("contract_version", result.contract_version)?;
    output.set_item("expected_current_value", result.expected_current_value)?;
    output.set_item("expected_sample_value", result.expected_sample_value)?;
    output.set_item(
        "expected_perfect_information",
        result.expected_perfect_information,
    )?;
    output.set_item("information_fraction", result.information_fraction)?;
    output.set_item("evsi", result.evsi)?;
    output.set_item("sample_count", result.sample_count)?;
    output.set_item("strategy_count", result.strategy_count)?;
    output.set_item("parameter_count", result.parameter_count)?;
    Ok(output)
}

/// Compute the callback-driven deterministic regression aggregation kernel.
#[pyfunction]
fn compute_evsi_regression<'py>(
    py: Python<'py>,
    targets: &Bound<'_, PyAny>,
    parameter_samples: &Bound<'_, PyAny>,
    prediction_samples: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let targets = matrix_from_python(targets, "targets")?;
    let parameter_samples = matrix_from_python(parameter_samples, "parameter_samples")?;
    let prediction_samples = matrix_from_python(prediction_samples, "prediction_samples")?;
    let result =
        evsi_regression(&targets, &parameter_samples, &prediction_samples).map_err(|error| {
            match error.category() {
                ErrorCategory::DimensionMismatch => {
                    DimensionMismatchError::new_err(("dimension_mismatch", error.to_string()))
                }
                _ => InputError::new_err(("invalid_input", error.to_string())),
            }
        })?;
    let output = PyDict::new(py);
    output.set_item("estimator", result.estimator)?;
    output.set_item("contract_version", result.contract_version)?;
    output.set_item("expected_sample_value", result.expected_sample_value)?;
    output.set_item("sample_count", result.sample_count)?;
    output.set_item("prediction_count", result.prediction_count)?;
    output.set_item("parameter_count", result.parameter_count)?;
    Ok(output)
}

fn add_operation_snapshot(
    py: Python<'_>,
    operations: &Bound<'_, PyDict>,
    name: &str,
    telemetry: &OperationTelemetry,
) -> PyResult<()> {
    let snapshot = telemetry.snapshot();
    let operation = PyDict::new(py);
    operation.set_item("native_entries", snapshot.attempts)?;
    operation.set_item("successes", snapshot.successes)?;
    operation.set_item("failures", snapshot.failures)?;
    operation.set_item("calls", snapshot.calls)?;
    operation.set_item("digest_algorithm", DIGEST_ALGORITHM)?;
    operation.set_item("last_payload_sha256", snapshot.last_payload_sha256)?;
    operations.set_item(name, operation)
}

/// Construct, validate, and serialize a canonical CEAF result without computing it.
#[pyfunction]
#[pyo3(signature = (*, analysis_id, decision_problem_id, wtp_thresholds, optimal_strategy_indices, optimal_strategy_names, acceptability_probabilities, probability_lower, probability_upper, expected_net_benefit, reporting=None))]
#[allow(clippy::too_many_arguments)]
fn serialize_ceaf_result<'py>(
    py: Python<'py>,
    analysis_id: String,
    decision_problem_id: String,
    wtp_thresholds: Vec<f64>,
    optimal_strategy_indices: &Bound<'_, PyAny>,
    optimal_strategy_names: Vec<String>,
    acceptability_probabilities: Vec<f64>,
    probability_lower: Vec<f64>,
    probability_upper: Vec<f64>,
    expected_net_benefit: Vec<f64>,
    reporting: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    CEAF_TELEMETRY.record_attempt();
    let outcome = (|| {
        validate_identifiers(&analysis_id, &decision_problem_id)
            .map_err(BoundaryError::into_pyerr)?;
        let optimal_strategy_indices =
            indices_from_python(optimal_strategy_indices, "optimal_strategy_indices")?;
        validate_ceaf_alignment(&[
            wtp_thresholds.len(),
            optimal_strategy_indices.len(),
            optimal_strategy_names.len(),
            acceptability_probabilities.len(),
            probability_lower.len(),
            probability_upper.len(),
            expected_net_benefit.len(),
        ])
        .map_err(BoundaryError::into_pyerr)?;
        let result = CeafResultV1::try_from(CeafResultV1Input {
            analysis_id,
            decision_problem_id,
            wtp_thresholds,
            optimal_strategy_indices,
            optimal_strategy_names,
            acceptability_probabilities,
            probability_lower,
            probability_upper,
            expected_net_benefit,
            reporting: reporting_from_python(reporting)?,
        })
        .map_err(|error| InputError::new_err(("invalid_input", error.to_string())))?;
        result_to_dict(py, &result)
    })();
    match outcome {
        Ok((result, canonical_payload)) => {
            CEAF_TELEMETRY.record_success(&canonical_payload);
            Ok(result)
        }
        Err(error) => {
            CEAF_TELEMETRY.record_failure();
            Err(error)
        }
    }
}

/// Construct, validate, and serialize a canonical dominance result without computing it.
#[pyfunction]
#[pyo3(signature = (*, analysis_id, decision_problem_id, strategy_names, costs, effects, frontier_indices, strongly_dominated_indices, extended_dominated_indices, status, incremental_costs, incremental_effects, icers, reporting=None))]
#[allow(clippy::too_many_arguments)]
fn serialize_dominance_result<'py>(
    py: Python<'py>,
    analysis_id: String,
    decision_problem_id: String,
    strategy_names: Vec<String>,
    costs: Vec<f64>,
    effects: Vec<f64>,
    frontier_indices: &Bound<'_, PyAny>,
    strongly_dominated_indices: &Bound<'_, PyAny>,
    extended_dominated_indices: &Bound<'_, PyAny>,
    status: Vec<String>,
    incremental_costs: Vec<f64>,
    incremental_effects: Vec<f64>,
    icers: Vec<f64>,
    reporting: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    DOMINANCE_TELEMETRY.record_attempt();
    let outcome = (|| {
        validate_identifiers(&analysis_id, &decision_problem_id)
            .map_err(BoundaryError::into_pyerr)?;
        let frontier_indices = indices_from_python(frontier_indices, "frontier_indices")?;
        let strongly_dominated_indices =
            indices_from_python(strongly_dominated_indices, "strongly_dominated_indices")?;
        let extended_dominated_indices =
            indices_from_python(extended_dominated_indices, "extended_dominated_indices")?;
        validate_dominance_alignment(
            strategy_names.len(),
            costs.len(),
            effects.len(),
            status.len(),
            frontier_indices.len(),
            [
                incremental_costs.len(),
                incremental_effects.len(),
                icers.len(),
            ],
        )
        .map_err(BoundaryError::into_pyerr)?;
        let status = status
            .into_iter()
            .map(|value| match value.as_str() {
                "frontier" => Ok(DominanceStatus::Frontier),
                "strongly_dominated" => Ok(DominanceStatus::StronglyDominated),
                "extended_dominated" => Ok(DominanceStatus::ExtendedDominated),
                _ => Err(InputError::new_err((
                    "invalid_input",
                    format!("unknown dominance status: {value}"),
                ))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let result = DominanceResultV1::try_from(DominanceResultV1Input {
            analysis_id,
            decision_problem_id,
            strategy_names,
            costs,
            effects,
            frontier_indices,
            strongly_dominated_indices,
            extended_dominated_indices,
            status,
            incremental_costs,
            incremental_effects,
            icers,
            reporting: reporting_from_python(reporting)?,
        })
        .map_err(|error| InputError::new_err(("invalid_input", error.to_string())))?;
        result_to_dict(py, &result)
    })();
    match outcome {
        Ok((result, canonical_payload)) => {
            DOMINANCE_TELEMETRY.record_success(&canonical_payload);
            Ok(result)
        }
        Err(error) => {
            DOMINANCE_TELEMETRY.record_failure();
            Err(error)
        }
    }
}

/// Register the private `voiage._core` extension module.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("InputError", module.py().get_type::<InputError>())?;
    module.add(
        "DimensionMismatchError",
        module.py().get_type::<DimensionMismatchError>(),
    )?;
    module.add(
        "SerializationError",
        module.py().get_type::<SerializationError>(),
    )?;
    module.add_function(wrap_pyfunction!(runtime_info, module)?)?;
    module.add_function(wrap_pyfunction!(compute_evpi, module)?)?;
    module.add_function(wrap_pyfunction!(compute_enbs, module)?)?;
    module.add_function(wrap_pyfunction!(compute_heterogeneity, module)?)?;
    module.add_function(wrap_pyfunction!(compute_structural_evpi, module)?)?;
    module.add_function(wrap_pyfunction!(compute_structural_evppi, module)?)?;
    module.add_function(wrap_pyfunction!(compute_dominance, module)?)?;
    module.add_function(wrap_pyfunction!(compute_ceaf, module)?)?;
    module.add_function(wrap_pyfunction!(compute_evppi, module)?)?;
    module.add_function(wrap_pyfunction!(compute_evsi, module)?)?;
    module.add_function(wrap_pyfunction!(compute_evsi_efficient_linear, module)?)?;
    module.add_function(wrap_pyfunction!(compute_evsi_moment_based, module)?)?;
    module.add_function(wrap_pyfunction!(compute_evsi_regression, module)?)?;
    module.add_function(wrap_pyfunction!(serialize_ceaf_result, module)?)?;
    module.add_function(wrap_pyfunction!(serialize_dominance_result, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;
    use serde_json::json;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn validation_categories_dimension_mismatches_stably() {
        assert_eq!(
            validate_ceaf_alignment(&[1, 0, 1, 1, 1, 1, 1]),
            Err(BoundaryError::DimensionMismatch(
                "CEAF arrays must be aligned.".into()
            ))
        );
        assert!(matches!(
            validate_ceaf_alignment(&[]),
            Err(BoundaryError::Input(_))
        ));
    }

    #[test]
    fn identifier_errors_preserve_the_frozen_field_names() {
        assert!(matches!(
            validate_identifiers("", "decision"),
            Err(BoundaryError::Input(message)) if message.contains("analysis_id")
        ));
        assert!(matches!(
            validate_identifiers("analysis", " "),
            Err(BoundaryError::Input(message)) if message.contains("decision_problem_id")
        ));
    }

    #[test]
    fn actual_python_indices_reject_negative_and_oversized_values() {
        Python::initialize();
        Python::attach(|py| {
            let valid = PyList::new(py, [0_i64, 2]).unwrap();
            assert_eq!(
                indices_from_python(valid.as_any(), "indices").unwrap(),
                vec![0, 2]
            );

            let builtins = py.import("builtins").unwrap();
            let pow = builtins.getattr("pow").unwrap();
            let cases = [
                (-1_i64).into_pyobject(py).unwrap().into_any(),
                pow.call1((2, 63)).unwrap(),
                pow.call1((2, 200)).unwrap(),
            ];
            for value in cases {
                let values = PyList::empty(py);
                values.append(value).unwrap();
                let error = indices_from_python(values.as_any(), "indices").unwrap_err();
                assert!(error.is_instance_of::<InputError>(py));
            }
        });
    }

    #[test]
    fn dominance_alignment_is_typed_without_parsing_messages() {
        assert!(matches!(
            validate_dominance_alignment(2, 1, 2, 2, 1, [0, 0, 0]),
            Err(BoundaryError::DimensionMismatch(_))
        ));
        assert!(matches!(
            validate_dominance_alignment(2, 2, 2, 2, 2, [0, 1, 1]),
            Err(BoundaryError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn embedded_build_metadata_satisfies_the_private_schema() {
        let metadata = build_metadata();
        assert_eq!(metadata.runtime_info_schema, 3);
        assert!(
            metadata.source_revision == "unknown" || is_lower_hex(metadata.source_revision, 40)
        );
        assert!(!metadata.target_triple.is_empty());
        assert!(metadata.rustc_version.starts_with("rustc "));
        assert!(!metadata.build_profile.is_empty());
        assert!(is_lower_hex(metadata.cargo_lock_sha256, 64));
        assert!(
            metadata.source_tree_git_oid == "unknown"
                || is_lower_hex(metadata.source_tree_git_oid, 40)
        );
        assert!(is_lower_hex(metadata.build_id, 64));
        assert!(is_lower_hex(metadata.source_state_sha256, 64));
        assert_eq!(
            metadata.source_state_algorithm,
            "git-diff-and-untracked-sha256-v1"
        );
        assert_eq!(metadata.build_id, build_id_for_metadata(&metadata));
        if let Some(epoch) = metadata.source_date_epoch {
            assert!(epoch >= 0);
        }
    }

    #[test]
    fn operation_telemetry_counts_attempts_outcomes_and_payloads() {
        let telemetry = OperationTelemetry::new();
        let payload = br#"{"schema_version":"1.0"}"#;

        telemetry.record_attempt();
        telemetry.record_success(payload);
        telemetry.record_attempt();
        telemetry.record_failure();

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.attempts, 2);
        assert_eq!(snapshot.successes, 1);
        assert_eq!(snapshot.failures, 1);
        assert_eq!(snapshot.calls, snapshot.successes);
        assert_eq!(snapshot.last_payload_sha256, Some(sha256_hex(payload)));
    }

    #[test]
    fn operation_telemetry_saturates_instead_of_wrapping() {
        let telemetry = OperationTelemetry::with_counts_for_test(u64::MAX, u64::MAX, u64::MAX);
        telemetry.record_attempt();
        telemetry.record_success(b"{}");
        telemetry.record_failure();

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.attempts, u64::MAX);
        assert_eq!(snapshot.successes, u64::MAX);
        assert_eq!(snapshot.failures, u64::MAX);
    }

    #[test]
    fn operation_telemetry_snapshots_are_thread_safe() {
        let telemetry = Arc::new(OperationTelemetry::new());
        let workers = 16;
        let iterations = 250;
        let handles = (0..workers)
            .map(|worker| {
                let telemetry = Arc::clone(&telemetry);
                thread::spawn(move || {
                    for iteration in 0..iterations {
                        telemetry.record_attempt();
                        if (worker + iteration) % 2 == 0 {
                            telemetry.record_success(b"{}");
                        } else {
                            telemetry.record_failure();
                        }
                        let snapshot = telemetry.snapshot();
                        assert!(snapshot.attempts >= snapshot.successes);
                        assert!(snapshot.attempts >= snapshot.failures);
                    }
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.attempts, workers * iterations);
        assert_eq!(snapshot.successes + snapshot.failures, workers * iterations);
        assert_eq!(snapshot.calls, snapshot.successes);
    }

    #[test]
    fn payload_digest_uses_versioned_rfc8785_canonical_json() {
        let payload = json!({
            "z": {
                "unicode": "€😀",
                "escaped": "quote: \" slash: \\ control:\n",
                "negative_zero": -0.0,
            },
            "numbers": [1e-7, 1e20],
            "a": {"second": 2, "first": 1},
        });

        let canonical = canonical_payload_bytes(&payload).unwrap();
        assert_eq!(
            std::str::from_utf8(&canonical).unwrap(),
            "{\"a\":{\"first\":1,\"second\":2},\"numbers\":[1e-7,100000000000000000000],\"z\":{\"escaped\":\"quote: \\\" slash: \\\\ control:\\n\",\"negative_zero\":0,\"unicode\":\"€😀\"}}"
        );
        assert_eq!(
            sha256_hex(&canonical),
            "be0884a551684f58df0932c3f9f8cf79c0760527dadb798b9844357cacf165d4"
        );
        assert_eq!(DIGEST_ALGORITHM, "rfc8785-sha256-v1");
    }

    #[test]
    fn payload_digest_is_invariant_to_recursive_mapping_order() {
        let left = json!({"outer": {"b": 2, "a": {"d": 4, "c": 3}}});
        let right = json!({"outer": {"a": {"c": 3, "d": 4}, "b": 2}});

        let left = canonical_payload_bytes(&left).unwrap();
        let right = canonical_payload_bytes(&right).unwrap();
        assert_eq!(left, right);
        assert_eq!(sha256_hex(&left), sha256_hex(&right));
    }

    #[test]
    fn runtime_info_identifies_the_digest_algorithm() {
        Python::initialize();
        Python::attach(|py| {
            let info = runtime_info(py).unwrap();
            assert_eq!(
                info.get_item("digest_algorithm")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                DIGEST_ALGORITHM
            );
            assert_eq!(
                info.get_item("build_id_algorithm")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                BUILD_ID_ALGORITHM
            );
            let operations = info
                .get_item("operations")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            for name in ["serialize_ceaf_result", "serialize_dominance_result"] {
                let operation = operations
                    .get_item(name)
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap();
                assert_eq!(
                    operation
                        .get_item("digest_algorithm")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    DIGEST_ALGORITHM
                );
            }
        });
    }

    #[test]
    fn pyo3_extraction_failures_precede_native_entry() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(serialize_ceaf_result, &module).unwrap())
                .unwrap();
            let function = module.getattr("serialize_ceaf_result").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("analysis_id", "analysis").unwrap();
            kwargs.set_item("decision_problem_id", "decision").unwrap();
            kwargs
                .set_item("wtp_thresholds", "not-a-float-list")
                .unwrap();
            kwargs
                .set_item("optimal_strategy_indices", [0_i64])
                .unwrap();
            kwargs
                .set_item("optimal_strategy_names", ["strategy"])
                .unwrap();
            kwargs
                .set_item("acceptability_probabilities", [0.5_f64])
                .unwrap();
            kwargs.set_item("probability_lower", [0.4_f64]).unwrap();
            kwargs.set_item("probability_upper", [0.6_f64]).unwrap();
            kwargs.set_item("expected_net_benefit", [1.0_f64]).unwrap();

            let before = CEAF_TELEMETRY.snapshot();
            assert!(function.call((), Some(&kwargs)).is_err());
            let after = CEAF_TELEMETRY.snapshot();
            assert_eq!(after.attempts, before.attempts);
            assert_eq!(after.successes, before.successes);
            assert_eq!(after.failures, before.failures);
        });
    }

    #[test]
    fn compute_evppi_executes_the_rust_kernel_for_python_sequences() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_evppi, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_evppi").unwrap();
            let result = function
                .call1((
                    vec![
                        vec![0.0_f64, 2.0],
                        vec![1.0, 0.0],
                        vec![2.0, 1.0],
                        vec![3.0, 4.0],
                    ],
                    vec![vec![0.0_f64], vec![1.0], vec![2.0], vec![3.0]],
                ))
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert!((result - 0.05).abs() <= 1.0e-10);
        });
    }

    #[test]
    fn compute_evpi_executes_the_rust_kernel_for_python_sequences() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_evpi, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_evpi").unwrap();
            let result = function
                .call1((vec![vec![0.0_f64, 2.0], vec![1.0, 0.0]],))
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert!((result - 0.5).abs() <= 1.0e-12);
        });
    }

    #[test]
    fn compute_enbs_executes_the_rust_kernel_for_python_scalars() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_enbs, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_enbs").unwrap();
            let result = function
                .call1((12.5_f64, 5.0_f64))
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert!((result - 7.5).abs() <= 1.0e-12);
        });
    }

    #[test]
    fn compute_dominance_executes_the_rust_kernel_for_python_sequences() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_dominance, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_dominance").unwrap();
            let result = function
                .call1((vec![1.0_f64, 2.0, 4.0], vec![1.0_f64, 2.0, 3.0]))
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let frontier = result
                .get_item("frontier_indices")
                .unwrap()
                .unwrap()
                .extract::<Vec<usize>>()
                .unwrap();
            assert_eq!(frontier, vec![0, 1, 2]);
        });
    }

    #[test]
    fn compute_ceaf_executes_the_rust_kernel_for_python_sequences() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_ceaf, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_ceaf").unwrap();
            let result = function
                .call1((
                    vec![
                        vec![vec![1.0_f64, 2.0], vec![2.0, 1.0]],
                        vec![vec![2.0, 1.0], vec![1.0, 3.0]],
                    ],
                    vec![0.0_f64, 1.0],
                    0.95_f64,
                ))
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let indices = result
                .get_item("optimal_strategy_indices")
                .unwrap()
                .unwrap()
                .extract::<Vec<usize>>()
                .unwrap();
            assert_eq!(indices, vec![0, 1]);
        });
    }

    #[test]
    fn compute_evsi_executes_the_seeded_kernel_for_python_sequences() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_evsi, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_evsi").unwrap();
            let result = function
                .call1((
                    vec![
                        vec![10.0_f64, 4.0],
                        vec![8.0, 6.0],
                        vec![6.0, 8.0],
                        vec![4.0, 10.0],
                    ],
                    2_usize,
                    4_usize,
                    42_u64,
                ))
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let evsi = result
                .get_item("evsi")
                .unwrap()
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert!((evsi - 0.75).abs() <= 1.0e-12);
        });
    }

    #[test]
    fn compute_evsi_efficient_linear_executes_the_rust_kernel_for_python_sequences() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_evsi_efficient_linear, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_evsi_efficient_linear").unwrap();
            let result = function
                .call1((
                    vec![
                        vec![0.0_f64, 6.0],
                        vec![2.0, 4.0],
                        vec![4.0, 2.0],
                        vec![6.0, 0.0],
                    ],
                    vec![vec![-1.0_f64], vec![0.0], vec![1.0], vec![2.0]],
                    2_usize,
                ))
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let evsi = result
                .get_item("evsi")
                .unwrap()
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert!((evsi - (2.0 / 3.0)).abs() <= 1.0e-12);
        });
    }

    #[test]
    fn compute_evsi_moment_based_executes_the_rust_kernel_for_python_sequences() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test").unwrap();
            module
                .add_function(wrap_pyfunction!(compute_evsi_moment_based, &module).unwrap())
                .unwrap();
            let function = module.getattr("compute_evsi_moment_based").unwrap();
            let result = function
                .call1((
                    vec![
                        vec![1.0_f64, 2.0],
                        vec![0.0, 3.0],
                        vec![1.0, 2.0],
                        vec![4.0, -1.0],
                    ],
                    vec![vec![-1.0_f64], vec![0.0], vec![1.0], vec![2.0]],
                    2_usize,
                ))
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let evsi = result
                .get_item("evsi")
                .unwrap()
                .unwrap()
                .extract::<f64>()
                .unwrap();
            assert!((evsi - (5.0 / 12.0)).abs() <= 1.0e-12);
        });
    }
}
