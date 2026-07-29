//! Leaf C ABI adapter for the voiage Rust core.
//!
//! The versioned C ABI exposes portable discovery and Rust-owned scalar kernels.

#![deny(unsafe_op_in_unsafe_fn)]

// SAFETY: error transport validates caller-owned pointers. Every unsafe block
// is locally documented and guarded by the v1 pointer contract.
#[allow(unsafe_code)]
mod error_transport;
// SAFETY: capability document transport validates caller-owned pointers and
// contains panics before copying a generated immutable byte slice.
#[allow(unsafe_code)]
mod capability_document;
// SAFETY: CEAF transport validates all caller-owned pointers, capacities, and
// dimensions, contains panics, and writes only after successful computation.
#[allow(unsafe_code)]
mod ceaf_result;
// SAFETY: dominance transport validates all caller-owned pointers, capacities,
// and addressable lengths, contains panics, and writes only after computation.
#[allow(unsafe_code)]
mod dominance_result;
// SAFETY: JSON transport validates caller-owned input/output regions, contains
// panics, and performs no partial document writes.
#[allow(unsafe_code)]
mod decision_problem_json;
mod generated_capabilities;
// SAFETY: shared JSON transport owns the checked raw-pointer copy boundary.
#[allow(unsafe_code)]
mod json_transport;
// SAFETY: expected-loss transport validates all caller-owned pointers and
// capacities, contains panics, and writes only after successful computation.
#[allow(unsafe_code)]
mod expected_loss_result;
// SAFETY: EVPPI transport validates both caller-owned matrices and the result,
// contains panics, and writes only after successful regression.
#[allow(unsafe_code)]
mod evppi_regression_result;
// SAFETY: EVPI JSON transport validates caller-owned input/output regions,
// contains panics, and performs no partial document writes.
#[allow(unsafe_code)]
mod evpi_result_json;
// SAFETY: EVSI transport validates both caller-owned matrices and the result,
// contains panics, and writes only after successful estimator execution.
#[allow(unsafe_code)]
mod evsi_approximation_result;
// SAFETY: lifecycle validates its sole caller-owned output pointer before the
// one documented write. Export wrappers contain panics before returning to C.
#[allow(unsafe_code)]
mod lifecycle;
// SAFETY: R adapters validate signed scalar pointers before delegating to the
// checked, panic-contained JSON transports.
#[allow(unsafe_code)]
mod r_json_adapter;
// SAFETY: scalar JSON wrappers delegate to the shared checked transport and
// contain panics.
#[allow(unsafe_code)]
mod scalar_result_json;
// SAFETY: statistical-assurance JSON transport delegates to the shared
// checked transport and contains panics.
#[allow(unsafe_code)]
mod statistical_assurance_json;
mod status;
// SAFETY: structured JSON wrappers delegate to the shared checked transport
// and contain panics.
#[allow(unsafe_code)]
mod structured_result_json;
// SAFETY: structural VOI transport validates caller-owned cube, probability,
// index, and result pointers, contains panics, and writes only after success.
#[allow(unsafe_code)]
mod structural_result;

use std::panic::{self, AssertUnwindSafe};

use voiage_domain::SampleMatrix;
use voiage_numerics::{enbs, evpi, evpi_with_assurance, EvpiKernelResult};

pub use capability_document::voiage_v1_capabilities_json;
pub use ceaf_result::{voiage_v1_ceaf_result, VoiageCeafResultV1};
pub use decision_problem_json::voiage_v1_decision_problem_json;
pub use dominance_result::{voiage_v1_dominance_result, VoiageDominanceResultV1};
pub use error_transport::voiage_v1_error_message;
pub use evpi_result_json::voiage_v1_evpi_result_json;
pub use evppi_regression_result::{
    voiage_v1_evppi_regression_result, VoiageEvppiRegressionResultV1,
    VOIAGE_EVPPI_ASSURANCE_INCOMPLETE,
};
pub use evsi_approximation_result::{
    voiage_v1_evsi_moment_matching_result, voiage_v1_evsi_regression_result,
    VoiageEvsiApproximationResultV1, VOIAGE_EVSI_ASSURANCE_INCOMPLETE,
    VOIAGE_EVSI_ESTIMATOR_MOMENT_MATCHING, VOIAGE_EVSI_ESTIMATOR_REGRESSION,
};
pub use expected_loss_result::{voiage_v1_expected_loss_result, VoiageExpectedLossResultV1};
pub use lifecycle::{voiage_v1_handle_create, voiage_v1_handle_free};
pub use r_json_adapter::{
    voiage_v1_decision_problem_json_i32_r, voiage_v1_statistical_assurance_json_i32_r,
};
pub use scalar_result_json::{
    voiage_v1_enbs_result_json, voiage_v1_evppi_result_json, voiage_v1_evsi_result_json,
};
pub use statistical_assurance_json::voiage_v1_statistical_assurance_json;
pub use status::VoiageStatusV1;
pub use structural_result::{
    voiage_v1_structural_evpi_result, voiage_v1_structural_evppi_result,
    VoiageStructuralVoiResultV1,
};
pub use structured_result_json::{
    voiage_v1_ceaf_result_json, voiage_v1_dominance_result_json,
    voiage_v1_expected_loss_result_json,
};

/// Identifies this crate while the versioned C ABI is introduced.
pub const CRATE_NAME: &str = "voiage-ffi";

/// ABI major version implemented by the `voiage_v1_*` symbol namespace.
pub const VOIAGE_V1_ABI_MAJOR: u32 = 1;

/// Backwards-compatible ABI minor version implemented by this library.
pub const VOIAGE_V1_ABI_MINOR: u32 = 15;

/// Capability bit for ABI version negotiation.
pub const VOIAGE_ABI_VERSION_NEGOTIATION: u64 = 1 << 0;

/// Capability bit for infrastructure capability discovery.
pub const VOIAGE_ABI_CAPABILITY_QUERY: u64 = 1 << 1;

/// Capability bit for the stable scalar EVPI operation.
pub const VOIAGE_ABI_EVPI: u64 = 1 << 2;

/// Capability bit for the typed EVPI result envelope.
pub const VOIAGE_ABI_EVPI_RESULT: u64 = 1 << 3;

/// Capability bit for the registry-generated JSON capability document.
pub const VOIAGE_ABI_CAPABILITY_DOCUMENT: u64 = 1 << 4;

/// Capability bit for typed expected-loss results and caller-owned arrays.
pub const VOIAGE_ABI_EXPECTED_LOSS_RESULT: u64 = 1 << 5;

/// Capability bit for Rust-authoritative expected net benefit of sampling.
pub const VOIAGE_ABI_ENBS: u64 = 1 << 6;

/// Capability bit for deterministic dominance, frontier, and ICER results.
pub const VOIAGE_ABI_DOMINANCE_RESULT: u64 = 1 << 7;

/// Capability bit for threshold-aligned CEAF and assurance results.
pub const VOIAGE_ABI_CEAF_RESULT: u64 = 1 << 8;

/// Capability bit for structural and model-form VOI results.
pub const VOIAGE_ABI_STRUCTURAL_VOI_RESULT: u64 = 1 << 9;

/// Capability bit for the stable full-sample linear-regression EVPPI result.
pub const VOIAGE_ABI_EVPPI_REGRESSION_RESULT: u64 = 1 << 10;

/// Capability bit for promoted Rust-native EVSI approximation results.
pub const VOIAGE_ABI_EVSI_APPROXIMATION_RESULT: u64 = 1 << 11;

/// Capability bit for validated Decision Problem JSON transport.
pub const VOIAGE_ABI_DECISION_PROBLEM_JSON: u64 = 1 << 12;

/// Capability bit for schema-validated EVPI result JSON transport.
pub const VOIAGE_ABI_EVPI_RESULT_JSON: u64 = 1 << 13;

/// Capability bit for EVPPI, EVSI, and ENBS result JSON transport.
pub const VOIAGE_ABI_SCALAR_RESULT_JSON: u64 = 1 << 14;

/// Capability bit for expected-loss, CEAF, and dominance result JSON transport.
pub const VOIAGE_ABI_STRUCTURED_RESULT_JSON: u64 = 1 << 15;

/// Capability bit for statistical-assurance envelope JSON transport.
pub const VOIAGE_ABI_STATISTICAL_ASSURANCE_JSON: u64 = 1 << 16;

const ABI_VERSION_STRUCT_SIZE: u32 = 12;
const ABI_CAPABILITIES_STRUCT_SIZE: u32 = 16;
const EVPI_RESULT_STRUCT_SIZE: u32 = 56;

/// Fixed-width, self-describing v1 ABI version response.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoiageAbiVersionV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// ABI major version. Breaking changes require a new symbol namespace.
    pub abi_major: u32,
    /// ABI minor version. Additive compatible changes increment this value.
    pub abi_minor: u32,
}

/// Fixed-width, self-describing v1 capability response.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoiageAbiCapabilitiesV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this capability response structure.
    pub struct_version: u32,
    /// Capability bitset for supported ABI operations.
    pub capability_bits: u64,
}

/// Fixed-width typed EVPI result with explicit assurance availability.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiageEvpiResultV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this result structure.
    pub struct_version: u32,
    /// Expected value of perfect information.
    pub value: f64,
    /// Number of uncertainty samples.
    pub sample_count: u64,
    /// Number of strategies.
    pub strategy_count: u64,
    /// One when variance and Monte Carlo error are available, otherwise zero.
    pub has_assurance: u32,
    /// Reserved; callers must ignore and producers set to zero.
    pub reserved: u32,
    /// Unbiased selected-strategy opportunity-loss variance.
    pub opportunity_loss_variance: f64,
    /// Monte Carlo standard error of EVPI.
    pub monte_carlo_standard_error: f64,
}

/// Returns the portable v1 ABI version contract.
// SAFETY: the export attribute is the only unsafe-code lint exception. The
// function has no pointers, mutable state, allocation, or numerical behavior.
#[allow(unsafe_code)]
#[no_mangle]
pub extern "C" fn voiage_v1_abi_version() -> VoiageAbiVersionV1 {
    VoiageAbiVersionV1 {
        struct_size: ABI_VERSION_STRUCT_SIZE,
        abi_major: VOIAGE_V1_ABI_MAJOR,
        abi_minor: VOIAGE_V1_ABI_MINOR,
    }
}

/// Returns the ABI capabilities implemented by this library.
// SAFETY: the export attribute is the only unsafe-code lint exception. The
// function returns a fixed-width value and does not access caller memory.
#[allow(unsafe_code)]
#[no_mangle]
pub extern "C" fn voiage_v1_capabilities() -> VoiageAbiCapabilitiesV1 {
    VoiageAbiCapabilitiesV1 {
        struct_size: ABI_CAPABILITIES_STRUCT_SIZE,
        struct_version: 1,
        capability_bits: VOIAGE_ABI_VERSION_NEGOTIATION
            | VOIAGE_ABI_CAPABILITY_QUERY
            | VOIAGE_ABI_EVPI
            | VOIAGE_ABI_EVPI_RESULT
            | VOIAGE_ABI_CAPABILITY_DOCUMENT
            | VOIAGE_ABI_EXPECTED_LOSS_RESULT
            | VOIAGE_ABI_ENBS
            | VOIAGE_ABI_DOMINANCE_RESULT
            | VOIAGE_ABI_CEAF_RESULT
            | VOIAGE_ABI_STRUCTURAL_VOI_RESULT
            | VOIAGE_ABI_EVPPI_REGRESSION_RESULT
            | VOIAGE_ABI_EVSI_APPROXIMATION_RESULT
            | VOIAGE_ABI_DECISION_PROBLEM_JSON
            | VOIAGE_ABI_EVPI_RESULT_JSON
            | VOIAGE_ABI_SCALAR_RESULT_JSON
            | VOIAGE_ABI_STRUCTURED_RESULT_JSON
            | VOIAGE_ABI_STATISTICAL_ASSURANCE_JSON,
    }
}

pub(crate) fn checked_dimensions(
    rows: u64,
    columns: u64,
) -> Result<(usize, usize, usize), VoiageStatusV1> {
    if rows == 0 || columns == 0 {
        return Err(VoiageStatusV1::InvalidArgument);
    }
    let length = rows
        .checked_mul(columns)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(VoiageStatusV1::InvalidArgument)?;
    if length > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(VoiageStatusV1::InvalidArgument);
    }
    let row_count = usize::try_from(rows).map_err(|_| VoiageStatusV1::InvalidArgument)?;
    let column_count = usize::try_from(columns).map_err(|_| VoiageStatusV1::InvalidArgument)?;
    Ok((row_count, column_count, length))
}

pub(crate) fn validated_matrix(
    values: &[f64],
    row_count: usize,
    column_count: usize,
) -> Result<SampleMatrix, VoiageStatusV1> {
    let matrix = (0..row_count)
        .map(|row| {
            let start = row * column_count;
            values[start..start + column_count].to_vec()
        })
        .collect::<Vec<_>>();
    SampleMatrix::try_from(matrix).map_err(|_| VoiageStatusV1::InvalidArgument)
}

/// Computes EVPI from a row-major, finite net-benefit matrix.
///
/// # Safety
///
/// `values` must point to `rows * columns` readable `f64` values and `out`
/// must be non-null, aligned, and writable for one `f64`. Neither pointer is
/// retained after this call.
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evpi(
    values: *const f64,
    rows: u64,
    columns: u64,
    out: *mut f64,
) -> VoiageStatusV1 {
    if values.is_null()
        || out.is_null()
        || rows == 0
        || columns == 0
        || (values as usize) % std::mem::align_of::<f64>() != 0
        || (out as usize) % std::mem::align_of::<f64>() != 0
    {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok((row_count, column_count, length)) = checked_dimensions(rows, columns) else {
        return VoiageStatusV1::InvalidArgument;
    };
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller contract guarantees a readable row-major region.
        let slice = unsafe { std::slice::from_raw_parts(values, length) };
        let matrix = validated_matrix(slice, row_count, column_count)?;
        evpi(&matrix).map_err(|_| VoiageStatusV1::NumericalFailure)
    }));
    match result {
        Ok(Ok(value)) => {
            // SAFETY: nullness and alignment were validated above.
            unsafe { out.write(value) };
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

/// Computes raw expected net benefit of sampling as `EVSI - research cost`.
///
/// Negative results are valid and are not clipped.
///
/// # Safety
///
/// `out` must be non-null, aligned, and writable for one `f64`. It is not
/// retained. Invalid inputs and contained panics do not write output.
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_enbs(
    evsi_result: f64,
    research_cost: f64,
    out: *mut f64,
) -> VoiageStatusV1 {
    if out.is_null() || (out as usize) % std::mem::align_of::<f64>() != 0 {
        return VoiageStatusV1::InvalidArgument;
    }
    match panic::catch_unwind(AssertUnwindSafe(|| enbs(evsi_result, research_cost))) {
        Ok(Ok(value)) => {
            // SAFETY: nullness and alignment were validated above.
            unsafe { out.write(value) };
            VoiageStatusV1::Ok
        }
        Ok(Err(_)) => VoiageStatusV1::InvalidArgument,
        Err(_) => VoiageStatusV1::Panic,
    }
}

/// Computes EVPI into a self-describing typed result envelope.
///
/// # Safety
///
/// `values` must point to `rows * columns` readable `f64` values and `out`
/// must be non-null, aligned, and writable for one [`VoiageEvpiResultV1`].
/// Neither pointer is retained after this call.
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evpi_result(
    values: *const f64,
    rows: u64,
    columns: u64,
    out: *mut VoiageEvpiResultV1,
) -> VoiageStatusV1 {
    if values.is_null()
        || out.is_null()
        || rows == 0
        || columns == 0
        || (values as usize) % std::mem::align_of::<f64>() != 0
        || (out as usize) % std::mem::align_of::<VoiageEvpiResultV1>() != 0
    {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok((row_count, column_count, length)) = checked_dimensions(rows, columns) else {
        return VoiageStatusV1::InvalidArgument;
    };
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller contract guarantees a readable row-major region.
        let slice = unsafe { std::slice::from_raw_parts(values, length) };
        let matrix = validated_matrix(slice, row_count, column_count)?;
        evpi_with_assurance(&matrix).map_err(|_| VoiageStatusV1::NumericalFailure)
    }));
    match result {
        Ok(Ok(result)) => {
            let envelope = evpi_result_envelope(&result);
            // SAFETY: nullness and alignment were validated above.
            unsafe { out.write(envelope) };
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

fn evpi_result_envelope(result: &EvpiKernelResult) -> VoiageEvpiResultV1 {
    let assurance = result
        .opportunity_loss_variance
        .zip(result.monte_carlo_standard_error);
    VoiageEvpiResultV1 {
        struct_size: EVPI_RESULT_STRUCT_SIZE,
        struct_version: 1,
        value: result.value,
        sample_count: result.sample_count as u64,
        strategy_count: result.strategy_count as u64,
        has_assurance: u32::from(assurance.is_some()),
        reserved: 0,
        opportunity_loss_variance: assurance.map_or(0.0, |values| values.0),
        monte_carlo_standard_error: assurance.map_or(0.0, |values| values.1),
    }
}

/// Computes EVPI with signed 32-bit dimensions for runtimes such as base R
/// whose `.C` interface does not expose a portable unsigned 64-bit scalar.
///
/// # Safety
///
/// The pointer requirements are identical to [`voiage_v1_evpi`].
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evpi_i32(
    values: *const f64,
    rows: i32,
    columns: i32,
    out: *mut f64,
) -> VoiageStatusV1 {
    if rows <= 0 || columns <= 0 {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok(rows) = u64::try_from(rows) else {
        return VoiageStatusV1::InvalidArgument;
    };
    let Ok(columns) = u64::try_from(columns) else {
        return VoiageStatusV1::InvalidArgument;
    };
    // SAFETY: this adapter preserves the pointer contract of voiage_v1_evpi.
    unsafe { voiage_v1_evpi(values, rows, columns, out) }
}

/// Calls [`voiage_v1_evpi_i32`] and writes its status for `.C` runtimes that
/// cannot observe a C return value.
///
/// # Safety
///
/// `values` and `out_value` follow [`voiage_v1_evpi`] requirements, and
/// `out_status` must be non-null, aligned, and writable for one `i32`.
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evpi_i32_r(
    values: *const f64,
    rows: *const i32,
    columns: *const i32,
    out_value: *mut f64,
    out_status: *mut i32,
) {
    if rows.is_null()
        || columns.is_null()
        || out_status.is_null()
        || (rows as usize) % std::mem::align_of::<i32>() != 0
        || (columns as usize) % std::mem::align_of::<i32>() != 0
        || (out_status as usize) % std::mem::align_of::<i32>() != 0
    {
        return;
    }
    // SAFETY: nullness and alignment were validated above.
    let (rows, columns) = unsafe { (rows.read(), columns.read()) };
    // SAFETY: the caller contract validates out_status above; the delegated
    // operation validates the remaining pointers.
    let status = unsafe { voiage_v1_evpi_i32(values, rows, columns, out_value) };
    // SAFETY: nullness and alignment were validated above.
    unsafe { out_status.write(status.as_i32()) };
}
