//! Structural and model-form VOI C ABI results.

use std::panic::{self, AssertUnwindSafe};

use voiage_domain::{SampleCube, SampleVector};
use voiage_numerics::{
    structural_evpi_with_assurance, structural_evppi_with_assurance, StructuralVoiKernelResult,
};

use crate::VoiageStatusV1;

const STRUCTURAL_VOI_RESULT_STRUCT_SIZE: u32 = 64;

/// Fixed-width structural VOI result with sample-average assurance.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiageStructuralVoiResultV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this result structure.
    pub struct_version: u32,
    /// Structural EVPI or structural EVPPI estimate.
    pub value: f64,
    /// Number of model structures.
    pub structure_count: u64,
    /// Number of uncertainty samples per structure.
    pub sample_count: u64,
    /// Number of strategies.
    pub strategy_count: u64,
    /// One when variance and Monte Carlo error are available, otherwise zero.
    pub has_assurance: u32,
    /// Reserved; callers must ignore and producers set to zero.
    pub reserved: u32,
    /// Unbiased variance of the sample-level informed value.
    pub informed_value_variance: f64,
    /// Monte Carlo standard error of the structural VOI estimate.
    pub monte_carlo_standard_error: f64,
}

/// Computes structural EVPI from row-major `[structure][sample][strategy]`
/// net benefit and one probability per structure.
///
/// # Safety
///
/// Inputs and output must be non-null, correctly aligned, readable or writable
/// for their declared lengths, and mutually non-overlapping. No pointer is
/// retained, and output is not written on failure.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_structural_evpi_result(
    values: *const f64,
    structure_count: u64,
    sample_count: u64,
    strategy_count: u64,
    structure_probabilities: *const f64,
    out_result: *mut VoiageStructuralVoiResultV1,
) -> VoiageStatusV1 {
    if !base_pointers_valid(values, structure_probabilities, out_result) {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok(dimensions) = checked_dimensions(structure_count, sample_count, strategy_count) else {
        return VoiageStatusV1::InvalidArgument;
    };
    let computed = panic::catch_unwind(AssertUnwindSafe(|| {
        let (cube, probabilities) = read_inputs(values, structure_probabilities, dimensions)?;
        structural_evpi_with_assurance(&cube, &probabilities)
            .map_err(|_| VoiageStatusV1::InvalidArgument)
    }));
    finish(computed, out_result)
}

/// Computes structural EVPPI for caller-selected structure indices.
///
/// A null `structures_of_interest` pointer is valid only when
/// `structures_of_interest_count` is zero.
///
/// # Safety
///
/// Inputs and output must be correctly aligned, readable or writable for their
/// declared lengths, and mutually non-overlapping. No pointer is retained, and
/// output is not written on failure.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_structural_evppi_result(
    values: *const f64,
    structure_count: u64,
    sample_count: u64,
    strategy_count: u64,
    structure_probabilities: *const f64,
    structures_of_interest: *const u64,
    structures_of_interest_count: u64,
    out_result: *mut VoiageStructuralVoiResultV1,
) -> VoiageStatusV1 {
    if !base_pointers_valid(values, structure_probabilities, out_result)
        || !optional_indices_pointer_valid(structures_of_interest, structures_of_interest_count)
    {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok(dimensions) = checked_dimensions(structure_count, sample_count, strategy_count) else {
        return VoiageStatusV1::InvalidArgument;
    };
    let Ok(index_count) = checked_index_count(structures_of_interest_count) else {
        return VoiageStatusV1::InvalidArgument;
    };
    let computed = panic::catch_unwind(AssertUnwindSafe(|| {
        let (cube, probabilities) = read_inputs(values, structure_probabilities, dimensions)?;
        let raw_indices = if index_count == 0 {
            &[]
        } else {
            // SAFETY: pointer validity and addressable length were validated.
            unsafe { std::slice::from_raw_parts(structures_of_interest, index_count) }
        };
        let indices = raw_indices
            .iter()
            .map(|&index| usize::try_from(index).map_err(|_| VoiageStatusV1::InvalidArgument))
            .collect::<Result<Vec<_>, _>>()?;
        structural_evppi_with_assurance(&cube, &probabilities, &indices)
            .map_err(|_| VoiageStatusV1::InvalidArgument)
    }));
    finish(computed, out_result)
}

type Dimensions = (usize, usize, usize, usize);

fn checked_dimensions(
    structure_count: u64,
    sample_count: u64,
    strategy_count: u64,
) -> Result<Dimensions, VoiageStatusV1> {
    if structure_count == 0 || sample_count == 0 || strategy_count == 0 {
        return Err(VoiageStatusV1::InvalidArgument);
    }
    let value_count = structure_count
        .checked_mul(sample_count)
        .and_then(|count| count.checked_mul(strategy_count))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(VoiageStatusV1::InvalidArgument)?;
    if value_count > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(VoiageStatusV1::InvalidArgument);
    }
    Ok((
        usize::try_from(structure_count).map_err(|_| VoiageStatusV1::InvalidArgument)?,
        usize::try_from(sample_count).map_err(|_| VoiageStatusV1::InvalidArgument)?,
        usize::try_from(strategy_count).map_err(|_| VoiageStatusV1::InvalidArgument)?,
        value_count,
    ))
}

fn checked_index_count(count: u64) -> Result<usize, VoiageStatusV1> {
    let count = usize::try_from(count).map_err(|_| VoiageStatusV1::InvalidArgument)?;
    if count > isize::MAX as usize / std::mem::size_of::<u64>() {
        return Err(VoiageStatusV1::InvalidArgument);
    }
    Ok(count)
}

fn base_pointers_valid(
    values: *const f64,
    probabilities: *const f64,
    out_result: *mut VoiageStructuralVoiResultV1,
) -> bool {
    !values.is_null()
        && !probabilities.is_null()
        && !out_result.is_null()
        && (values as usize) % std::mem::align_of::<f64>() == 0
        && (probabilities as usize) % std::mem::align_of::<f64>() == 0
        && (out_result as usize) % std::mem::align_of::<VoiageStructuralVoiResultV1>() == 0
}

fn optional_indices_pointer_valid(indices: *const u64, count: u64) -> bool {
    (count == 0 && indices.is_null())
        || (!indices.is_null() && (indices as usize) % std::mem::align_of::<u64>() == 0)
}

fn read_inputs(
    values: *const f64,
    structure_probabilities: *const f64,
    (structures, samples, strategies, value_count): Dimensions,
) -> Result<(SampleCube, SampleVector), VoiageStatusV1> {
    // SAFETY: pointer validity and addressable length were validated; the
    // caller guarantees a readable, non-overlapping input region.
    let flat_values = unsafe { std::slice::from_raw_parts(values, value_count) };
    // SAFETY: structure count is non-zero and bounded by the value dimensions.
    let probability_values =
        unsafe { std::slice::from_raw_parts(structure_probabilities, structures) };
    let mut offset = 0;
    let cube_values = (0..structures)
        .map(|_| {
            (0..samples)
                .map(|_| {
                    let row = flat_values[offset..offset + strategies].to_vec();
                    offset += strategies;
                    row
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cube = SampleCube::try_from(cube_values).map_err(|_| VoiageStatusV1::InvalidArgument)?;
    let probabilities = SampleVector::try_from(probability_values.to_vec())
        .map_err(|_| VoiageStatusV1::InvalidArgument)?;
    Ok((cube, probabilities))
}

fn finish(
    computed: Result<
        Result<StructuralVoiKernelResult, VoiageStatusV1>,
        Box<dyn std::any::Any + Send>,
    >,
    out_result: *mut VoiageStructuralVoiResultV1,
) -> VoiageStatusV1 {
    match computed {
        Ok(Ok(result)) => {
            let has_assurance = result.informed_value_variance.is_some()
                && result.monte_carlo_standard_error.is_some();
            let envelope = VoiageStructuralVoiResultV1 {
                struct_size: STRUCTURAL_VOI_RESULT_STRUCT_SIZE,
                struct_version: 1,
                value: result.value,
                structure_count: result.structure_count as u64,
                sample_count: result.sample_count as u64,
                strategy_count: result.strategy_count as u64,
                has_assurance: u32::from(has_assurance),
                reserved: 0,
                informed_value_variance: result.informed_value_variance.unwrap_or(0.0),
                monte_carlo_standard_error: result.monte_carlo_standard_error.unwrap_or(0.0),
            };
            // SAFETY: output nullness and alignment were validated before
            // computation, and no write occurs on any failure path.
            unsafe { out_result.write(envelope) };
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}
