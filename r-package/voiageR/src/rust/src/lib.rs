//! Dependency-free Rust kernels linked into the `voiageR` source package.

use std::slice;

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 1;

/// Compute EVPI for a finite row-major sample matrix.
///
/// @srrstats {G3.0} Floating-point results are compared only through explicit
/// tolerances in the shared numerical-reference tests.
///
/// # Safety
///
/// `values` must identify `rows * columns` readable doubles and `out` must
/// identify one writable double. The pointers are never retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_rust_evpi(
    values: *const f64,
    rows: i32,
    columns: i32,
    out: *mut f64,
) -> i32 {
    if values.is_null() || out.is_null() || rows <= 0 || columns <= 0 {
        return INVALID_ARGUMENT;
    }
    let Some(length) = (rows as usize).checked_mul(columns as usize) else {
        return INVALID_ARGUMENT;
    };
    // SAFETY: the caller contract supplies a readable matrix of this length.
    let samples = unsafe { slice::from_raw_parts(values, length) };
    if samples.iter().any(|value| !value.is_finite()) {
        return INVALID_ARGUMENT;
    }

    let rows = rows as usize;
    let columns = columns as usize;
    let mut expected_perfect = 0.0;
    let mut column_sums = vec![0.0; columns];
    for row in samples.chunks_exact(columns) {
        let mut row_best = f64::NEG_INFINITY;
        for (column, value) in row.iter().copied().enumerate() {
            row_best = row_best.max(value);
            column_sums[column] += value;
        }
        expected_perfect += row_best;
    }
    expected_perfect /= rows as f64;
    let current_best = column_sums
        .into_iter()
        .map(|sum| sum / rows as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    // SAFETY: the caller contract supplies one writable output double.
    unsafe { out.write(expected_perfect - current_best) };
    OK
}

/// Compute raw ENBS as `EVSI - research cost`.
///
/// # Safety
///
/// `out` must identify one writable double and is never retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_rust_enbs(
    evsi_result: f64,
    research_cost: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null()
        || !evsi_result.is_finite()
        || !research_cost.is_finite()
        || research_cost < 0.0
    {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the caller contract supplies one writable output double.
    unsafe { out.write(evsi_result - research_cost) };
    OK
}
