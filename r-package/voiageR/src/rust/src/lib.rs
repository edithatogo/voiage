//! Dependency-free Rust kernels linked into the `voiageR` source package.
//!
//! Deterministic `#![no_std]` vendored authority aligned with
//! `rust/crates/voiage-numerics/src/{evpi,enbs}.rs` and verified by
//! `tests/test_q02_rust_authority_parity.py`.

#![no_std]

use core::{panic::PanicInfo, slice};

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 1;

#[inline]
fn neumaier_step(sum: f64, comp: f64, value: f64) -> (f64, f64) {
    let next = sum + value;
    let abs_sum = if sum < 0.0 { -sum } else { sum };
    let abs_val = if value < 0.0 { -value } else { value };
    let next_comp = if abs_sum >= abs_val {
        comp + ((sum - next) + value)
    } else {
        comp + ((value - next) + sum)
    };
    (next, next_comp)
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

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

    let divisor = f64::from(rows);
    let columns = columns as usize;
    let mut perfect_sum = 0.0;
    let mut perfect_comp = 0.0;
    for row in samples.chunks_exact(columns) {
        let mut row_best = f64::NEG_INFINITY;
        for value in row.iter().copied() {
            row_best = row_best.max(value);
        }
        (perfect_sum, perfect_comp) = neumaier_step(perfect_sum, perfect_comp, row_best / divisor);
    }
    let expected_perfect = perfect_sum + perfect_comp;
    if !expected_perfect.is_finite() {
        return INVALID_ARGUMENT;
    }
    let mut current_best = f64::NEG_INFINITY;
    for column in 0..columns {
        let mut col_sum = 0.0;
        let mut col_comp = 0.0;
        for value in samples.iter().skip(column).step_by(columns).copied() {
            (col_sum, col_comp) = neumaier_step(col_sum, col_comp, value / divisor);
        }
        let col_mean = col_sum + col_comp;
        if !col_mean.is_finite() {
            return INVALID_ARGUMENT;
        }
        current_best = current_best.max(col_mean);
    }
    let diff = expected_perfect - current_best;
    if !current_best.is_finite() || !diff.is_finite() {
        return INVALID_ARGUMENT;
    }
    let result = diff.max(0.0);
    // SAFETY: the caller contract supplies one writable output double.
    unsafe { out.write(result) };
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

/// Compute population ENBS as `per_patient_evsi * effective_population - research_cost`.
///
/// # Safety
///
/// `out` must identify one writable double and is never retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_rust_population_enbs(
    per_patient_evsi: f64,
    research_cost: f64,
    effective_population: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null()
        || !per_patient_evsi.is_finite()
        || per_patient_evsi < 0.0
        || !research_cost.is_finite()
        || research_cost < 0.0
        || !effective_population.is_finite()
        || effective_population < 0.0
    {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the caller contract supplies one writable output double.
    unsafe { out.write(per_patient_evsi * effective_population - research_cost) };
    OK
}
