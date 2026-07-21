//! WebAssembly adapter exposing Rust-owned scalar kernels to JavaScript.

#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

use voiage_domain::SampleMatrix;
use voiage_numerics::evpi as rust_evpi;

/// Identifies this crate while the WebAssembly adapter is introduced.
pub const CRATE_NAME: &str = "voiage-wasm";

/// Computes EVPI from a row-major JavaScript `Float64Array`.
#[wasm_bindgen]
pub fn evpi(values: &[f64], rows: u32, columns: u32) -> Result<f64, JsValue> {
    if rows == 0 || columns == 0 {
        return Err(JsValue::from_str("rows and columns must be positive"));
    }
    let rows = usize::try_from(rows).map_err(|_| JsValue::from_str("rows are too large"))?;
    let columns =
        usize::try_from(columns).map_err(|_| JsValue::from_str("columns are too large"))?;
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| JsValue::from_str("matrix dimensions overflow"))?;
    if values.len() != expected {
        return Err(JsValue::from_str("matrix dimensions do not match values"));
    }
    let matrix = (0..rows)
        .map(|row| values[row * columns..(row + 1) * columns].to_vec())
        .collect::<Vec<_>>();
    let matrix = SampleMatrix::try_from(matrix)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    rust_evpi(&matrix).map_err(|error| JsValue::from_str(&error.to_string()))
}
