//! Installed-consumer tests for the supported Rust facade.

use voiage::domain::SampleMatrix;

#[test]
fn public_facade_executes_the_stable_evpi_kernel() {
    let values: SampleMatrix = vec![vec![10.0, 4.0], vec![2.0, 8.0]]
        .try_into()
        .expect("valid matrix");

    let result = voiage::numerics::evpi(&values).expect("EVPI");

    assert!((result - 3.0).abs() < f64::EPSILON);
    assert_eq!(voiage::VERSION, env!("CARGO_PKG_VERSION"));
}

#[test]
fn public_facade_keeps_binding_adapters_out_of_the_api() {
    let manifest = include_str!("../Cargo.toml");

    assert!(!manifest.contains("voiage-ffi"));
    assert!(!manifest.contains("voiage-python"));
    assert!(!manifest.contains("pyo3"));
}
