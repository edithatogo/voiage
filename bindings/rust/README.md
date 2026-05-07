# voiage-core

Canonical Rust engine for the voiage core API contract and domain model.

## Domain model boundary

`voiage-core` now exposes the stable Rust data model that downstream numerics
and adapter work should target:

- `ValueArray`, `ParameterSet`, `TrialDesign`, and `TrialArm` for the core
  containers
- `Diagnostics`, `DiagnosticWarning`, `MethodMetadata`, and `Reporting` for
  result metadata and CHEERS-style payloads
- typed result envelopes such as `AnalysisEnvelope<T>`, `EvpiSummary`,
  `EvppiSummary`, `EvsiSummary`, and `EnbsSummary`

The intended boundary is container- and metadata-first. Future numerics work
should build on these types rather than inventing new wire shapes. Adapter
crates should translate language-native inputs into these Rust structs, then
serialize or forward the resulting envelopes without changing field names or
ordering-sensitive collections.

## Setup

From `bindings/rust/`:

```bash
cargo test --locked
cargo doc --no-deps --locked
cargo package --locked --allow-dirty
```

## First workflow

```rust
use voiage_core::evpi;

fn main() -> Result<(), &'static str> {
    let evpi_value = evpi(&[vec![10.0, 1.0], vec![2.0, 8.0]])?;
    println!("EVPI: {:.1}", evpi_value);
    Ok(())
}
```

This returns `EVPI: 3.0` for the simple two-strategy matrix above. The Rust
crate owns the core calculation policy and domain model; the other bindings
should stay thin adapters over this engine and reuse the same serialized
containers and result envelopes.

## Scalar CPU baseline

The `benches/` directory contains the lightweight scalar CPU baseline for the
Rust core performance track. It uses the same deterministic EVPI workload as
the library tests and is intended as a local baseline before Criterion-style
benchmarking, CI thresholds, or parallel variants are added.

The EVSI boundary is split in two on purpose: the stable summary contract
already lives in Rust core, while the stochastic kernel is tracked separately
as follow-on work. Any future EVSI approximation policy should stay under the
kernel boundary and preserve the same summary/reporting envelope.

## Release and caveats

The release workflow is triggered by `rust-v*` tags, validates the package
with `cargo fmt`, `cargo clippy`, `cargo test --locked`, `cargo doc --no-deps
--locked`, and `cargo package --locked --allow-dirty`, and publishes to
crates.io when a `CARGO_REGISTRY_TOKEN` secret is configured. It also attaches
a source archive to the GitHub release for traceability. Rust is the canonical
engine in the Rust-core migration track, so this crate should stay thin and
policy-neutral: it exposes the core calculation and domain-model semantics
rather than duplicating binding-specific behavior.

## Handoff notes for numerics and interop

- Keep the field layout deterministic and serde-friendly.
- Keep collection ordering stable where possible; use `BTreeMap` for named
  payload maps.
- Preserve the validation constructors as the source of truth for invalid
  shapes, ragged matrices, duplicate IDs, and non-finite values.
- Treat adapter crates as translation layers only; they should not redefine
  the core container or reporting shapes.
- See [the Rust core handoff guide](../../docs/developer_guide/rust_core_handoff.rst)
  for the canonical boundary rules on what stays in Rust core versus what
  remains in bindings or adapters.
