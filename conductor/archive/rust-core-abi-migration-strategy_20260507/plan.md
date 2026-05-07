# Track Strategy Findings: Rust Core ABI And Migration Strategy

## Finding 1: Inventory Current API Surfaces [checkpoint: strategy-complete]

- [x] Stable public contracts identified: core domain types, diagnostics,
  reporting envelopes, and typed result surfaces.
- [x] Repo roles mapped: Rust core owns the contract and numeric kernels;
  Python remains the façade; language bindings remain adapters and packaging
  layers.
- [x] Review checkpoint closed as a strategy finding.

## Finding 2: Decide The ABI Strategy [checkpoint: strategy-complete]

- [x] Recommendation settled on pure Rust modularization first, with only a
  narrow C ABI as an optional edge for selected native consumers.
- [x] TypeScript should prefer WASM or N-API over a raw C ABI.
- [x] ABI policy recorded as contract-preserving and not public-API-first.

## Finding 3: Define The API-Preserving Migration Path [checkpoint: strategy-complete]

- [x] Migration path defined as Rust modularization plus thin adapter
  preservation.
- [x] Public API compatibility rules are additive, envelope-stable, and
  schema-first.
- [x] Adapter policy recorded for Python, R, Julia, TypeScript, Go, and
  .NET.

## Finding 4: Handoff And Future Compatibility Gates [checkpoint: strategy-complete]

- [x] Future gates defined: contract parity, ABI round-trip checks, adapter
  regression tests, and benchmark comparison against the Rust baseline.
- [x] Follow-on tracks identified for any future ABI pilot or adapter
  migration work.
- [x] Strategy review closed; implementation remains separate from this
  planning track.
