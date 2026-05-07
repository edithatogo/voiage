# Track Implementation Plan: Rust Core Domain Model

## Phase 1: Workspace Baseline And Core Type Mapping [checkpoint: ]

- [x] Task: Map the stable public contract to Rust types.
  - [x] Define the `voiage-core` workspace root and crate boundary.
  - [x] Define the core container types.
  - [x] Define the result envelope types.
  - [x] Define the diagnostics/reporting structures.
  - [x] Define the serialization-friendly field layout for later adapters.
- [x] Task: Define validation rules.
  - [x] Enforce shape and naming invariants.
  - [x] Define error handling for invalid inputs.
- [x] Task: Conductor - Automated Review And Checkpoint 'Workspace Baseline And Core Type Mapping' (Protocol in workflow.md)

## Phase 2: Serialization And Round-Trip Contracts [checkpoint: ]

- [x] Task: Implement deterministic serialization for the Rust types.
  - [x] Keep the schema mapping language-neutral.
  - [x] Preserve stable field names and nested structures.
- [x] Task: Add round-trip and fixture tests.
  - [x] Validate Rust-to-contract round trips.
  - [x] Validate that representative payloads remain fixture-compatible.
- [x] Task: Conductor - Automated Review And Checkpoint 'Serialization And Round-Trip Contracts' (Protocol in workflow.md)

## Phase 3: Result Metadata, Reporting, And Fixture Compatibility [checkpoint: ]

- [x] Task: Implement stable reporting and diagnostics payloads in Rust.
  - [x] Preserve method maturity metadata.
  - [x] Preserve CHEERS-style reporting envelopes.
- [x] Task: Verify compatibility with existing contract fixtures.
  - [x] Add tests for the representative result families.
  - [x] Verify that the Rust payloads remain fixture-compatible with the shared core contracts.
- [x] Task: Conductor - Automated Review And Checkpoint 'Result Metadata, Reporting, And Fixture Compatibility' (Protocol in workflow.md)

## Phase 4: Handoff To Numerics And Interop [checkpoint: ]

- [x] Task: Mark the Rust domain model ready for engine implementation.
  - [x] Document the data model API for downstream numerics work.
  - [x] Document any migration caveats for bindings and adapter crates.
  - [x] Document the expected `voiage-core` crate boundary for future release work.
- [x] Task: Conductor - Automated Review And Checkpoint 'Handoff To Numerics And Interop' (Protocol in workflow.md)

## Handoff Notes

- The Rust crate boundary is now intentionally container- and metadata-first:
  engine work should consume `ValueArray`, `ParameterSet`, `TrialDesign`, and
  the typed result envelopes rather than inventing new wire formats.
- Adapter crates should translate language-native inputs into the stable Rust
  structs and then serialize the envelopes as-is.
- The remaining numerics work can be parallelized from this point by layering
  engine implementations over the established domain model without changing
  the crate boundary again.
