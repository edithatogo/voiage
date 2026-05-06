# Track Specification: Rust Core Migration Foundation

## Overview

This track establishes the migration boundary for moving `voiage` toward a Rust-based execution core. The goal is to decide, document, and lock in the architectural shape before any large-scale implementation begins.

The architectural decision is fixed for this track:

- Rust is the authoritative execution core for deterministic VOI calculations, shared result contracts, and core serialization behavior.
- Python remains the primary façade for CLI entrypoints, orchestration, plotting, and compatibility wrappers.
- R, Julia, TypeScript, Go, .NET, and any other external language targets are thin bindings or adapters over the same Rust contract.

The migration is deliberately staged. Transitional Python code may remain in place where it provides compatibility, orchestration, or heavy ecosystem integration, but it should not remain the source of truth for core numerical behavior once the Rust core is available.

## Goals

1. Confirm and record Rust as the authoritative execution core.
2. Define the Python façade boundary and identify which areas remain transitional only.
3. Define the first-crate Rust workspace baseline for the core engine.
4. Define the compatibility contract for existing bindings and shared schemas.
5. Set the versioning, semver, and release expectations for a Rust-core future.

## Architecture Decision

The repository will converge on a layered model:

1. Rust core crates own the computational kernels, deterministic method logic, and result-schema serialization.
2. Python owns the public orchestration surface and compatibility façade until the Rust-backed APIs are stable.
3. Non-Python bindings stay thin and reuse the same Rust contract rather than duplicating analysis logic.

## Transitional Compatibility

The following areas are transitional only and should not be reimplemented as independent long-term cores:

- `voiage/analysis.py` orchestration and backend dispatch
- `EVPPI`, `EVSI`, and simulation-heavy or callback-heavy methods until lower-level contracts are stable
- plotting, CLI formatting, and docs-generation layers
- optional backend plumbing for JAX, scikit-learn, and GPU helpers

These areas may continue to exist in Python during the migration, but only as wrappers, adapters, or compatibility shims around Rust-owned kernels.

## First Workspace Baseline

The first Rust workspace should start small and focused:

- the current repo state is a single publishable Rust package at
  `bindings/rust/`, named `voiage-core`, with `voiage_core` as the library
  target
- one core crate for shared numeric types, result payloads, and deterministic VOI kernels
- one FFI or adapter crate only if needed for Python integration
- no binding-specific business logic in the first Rust workspace baseline
- explicit versioned contract types for arrays, sample containers, and method results

## Workspace And Toolchain Policy

The current Rust package is intentionally simple: it is a single-package
library crate today, not a multi-crate workspace. A workspace split is only
required once a second Rust package is needed for FFI glue or binding-specific
packaging.

- Package name: `voiage-core`
- Library target: `voiage_core`
- Current Rust ownership: `bindings/rust/src/domain.rs` for the domain model
  and `bindings/rust/src/scalar.rs` for the deterministic scalar contracts
- Interop crates: deferred until a binding actually needs Rust-side FFI glue
- Release gating: `cargo fmt --check`, `cargo clippy --all-targets --locked -- -D warnings`,
  `cargo test --locked`, `cargo doc --no-deps --locked`, and
  `cargo package --locked --allow-dirty`
- Release tags: `rust-v*` for the Rust package release path

## Migration Sequence And Compatibility Policy

The migration order is designed to keep the stable domain model and the
deterministic kernels in Rust first, while leaving higher-level orchestration in
Python until the Rust APIs are ready to replace them.

- The current Rust-first slice is the domain model and scalar contract helpers
  in `bindings/rust/src/domain.rs` and `bindings/rust/src/scalar.rs`.
- Python remains the primary façade for orchestration, plotting, CLI
  formatting, and compatibility wrappers until the Rust-backed APIs are stable.
- `voiage/analysis.py`, `EVPPI`, `EVSI`, plotting helpers, CLI formatting, and
  docs-generation layers are transitional only and should be treated as wrapper
  surfaces rather than permanent duplicate cores.
- Non-Python bindings remain thin adapters over the Rust contract, so the
  authoritative artifacts are the Rust crate, its serialized result contracts,
  and the shared fixture/schema set rather than any binding-local reimplementation.
- Compatibility is declared through semver-aligned Rust releases plus
  binding-specific release conventions that forward to the same Rust contract.

## Functional Requirements

1. Document the core migration choice.
2. Define the Rust crate/workspace layout and ownership boundaries.
3. Define how the stable cross-language contract maps to Rust types, error handling, and serialization.
4. Define the migration sequence for moving work out of the Python reference implementation.
5. Define how the existing binding release story changes once Rust becomes the engine.
6. Specify which parts of the Python surface remain transitional façades only.
7. Specify which crates or packages are the first implementation targets in the Rust workspace.

## Acceptance Criteria

1. The Rust-core migration shape is documented and recorded in the track spec.
2. The workspace and package boundaries are defined well enough for later implementation tracks to proceed independently.
3. The migration sequence identifies what should move first and what should remain as façades.
4. The release implications for Python and other bindings are clearly stated.
5. Transitional-only areas are explicitly named so later tracks do not treat them as permanent architecture.
6. The first-crate workspace baseline is concrete enough to begin implementation without renegotiating scope.

## Out Of Scope

1. Porting the full implementation.
2. Rewriting the binding packages.
3. Making the Python API behavior changes required by the eventual migration.
4. Moving plotting, CLI polish, or docs-only workflows into Rust during this track.

## Execution Notes

- This track should finish before any deep Rust implementation work starts.
- Later tracks should be able to consume its decisions without renegotiating the architecture.
