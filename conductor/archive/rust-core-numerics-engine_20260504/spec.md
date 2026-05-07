# Track Specification: Rust Core Numerics Engine

## Overview

This track ports the numerics-heavy, contract-stable VOI surface to Rust. It is
scalar-first and contract-first: the Rust core should expose deterministic
numeric kernels before it chases broader stochastic or frontier-adjacent
surfaces.

## Goals

1. Implement the deterministic core methods in Rust first.
2. Preserve the shared contract shapes, tolerances, diagnostics, and reporting
   envelopes.
3. Validate every Rust-facing method against the shared fixtures before the
   method family is considered complete.
4. Make the numerics engine benchmarkable against the existing Python/JAX
   baseline after contract parity is established.

## Functional Requirements

1. Implement the stable deterministic scalar methods first.
2. Implement the deterministic partial-information summary methods second.
3. Implement EVSI and the frontier-adjacent kernels only after the shared
   core contracts are in place.
4. Keep each method family batch-oriented and parallelization-friendly.
5. Preserve diagnostics, reporting, and maturity metadata across method
   outputs.
6. Treat shared fixtures as the source of truth for parity and regression
   checks.

## Acceptance Criteria

1. The deterministic Rust kernels exist for the agreed core methods.
2. The Rust outputs match the shared fixtures within tolerance.
3. Regression tests cover each method family and the shared contract envelope.
4. The core engine is benchmarkable independently of bindings.
5. The contract-first result shapes remain schema-compatible at the boundary.

## Out Of Scope

1. User-interface work.
2. Binding packaging.
3. High-risk stochastic approximations that still need policy decisions.
4. Any schema-breaking expansion of the stable result envelopes.

## Execution Notes

- Split the methods into parallel workstreams once the shared data model and
  fixture harness are ready.
- Keep the contract stable while the engine is ported.
- Prefer deterministic kernels whose correctness can be checked against the
  existing shared fixtures before widening scope.
