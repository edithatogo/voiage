# Track Specification: Rust Core Domain Model

## Overview

This track establishes the Rust `voiage-core` workspace baseline for the stable `voiage` domain model. It focuses on the portable container types, result envelopes, diagnostics/reporting payloads, and deterministic serialization rules that every future Rust kernel or adapter will need to share.

## Goals

1. Define Rust representations for the stable input, result, and metadata types.
2. Preserve the schema-backed contract across serialization and round-trip boundaries.
3. Make the Rust `voiage-core` crate the source of truth for later engine and adapter work.
4. Keep adapter boundaries explicit so core types remain independent of FFI or CLI concerns.

## Functional Requirements

1. Represent the stable core container types in Rust.
2. Represent the stable result envelopes in Rust.
3. Represent diagnostics, reporting, maturity, provenance, and reproducibility payloads in Rust.
4. Preserve named dimensions, validation rules, and deterministic serialization.
5. Define error types for contract violations and invalid shapes.
6. Ensure the Rust model can participate in cross-language fixtures and round-trip tests.
7. Keep the core crate fixture-compatible without requiring a Python runtime or binding layer.

## Acceptance Criteria

1. The Rust `voiage-core` baseline is defined and documented.
2. Core containers, result envelopes, and metadata payloads have explicit Rust shapes.
3. Validation, serialization, and round-trip rules are explicit.
4. Round-trip tests cover the stable contract shapes and fixture payloads.
5. The Rust model is ready for later numerics, interop, and adapter tracks.

## Out Of Scope

1. Porting every method implementation.
2. Binding generation or FFI adapters.
3. Performance tuning beyond basic correctness, shape validation, and deterministic serialization.
4. Registry publishing work beyond the core crate naming baseline.

## Execution Notes

- Treat the workspace baseline as a core crate first, with adapter crates deferred until the core contract is stable.
- This track should remain parallelizable by treating each major container family, result envelope family, and metadata payload family as a separate implementation slice.
- Keep fixture compatibility and serialization behavior deterministic so Rust remains aligned with the shared cross-language contract.
