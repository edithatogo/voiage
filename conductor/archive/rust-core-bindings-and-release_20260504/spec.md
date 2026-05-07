# Track Specification: Rust Core Bindings And Release

## Overview

This track defines how the Rust core is exposed to the rest of the ecosystem.
It establishes Rust as the canonical execution engine, Python as the primary
user-facing façade, and the remaining language packages as thin adapters that
forward to the same core semantics. The track covers packaging, release gates,
compatibility bands, and per-language CI/CD expectations for Python, R, Julia,
TypeScript, Go, and .NET.

## Goals

1. Define the binding shape for each supported language.
2. Define the release and packaging contract for a Rust-core future.
3. Keep the user-facing docs honest about what is native core versus adapter.
4. Specify the core-first semver policy and the compatibility bands each
   adapter must declare.
5. Specify the CI/CD gates each language must satisfy before a release tag can
   publish.

## Functional Requirements

1. Define how Python consumes the Rust core as the primary façade.
2. Define how R consumes the Rust core as a thin adapter with an R-native
   release channel.
3. Define how Julia, TypeScript, Go, and .NET adapt to the Rust core with
   thin language-specific wrappers.
4. Define the CI/CD gates needed to build, test, package-dry-run, and publish
   each binding against the Rust core.
5. Update the release docs so they describe a Rust-first architecture clearly.
6. Record registry-specific release gates for PyPI/TestPyPI, CRAN/r-universe,
   Julia General, npm, the Go module proxy, crates.io, and NuGet.

## Acceptance Criteria

1. The binding strategy for each language is written down.
2. The release matrix is updated to reflect Rust-core ownership.
3. The CI/CD expectations for core-first release are explicit.
4. The docs explain what remains a binding and what has moved into the core.
5. Core semver, compatibility-band policy, and release-gate expectations are
   explicit and consistent across languages.

## Out Of Scope

1. Implementing all bindings immediately.
2. Changing package registry targets unless the Rust-core model requires it.
3. Reworking the existing core-api and conformance-fixture contracts beyond
   what is needed to describe the Rust-core ownership model.

## Execution Notes

- Keep the binding strategy compatible with the existing fixture/conformance
  system.
- Prefer thin adapters over duplicated business logic.
- Treat the Rust core as the semantic authority; adapters may differ in syntax
  and packaging but not in supported result shapes or policy rules.
