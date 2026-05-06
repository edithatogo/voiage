# Track Specification: First External Bindings Release Matrix

## Overview

`voiage` already has the core API contract, canonical schemas, and
cross-language fixture scaffolding needed to support external bindings. This
track defines the release matrix and binding-quality contract for the first
external packages so later binding implementation tracks can inherit a single
publishing model instead of re-deciding it per language.

The contract now reflects the current release automation: Python publishes to
PyPI/TestPyPI from tag pushes, the polyglot release workflows publish npm,
crates.io, and NuGet artifacts, and Go, Julia, and R produce GitHub release
artifacts for registry and distribution handoff. The remaining external
registry dependencies are called out explicitly: conda-forge feedstock
approval, CRAN/r-universe, and the Julia General registry still require their
own upstream processes.

The track is about release readiness, not binding code generation. It covers
the package managers, CI/CD gates, dry-run requirements, provenance, and
release-trigger rules that each binding must satisfy before it can be treated
as a supported distribution channel.

## Functional Requirements

1. Define the release matrix for the initial external bindings:
   - Python: PyPI, TestPyPI, and Conda-forge.
   - R: CRAN when mature, with r-universe or GitHub Releases for early
     distribution.
   - Julia: Julia General registry.
   - TypeScript: npm.
   - Go: tagged Go modules via the Go module proxy, with GitHub Releases for
     release notes and artifacts.
   - Rust: crates.io.
   - .NET: NuGet targeting .NET 11 (`net11.0`).
2. Define the minimum CI/CD gates each binding must satisfy before release:
   build, lint/format, type/static checks where applicable, unit tests, docs
   checks, shared conformance-fixture validation, and package dry-run
   validation on pull requests.
3. Define the trusted release trigger model for each binding, including version
   tags/releases, registry credentials or trusted publishing, generated
   changelog/release notes, and rollback guidance.
4. Define how shared conformance fixtures are consumed by binding CI so the same
   canonical cases are run before publication across all supported languages.
5. Define the package metadata and provenance requirements needed to keep
   release outputs auditable and comparable across languages.

## Non-Functional Requirements

1. The contract must stay language-neutral and avoid Python-specific runtime
   assumptions.
2. The release model must be explicit enough for future binding tracks to reuse
   without reinterpretation.
3. The matrix must be specific enough to support CI automation and package
   publishing dry runs.
4. Release rules must keep experimental and stable binding channels distinct.

## Acceptance Criteria

1. A versioned release matrix exists that maps each target language to its
   package manager, release channel, and CI/CD gates.
2. The contract explicitly states the required package dry-run and trusted
   publishing behavior for each binding.
3. The contract references the shared conformance-fixture track as a
   prerequisite for binding release.
4. The roadmap and tracks registry point at this track for the first external
   bindings phase.

## Out of Scope

1. Implementing any binding runtime code.
2. Publishing an actual package to any registry.
3. Designing language-specific adapter APIs beyond the shared release contract.
