# Track Specification: Core API Spec Foundation

## Overview
This track creates the authoritative written foundation for a language-agnostic `voiage` core API. It defines the conceptual contract before conformance fixtures or Python cleanup begin, so later implementation work is driven by explicit semantics instead of inferred behavior from the current Python package.

## Functional Requirements
1. The track must produce a written specification that defines the v1 contract scope for a cross-language VOI core.
2. The contract format must be `JSON Schema` for machine-readable structures plus Markdown semantics for mathematical meaning, invariants, and workflow rules.
3. The v1 conceptual model must be net-benefit-first, while treating study-design and research-decision objects as first-class concepts rather than bolt-ons.
4. The v1 target runtimes must be Python, R, and Julia, with other languages explicitly deferred.
5. The specification must require every future binding track to include package-manager publishing and language-specific CI/CD before the binding can be marked complete:
   - Python: PyPI, TestPyPI, and Conda-forge.
   - R: CRAN when mature, with r-universe or GitHub Releases for early distribution.
   - Julia: Julia General registry.
   - TypeScript: npm.
   - Go: tagged Go modules consumable through the Go module proxy, with GitHub Releases for release notes/artifacts.
   - Rust: crates.io.
   - .NET: NuGet, targeting .NET 11 (`net11.0`).
6. Each binding's CI/CD contract must cover build, lint/format, type/static checks where applicable, unit tests, docs checks, conformance-fixture validation, package dry-run validation on pull requests, and registry publishing on trusted version tags/releases.
7. The specification must record backend and interchange principles:
   - xarray-labeled scientific arrays remain the Python in-memory reference model
   - NumPy remains the reference execution baseline
   - JAX is an optional acceleration backend, not the public contract
   - Arrow/Parquet is the interchange boundary for cross-language fixtures and bindings
   - Polars may be used as an adapter for tabular IO, not as the canonical computational model
8. The specification must incorporate the competitive feature scan as a scope-setting input and identify the minimum capability bar required for `voiage` to be credible against `voi`, `dampack`, `BCEA`, `TreeAge Pro`, `hesim`, and `heemod`.

## Non-Functional Requirements
1. The specification must be compact and explicit enough for a smaller execution model such as `gpt-5.4-mini` to follow phase-by-phase without interpretive guesswork.
2. Terms, entities, and result shapes must be named consistently so future bindings can avoid language-specific drift.
3. The document must distinguish normative v1 behavior from deferred or experimental work.

## Acceptance Criteria
1. A track-local written spec exists and is detailed enough to drive schema authoring in the next track.
2. Core vocabulary, scope boundaries, and backend/interchange principles are explicitly recorded.
3. The specification enumerates which capability families are in v1, which are deferred, and which are only reserved as extension points.
4. The specification identifies the exact downstream outputs required from the next two tracks: conformance fixtures and Python cleanup.
5. The specification includes a release/distribution matrix that maps each target language to its package manager, CI gates, release trigger, and required publishing dry run.

## Out of Scope
1. Authoring the actual JSON Schema files.
2. Creating conformance fixtures or binding code.
3. Refactoring the Python package implementation.
