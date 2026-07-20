# Mature Hardened v1.0 Architecture and Release Programme

## Overview

This umbrella programme takes `voiage` from its current contract-first, mixed-runtime state to a mature and hardened v1.0 release. Rust becomes the authoritative execution core, Python becomes a thin reference facade, retained non-Python bindings execute the same Rust implementation, Astro/Starlight becomes the sole documentation platform, and every stable artifact is published and clean-installable from its supported registry.

Existing Conductor tracks remain authoritative where their scope is still valid. Completed Rust foundation tracks are historical groundwork and must not be reopened or represented as proof that the production migration is complete. Existing registry and hardware-evidence tracks remain downstream children with their external gates preserved.

## Product Maturity Boundary

### Stable v1.0 core

The stable surface includes EVPI, EVPPI, EVSI, ENBS, CEAF, dominance and cost-effectiveness-frontier operations; canonical schemas; diagnostics, reporting, provenance and errors; the supported CLI; and core plotting and reporting workflows.

### Supported extensions

An extension may remain supported only when it has a justified product role, stable contracts, Rust execution for VOI policy, tests, documentation and independent optional packaging. Extensions may not duplicate core numerical policy.

### Experimental surface

Experimental functionality must be isolated from stable top-level exports, carry explicit maturity metadata or warnings, avoid bloating the stable installation, and remain outside the v1.0 release-critical path.

## Functional Requirements

1. Reconcile local and remote repository state, branches, pull requests, GitHub issues, releases, Conductor records, roadmap sources and generated artifacts before implementation claims are made.
2. Classify all tracks as v1.0-required, post-v1, externally blocked, superseded, duplicate or completed and register one explicit dependency order.
3. Freeze a normative stable API covering schemas, shapes, types, numerical tolerances, seeds, errors, missing values, supported platforms and compatibility policy.
4. Maintain language-neutral golden fixtures for normal, edge and invalid cases.
5. Make the Rust workspace the authoritative runtime for all stable numerical and domain operations.
6. Stabilize Rust domain types and a narrow versioned C ABI with explicit ownership, errors and compatibility guarantees.
7. Integrate Python through PyO3 and maturin with dynamic versioning and cross-platform wheels.
8. Migrate stable EVPI, EVPPI, EVSI, ENBS, CEAF, dominance and frontier kernels to Rust with parity, property, fuzz and benchmark evidence.
9. During the remaining 0.x series, route public Python APIs through Rust, issue deprecations and provide migration guidance; at v1.0 remove duplicate Python numerical kernels and silent fallbacks.
10. Retain outside Rust only essential Python facade responsibilities: schemas and I/O, orchestration, CLI, plotting, reporting and compatibility wrappers.
11. Extract, remove or explicitly isolate domain-specific, web, widget, accelerator, distributed and experimental code unless a separate supported-extension case is documented.
12. Convert retained R, Julia, TypeScript, Go and .NET bindings into thin Rust adapters using the C ABI, WASM or N-API as justified; remove independent numerical implementations.
13. Consolidate all maintained documentation into Astro/Starlight, migrate unique RST content and remove Sphinx configuration, dependencies and duplicate builds.
14. Preserve a solo-maintainer GitHub model with no mandatory external review while enforcing reliable CI, security, provenance, quality and release checks.
15. Enforce Python coverage of at least 90 percent and appropriate Rust and binding coverage, property, fuzz, mutation, sanitizer, ABI, dead-code and benchmark gates.
16. Produce SBOMs, provenance attestations, signatures, checksums, license evidence and reproducibility reports.
17. Publish and verify retained artifacts on TestPyPI, PyPI, conda-forge, crates.io, CRAN or the approved R target, Julia General, npm, the Go module proxy and NuGet.
18. Publish a signed GitHub v1.0 release with release notes, migration guidance, artifacts, SBOMs, provenance, signatures, checksums and reproducibility evidence.

## Non-Functional Requirements

1. Stable cross-language behavior must satisfy the same fixtures and documented tolerances.
2. The migration must not introduce undocumented regressions or weaken semantic-versioning guarantees.
3. Stable builds and release artifacts must be reproducible and traceable to source.
4. The core installation must work without JAX, GPU, web, widget, distributed or experimental dependencies.
5. Optional dependencies must not bloat or destabilize the base package.
6. Errors must be explicit across Rust, Python, the C ABI and retained bindings.
7. Documentation and maturity claims must be evidence-backed and machine-checkable.

## Existing Track Integration

1. Existing registry-publication tracks remain responsible for live submission, indexing, approval and externally blocked evidence under the v1.0 release gate.
2. Existing accelerator and hardware-evidence tracks are post-core follow-through and cannot block v1.0 unless a retained stable claim explicitly depends on that evidence.
3. Existing mature/stable frontier tracks must either satisfy the supported-extension rules, move to the experimental surface, or be deferred beyond v1.0.
4. Archived Rust, binding, versioning, documentation and strict-CI tracks provide historical inputs but do not replace the production acceptance criteria in this programme.

## Acceptance Criteria

1. Local and remote `main` are reconciled and the release branch starts from the authoritative remote state.
2. Every v1.0-required Conductor track is complete and archived with evidence; post-v1 and external gates remain distinctly registered.
3. The stable API and compatibility policy are frozen and executable through language-neutral fixtures.
4. Every stable numerical operation executes in Rust.
5. No duplicate or silently reachable Python numerical kernel remains in the v1.0 distribution.
6. Every retained binding demonstrably executes the Rust core and passes shared conformance and ABI tests.
7. Experimental and optional functionality is isolated from the stable installation and namespace.
8. Astro/Starlight is the sole authoritative documentation build and no Sphinx stack remains.
9. CI, security, coverage, typing, lint, dead-code, fuzz, sanitizer, mutation, benchmark, packaging and release gates pass.
10. Every retained registry artifact is live, indexed and installable, or v1.0 is explicitly held until the required external gate clears.
11. Clean installation and representative Rust-backed analyses pass on all supported platforms and runtimes.
12. The signed GitHub v1.0 release contains complete artifacts, checksums, SBOMs, provenance, signatures, release notes, migration guidance and reproducibility evidence.
13. No unresolved critical or high-severity security finding remains.
14. Version, metadata, documentation, registry and roadmap state consistently describe a stable v1.0 release.

## Out of Scope

1. Production FPGA or ASIC execution and fabricated-silicon evidence.
2. Mandatory accelerator or distributed-runtime dependencies.
3. Stable promotion of every frontier or research method.
4. Compatibility with undocumented internal implementation details.
5. Independent numerical implementations in language bindings.
6. A parallel Sphinx documentation stack.
7. Unapproved domain-specific, web, widget, accelerator, distributed or experimental code in the stable core.
