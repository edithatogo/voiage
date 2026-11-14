# Rust dependency rationalisation — voiage (2026-11-13)

**Status:** proposal and review contract; no code adoption, runtime verification or benchmark result is claimed here.

## Repository-specific decisions

- **Verified:** Rust workspaces live under `rust/`; private `voiage-python` already uses PyO3 with `abi3-py315`; `voiage-numerics` includes Proptest and a harness-free benchmark.
- **Retain:** Rust-core numerical calculations, binding-independent domain kernels and PyO3 strictly in the private adapter.
- **Evaluate:** Criterion as a benchmark harness only if current numerical benchmark method needs it; do not claim measured performance from a config change.
- **Extend:** Proptest and independent numerical validation for VOI/EVPI/EVPPI properties, permutation invariance, bounds, numeric stability and deterministic uncertainty estimates.
- **Conditional:** Rayon for independent Monte Carlo partitions after RNG and reduction-order guarantees, Polars for separately bounded batch tabulation, Serde for versioned data transfer.
- **Reject:** Tokio/SQLx/Ratatui/Syn/Quote in numerical core without observed need. Do not silently change HPC, Python or GPU feature contracts.

## Evidence and release gate
1. Discover actual source, package boundary, MSRV, lockfile, existing tests and Conductor ownership; avoid speculative dependencies.
2. For runtime behaviour, write failing contract tests before implementation, run targeted and full suites; preserve stable schemas, errors, CLI semantics and deterministic outputs.
3. Record `cargo tree -d`, feature selections, security/licensing/advisory reviews, transitive dependency cost and rollback before accepting an adoption.
4. For performance changes, benchmark identical inputs and correctness/invariance oracles; do not claim improvements without measurements.
5. This document is a planning entry, NOT proof of compilation, hosted CI or scientific validation; update status only after a revision-bound receipt.
