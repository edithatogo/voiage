# Track Specification: Rust Frontier Numerics Migration Completion

## Overview

Complete the Rust numerics migration for stable and frontier kernels while preserving Python as the public facade.

This cross-cutting track belongs to the comprehensive VOI roadmap follow-through program. It should land before or alongside downstream runtime, registry, dataset, Rust, or HPC work that depends on its decisions.

## Tooling And Execution Boundary

voiage-core, cargo, PyO3/maturin decision record, fixture parity, benchmarks, and Python wrapper tests.

GitHub Actions and `gh` are preferred for repeatable verification. Heavy optional dependencies, cloud runners, Colab, and browser workflows must stay optional and evidence-gated.

## Functional Requirements

1. Inventory remaining Python-owned numerical kernels and classify Rust migration priority by stability and performance value.
2. Define the Python bridge policy and fallback behavior before moving kernels.
3. Port frontier-adjacent kernels where runtime maturity and benchmark evidence justify it.
4. Add fixture parity, property tests, benchmarks, and docs for each migrated kernel.

## Non-Functional Requirements

1. Preserve public API compatibility unless an additive change is explicitly approved.
2. Avoid dependency conflicts in the base install; keep heavyweight or experimental backends optional.
3. Maintain the repository-wide 90 percent coverage gate and language-native binding quality gates.
4. Keep external, hardware, cloud, and experimental gates explicit until evidence exists.

## Acceptance Criteria

1. A tracked migration matrix shows kernel owner, Rust status, Python wrapper status, parity status, and benchmark status.
2. Migrated kernels preserve public Python behavior and result envelopes.
3. Rust benchmarks and tests run through CI before any kernel is called complete.

## Required Keywords For Validation

`Rust numerics core`, `Python facade`, `PyO3`, `maturin`, `fixture parity`, `benchmarks`

## Out Of Scope

1. Irreversible external submissions, paid cloud actions, or hardware purchases without explicit user approval.
2. Weakening existing CI, coverage, contract, or docs gates.
3. Marking experimental or external-gated work complete without evidence.
