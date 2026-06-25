# Track Specification: Federated And Privacy-Preserving VOI Mature Stable Path

## Overview

Add federated and privacy-preserving VOI as a mature/stable VOI method family for multi-site evidence generation where data cannot be centralized or must satisfy privacy, governance, or disclosure constraints.

This dedicated method-family track extends the existing privacy/linkage track by focusing on the execution model: site-local summaries, secure aggregation, synthetic-data handoff, privacy-budget accounting, and evidence value under governance constraints.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Define runtime APIs for federated site summaries, privacy-preserving aggregation, privacy-budget metadata, disclosure diagnostics, and site-level contribution value.
2. Add deterministic fixtures for multi-site summary aggregation, synthetic-data fallback, privacy-budget sensitivity, and blocked individual-level data access.
3. Implement diagnostics for site contribution, privacy loss, aggregation error, disclosure risk metadata, and expected value of privacy-preserving information.
4. Add CLI, docs, cross-language fixtures, property tests, and mature/stable promotion evidence.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`federated`, `privacy-preserving`, `secure aggregation`, `privacy budget`, `stable promotion`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
