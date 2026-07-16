# Track Specification: Evidence Obsolescence And Refresh VOI Mature Stable Path

## Overview

Add evidence obsolescence and refresh VOI as a mature/stable VOI method family for deciding when living evidence, surveillance updates, guideline updates, model refreshes, or data-refresh cycles are worth doing.

This dedicated method-family track is distinct from monitoring/surveillance VOI because it values the timing and cadence of refreshing already-adopted evidence products when evidence decays, technologies change, or decision contexts drift.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Define runtime APIs for evidence half-life, obsolescence risk, refresh cadence, update cost, living-review value, and model-refresh value.
2. Add deterministic fixtures for stable evidence, decaying evidence, technology replacement, guideline refresh, and delayed-update examples.
3. Implement diagnostics for evidence age, drift, obsolete-decision risk, refresh burden, update cadence, and expected value of evidence refresh information.
4. Add CLI, docs, cross-language fixtures, property tests, and mature/stable promotion evidence.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`evidence obsolescence`, `refresh VOI`, `living evidence`, `update cadence`, `stable promotion`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
