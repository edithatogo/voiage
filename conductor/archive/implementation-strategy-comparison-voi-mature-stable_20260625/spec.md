# Track Specification: Implementation Strategy Comparison VOI Mature Stable Path

## Overview

Extend implementation-adjusted VOI to compare alternative implementation strategies through mature/stable runtime status.

This dedicated method-family track covers implementation strategy comparison VOI all the way from runtime implementation through mature/stable promotion review. It exists because the prior adjacent-frontier umbrella track was too broad to prove each method family independently.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Define runtime APIs for comparing implementation strategies, uptake curves, adherence support, coverage expansion, delay, and scale-up costs.
2. Add fixtures that separate value of more efficacy evidence from value of better implementation evidence.
3. Implement diagnostics for strategy-specific net benefit, adoption uncertainty, implementation cost, and population impact.
4. Add CLI, docs, cross-language fixtures, property tests, and stable promotion evidence.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`implementation strategy`, `uptake`, `adherence`, `coverage`, `strategy comparison`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
