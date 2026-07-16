# Track Specification: Capacity And Budget-Constrained VOI Mature Stable Path

## Overview

Add capacity and budget-constrained VOI as a mature/stable VOI method family for decisions where implementation capacity, budget impact, queueing, workforce, or supply constraints change the value of information.

This dedicated method-family track is distinct from implementation-adjusted VOI because it treats constrained decision feasibility as part of the objective, not only as uptake or delay. The core question is whether information changes the best feasible strategy under resource limits.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Define runtime APIs for budget-constrained net benefit, capacity-constrained strategy choice, queueing/capacity scenarios, and resource-shadow-price diagnostics.
2. Add deterministic fixtures for fixed budget, fixed capacity, capacity expansion, and constrained adoption examples.
3. Implement diagnostics for opportunity cost, budget impact, capacity shortfall, constrained regret, and expected value of capacity-relevant information.
4. Add CLI, docs, cross-language fixtures, property tests, and mature/stable promotion evidence.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`capacity-constrained`, `budget-constrained`, `resource constraints`, `budget impact`, `stable promotion`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
