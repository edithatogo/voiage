# Track Specification: Adaptive Learning And Bandit VOI Mature Stable Path

## Overview

Add adaptive learning and bandit VOI as a mature/stable VOI method family for sequential allocation, exploration, exploitation, and stopping decisions.

This dedicated method-family track covers adaptive learning and bandit VOI all the way from runtime implementation through mature/stable promotion review. It is distinct from adaptive trial VOI because the decision unit can be an operational policy, online service, clinical pathway, monitoring program, or implementation strategy whose allocation changes continuously as evidence arrives.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Define runtime APIs for sequential allocation, exploration cost, exploitation regret, stopping rules, posterior updates, and adaptive policy value.
2. Support deterministic fixtures for Thompson sampling, upper-confidence-bound, epsilon-greedy, and budget-limited exploration examples.
3. Add diagnostics for regret, opportunity cost, decision-switch frequency, sampling burden, and expected value of continued adaptive learning.
4. Add CLI, docs, cross-language fixtures, property tests, and mature/stable promotion evidence.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`adaptive learning`, `bandit VOI`, `sequential allocation`, `exploration`, `stable promotion`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
