# Track Specification: Regulatory And Market-Access VOI Mature Stable Path

## Overview

Add regulatory and market-access VOI as a mature/stable VOI method family for approval, reimbursement, label-expansion, pricing, coverage-with-evidence, and evidence-package decisions.

This dedicated method-family track is distinct from implementation-adjusted VOI because the decision can be whether evidence changes external approval, coverage, pricing, or access conditions before implementation begins.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Define runtime APIs for approval probability, reimbursement probability, label scenarios, pricing scenarios, coverage decisions, and evidence-package value.
2. Add deterministic fixtures for approval, rejection, restricted-label, coverage-with-evidence, and price-threshold examples.
3. Implement diagnostics for regulatory uncertainty, payer uncertainty, access delay, evidence-package cost, expected access value, and expected value of market-access information.
4. Add CLI, docs, cross-language fixtures, property tests, and mature/stable promotion evidence.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`regulatory`, `market-access`, `reimbursement`, `approval probability`, `stable promotion`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
