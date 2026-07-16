# Track Specification: Perspective Uncertainty VOI Mature Stable Path

## Overview

Treat value of perspective as perspective uncertainty and promote it as a core frontier VOI method when evidence supports stable status.

This dedicated method-family track covers value of perspective, perspective uncertainty VOI all the way from runtime implementation through mature/stable promotion review. It exists because the prior adjacent-frontier umbrella track was too broad to prove each method family independently.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Model uncertainty over decision perspective, stakeholder weights, objective functions, and perspective-specific net benefit definitions.
2. Distinguish ordinary sensitivity analysis from VOI over perspective choice and quantify value of resolving perspective uncertainty.
3. Add fixtures for payer, societal, patient, provider, regulator, equity-weighted, and custom stakeholder perspectives.
4. Promote through runtime, CLI, docs, cross-language fixtures, Rust parity review, examples, and release-note gates.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`value of perspective`, `perspective uncertainty`, `stakeholder weights`, `objective uncertainty`, `stable promotion`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
