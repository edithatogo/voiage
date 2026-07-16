# Track Specification: AI-Assisted Evidence Triage VOI Mature Stable Path

## Overview

Add AI-assisted evidence triage VOI as a mature/stable VOI method family for deciding whether automated screening, extraction, summarization, prioritization, or synthesis support is worth using in evidence-generation workflows.

This dedicated method-family track is distinct from expert-elicitation and evidence-synthesis VOI because it treats automation quality, review burden, false inclusion, false exclusion, human override, and auditability as decision-relevant uncertainties.

## Tooling And Execution Boundary

GitHub Actions and `gh` are the default repeatable evidence path for test, docs, coverage, frontier-contract, and artifact checks. Browser, cloud, Colab, or hardware workflows remain optional and must be recorded as explicit gates when unavailable.

## Functional Requirements

1. Define runtime APIs for triage sensitivity/specificity, extraction error, reviewer time, audit sampling, automation cost, and decision impact.
2. Add deterministic fixtures for literature-screening triage, evidence extraction, active-review prioritization, and human-in-the-loop audit examples.
3. Implement diagnostics for false exclusion risk, false inclusion burden, reviewer time saved, audit value, model drift, and expected value of AI-assisted evidence triage.
4. Add CLI, docs, cross-language fixtures, property tests, and mature/stable promotion evidence.

## Mature/Stable Acceptance Criteria

1. Runtime API, result envelope, diagnostics, CLI command, docs, and examples exist.
2. Deterministic synthetic fixtures and relevant real/open-data examples or explicit data gates exist.
3. Cross-language conformance fixtures and Rust parity review are complete or explicitly deferred with rationale.
4. Unit, integration, CLI, property-based, docs, coverage, and frontier-contract tests pass.
5. Changelog, migration guide, maturity metadata, and release notes document the promotion decision.
6. The method remains experimental or fixture-backed if any mature/stable gate is missing.

## Required Keywords For Validation

`AI-assisted`, `evidence triage`, `human-in-the-loop`, `audit`, `stable promotion`

## Out Of Scope

1. Claiming mature/stable status before the promotion review passes.
2. Adding heavyweight dependencies to the base install without architecture-governance approval.
3. Performing irreversible external submissions, paid cloud actions, or hardware actions without explicit user approval.
