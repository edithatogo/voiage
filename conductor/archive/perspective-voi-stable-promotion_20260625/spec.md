# Track Specification: Perspective VOI Stable Promotion

## Overview

Promote Value of Perspective, including perspective uncertainty, from experimental to stable with cross-language fixtures, CLI/docs/examples, Rust parity, and release notes.

This is a follow-through track in the Frontier promotion lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

Frontier contract fixtures, CLI tests, Rust parity, and language adapter checks.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Audit the existing perspective runtime, CLI, plot helper, schemas, and deterministic fixtures.
2. Add cross-language conformance expectations and Rust-kernel parity where numerically appropriate.
3. Validate examples across payer, societal, patient, provider, regulator, equity-weighted, and custom perspectives.
4. Add a specific perspective-uncertainty review that treats uncertainty over stakeholder perspective, objective function, and perspective weights as decision-relevant uncertainty rather than ordinary sensitivity analysis.
5. Update maturity metadata only after compatibility and release-note gates pass.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. Perspective VOI and perspective uncertainty have stable promotion evidence and no experimental-only wording in stable docs.
2. Fixture parity passes across Python and relevant binding contracts.
3. The changelog records the stable promotion decision and compatibility boundary.

## Required Keywords For Validation

`Value of Perspective`, `perspective uncertainty`, `cross-language fixtures`, `Rust parity`, `stable`, `release note`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
