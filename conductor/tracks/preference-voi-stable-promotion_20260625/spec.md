# Track Specification: Preference VOI Stable Promotion

## Overview

Promote preference VOI and preference heterogeneity with cross-language fixture parity, adapter stability, schemas, examples, and docs.

This is a follow-through track in the Frontier promotion lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

Frontier fixtures, adapter checks, Rust parity, and CLI examples.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Audit preference profile schemas, runtime outputs, CLI behavior, docs, and deterministic fixtures.
2. Add or verify cross-language adapter-facing fixtures for preference information and individualized-care use cases.
3. Confirm result-envelope stability and deprecation policy before stable promotion.
4. Record release-note and migration-guide changes for users moving from experimental usage.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. Preference VOI has cross-language fixture-backed parity and stable maturity metadata.
2. Language adapter checks preserve schema and result-envelope behavior.
3. Docs include deterministic examples and stable API guidance.

## Required Keywords For Validation

`preference`, `preference heterogeneity`, `cross-language fixture parity`, `language adapters`, `schema stability`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
