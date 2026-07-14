# Track Specification: Adjacent Frontier Runtime Completion

## Overview

Implement runtime paths for dynamic real-options, causal transportability, data quality/privacy/linkage, computational refinement, and expert synthesis, then hand them to dedicated method-family mature/stable tracks with cross-language parity and release-note gates.

This is a follow-through track in the Frontier runtime completion lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

The umbrella runtime scope is now decomposed into dedicated mature/stable tracks:

- `causal-identification-transportability-voi-mature-stable_20260625`
- `data-quality-measurement-privacy-linkage-voi-mature-stable_20260625`
- `computational-model-refinement-voi-mature-stable_20260625`
- `expert-elicitation-evidence-synthesis-voi-mature-stable_20260625`
- `dynamic-real-options-voi-mature-stable_20260625`

## Tooling And Execution Boundary

Python APIs, Rust kernels, property tests, conformance fixtures, CLI examples, and docs.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Map each contract-only adjacent family to runtime APIs, result envelopes, diagnostics, and CLI commands.
2. Implement deterministic kernels or explicit approximation policies with Rust parity where appropriate.
3. Add synthetic fixtures, property tests, open-data examples, and docs for each family.
4. Promote families only through their dedicated mature/stable tracks and the shared frontier stable-promotion program after runtime completion, cross-language fixture review, and release-note preparation.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. Each adjacent family has runtime API, CLI, fixtures, tests, and docs.
2. Rust parity or a documented deferral exists for every numerical kernel.
3. No family is labeled stable until the promotion program, cross-language parity, and release-note gates pass.

## Required Keywords For Validation

`dynamic real-options`, `causal transportability`, `data quality`, `privacy`, `linkage`, `computational refinement`, `expert synthesis`, `cross-language parity`, `release-note`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
