# Track Specification: Spack Package Merge Follow-Through

## Overview

Convert Spack readiness into upstream package PR, CI evidence, maintainer review, and merge tracking.

This is a follow-through track in the HPC registry follow-through lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

Spack validation, GitHub Actions, upstream PR checks, and gh.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Validate the Spack package recipe against current release artifacts and dependencies.
2. Open or prepare the upstream Spack PR when credentials and repo state permit.
3. Track upstream CI, maintainer review, requested changes, merge, and package visibility separately.
4. Feed merged Spack evidence into E4S readiness without overclaiming inclusion.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. Spack PR or blocked handoff report exists with reproducible validation commands.
2. The registry snapshot records upstream package status and evidence URL.
3. E4S dependency evidence can consume the Spack result.

## Required Keywords For Validation

`Spack`, `package.py`, `spack install`, `upstream PR`, `maintainer review`, `E4S`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
