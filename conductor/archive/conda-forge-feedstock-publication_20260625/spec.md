# Track Specification: Conda-Forge Feedstock Publication

## Overview

Drive conda-forge feedstock PR creation, rerender checks, merge tracking, and package availability checks.

This is a follow-through track in the Registry follow-through lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

GitHub Actions, gh, conda-smithy/rerender evidence, and conda-forge PR status checks.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Verify the release artifact and hash used by the feedstock update workflow.
2. Create or refresh the feedstock PR using the existing automation when credentials permit.
3. Track rerender, CI, maintainer review, merge, and package index visibility as separate states.
4. Record the external conda-forge maintainer gate until the package is visible in the channel.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. A feedstock PR or blocked credential report exists with evidence links.
2. The package availability check records whether conda-forge indexing succeeded.
3. Docs and registry snapshot use external-gate language until merge/indexing are verified.

## Required Keywords For Validation

`conda-forge`, `feedstock`, `rerender`, `PR`, `merge`, `indexed`, `external approval`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
