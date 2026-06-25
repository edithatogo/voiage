# Track Specification: Custom-Circuit Production Acceleration Review

## Overview

Decide whether FPGA or ASIC evidence justifies production acceleration claims after hardware evidence exists.

This is a follow-through track in the FPGA/ASIC external gates lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

Benchmark review, deployment cost review, fallback contract review, and Conductor decision records.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Collect FPGA board, ASIC shuttle, pre-silicon, CPU fallback, and benchmark evidence into one review packet.
2. Compare production speedup, reproducibility, deployment cost, maintenance burden, and user value.
3. Produce an explicit go/no-go decision for production acceleration claims.
4. If no-go, document why the placeholder or pre-silicon state remains the correct boundary.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. The custom-circuit decision is backed by evidence rather than feasibility enthusiasm.
2. Docs and roadmap reflect the go/no-go outcome without ambiguity.
3. Any production claim includes CPU comparison, hardware runtime evidence, and fallback proof.

## Required Keywords For Validation

`custom-circuit`, `production acceleration`, `deployment cost`, `go/no-go`, `fallback contract`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
