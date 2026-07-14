# Track Specification: Discrete GPU Production Speedup Evidence

## Overview

Capture CUDA-class GPU speedup on production-sized VOI workloads using Colab CLI, GitHub Actions artifacts, and optional cloud GPU runners.

This is a follow-through track in the HPC production evidence lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

colab GPU runtime, gh artifact retrieval, optional cloud GPU, and benchmark manifests.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Extend compact GPU parity evidence to production-sized workloads with warm-up and repeated timings.
2. Use colab CLI for free GPU evidence where runtime availability permits.
3. Capture device metadata, transfer overhead, compile overhead, workload hashes, and CPU comparisons.
4. Record quota or runtime unavailability as blocked evidence rather than success.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. GPU evidence packets contain timing, throughput, parity, and CPU comparison fields.
2. Artifacts are stored under an accelerator evidence handoff location and linked from docs.
3. The production speedup claim remains gated until evidence passes review.

## Required Keywords For Validation

`discrete GPU`, `CUDA`, `Colab`, `production-sized`, `speedup`, `benchmark packet`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
