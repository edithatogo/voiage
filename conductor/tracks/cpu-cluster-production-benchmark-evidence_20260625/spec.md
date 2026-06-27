# Track Specification: CPU Cluster Production Benchmark Evidence

## Overview

Prove CPU cluster and distributed scheduler value on larger VOI workloads with repeatable benchmark evidence.

This is a follow-through track in the HPC production evidence lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

GitHub Actions matrix, local scheduler smoke tests, optional cloud runners, and benchmark artifacts.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Select production-sized CPU workloads that stress parallelism without requiring proprietary data.
2. Run local scheduler smoke checks and GitHub Actions matrix benchmarks where feasible.
3. Record worker count, workload hash, runtime, throughput, warm-up, and CPU baseline comparison.
4. Document when cloud or multi-node capacity is unavailable instead of overclaiming cluster proof.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. CPU cluster benchmark packets exist for representative workloads or blocked runner status is explicit.
2. Distributed execution preserves result envelopes and diagnostics.
3. Docs describe CPU HPC capacity separately from accelerator claims.

## Required Keywords For Validation

`CPU cluster`, `distributed scheduler`, `matrix`, `production workload`, `speedup`, `throughput`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
