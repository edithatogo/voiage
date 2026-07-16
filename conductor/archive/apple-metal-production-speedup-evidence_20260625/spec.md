# Track Specification: Apple Metal Production Speedup Evidence

## Overview

Capture Apple Silicon Metal/MPS speedup against CPU baselines while preserving CPU fallback behavior.

This is a follow-through track in the HPC production evidence lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

Local Apple Silicon runner where available, GitHub Actions fallback, benchmark packets, and evidence manifests.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Run Apple Metal/MPS benchmark packets on Apple Silicon when device access exists.
2. Record CPU baseline, Metal result, warm-up, timing, throughput, device metadata, and workload hash.
3. Emit explicit blocked status in CI when Apple hardware is unavailable.
4. Keep public APIs unchanged and CPU fallback authoritative.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. Apple Metal evidence includes production workload speedup or hardware-unavailable blocked status.
2. The docs stop short of an Apple acceleration claim until speedup evidence passes.
3. Benchmark packets validate schema, result parity, and fallback behavior.

## Required Keywords For Validation

`Apple Metal`, `MPS`, `Apple Silicon`, `speedup`, `CPU fallback`, `blocked status`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
