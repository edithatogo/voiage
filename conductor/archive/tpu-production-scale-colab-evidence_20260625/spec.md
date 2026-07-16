# Track Specification: TPU Production-Scale Colab Evidence

## Overview

Extend Colab v5e visibility and EVPI parity into production-scale TPU benchmark packets.

This is a follow-through track in the HPC production evidence lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

colab TPU runtime, gcloud when quota/project access exists, gh artifact tracking, and benchmark manifests.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Reuse the existing Colab notebook path and extend it with production-sized TPU workloads.
2. Capture compile overhead, transfer overhead, warm-up, repeated timings, throughput, and CPU baseline comparisons.
3. Use gcloud only when project, quota, billing, and authentication are available.
4. Record Colab quota, runtime, or TPU allocation failures as explicit blocked states.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. TPU evidence packets prove speedup or document a concrete availability gate.
2. Compact v5e parity evidence remains cited only as visibility/parity evidence.
3. Docs and registry notes do not claim production TPU acceleration until this track passes.

## Required Keywords For Validation

`TPU`, `Colab`, `v5e`, `gcloud`, `production-scale`, `speedup`, `EVPI parity`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
