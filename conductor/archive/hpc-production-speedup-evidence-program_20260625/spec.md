# Track Specification: HPC Production Speedup Evidence Program

## Overview

Coordinate production-scale accelerator evidence without overclaiming HPC-native status from setup, visibility, or parity alone.

This is a follow-through track in the HPC production evidence lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

GitHub Actions artifacts, benchmark workflow, evidence manifests, gh, colab, and gcloud when available.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Define benchmark packet schemas for production workloads, warm-up, timing, throughput, device metadata, and CPU fallback.
2. Coordinate CPU cluster, Apple Metal, discrete GPU, TPU, FPGA, and ASIC evidence tracks.
3. Use GitHub Actions artifacts and gh run monitoring for reproducible evidence capture.
4. Keep HPC-native claims blocked until production workload speedup evidence is captured and reviewed.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. A production-speedup evidence manifest exists for every backend with passed, blocked, or not-available status.
2. Docs distinguish setup, visibility, parity, speedup, and production acceleration.
3. No accelerator backend is promoted without CPU comparison and benchmark packet evidence.

## Required Keywords For Validation

`production speedup`, `benchmark packet`, `warm-up`, `throughput`, `CPU fallback`, `HPC-native`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
