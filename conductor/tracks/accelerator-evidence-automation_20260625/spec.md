# Track Specification: Accelerator Evidence Automation

## Overview

Standardize notebook/script execution and artifact ingestion for GPU, TPU, Metal, and future accelerator evidence.

This is a follow-through track in the HPC production evidence lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

colab, gh run watch, GitHub Actions artifacts, JSON schema validation, and evidence manifests.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Define a single evidence schema for accelerator benchmark packets and blocked-run reports.
2. Automate Colab notebook execution and artifact retrieval where the CLI supports it.
3. Validate evidence packets for workload hash, device metadata, warm-up, timing, throughput, and fallback fields.
4. Ingest GitHub Actions artifacts into the documented handoff location without hiding failed or blocked runs.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. Automation can validate and index GPU, TPU, Metal, and blocked-run evidence packets.
2. Evidence manifests link source commands, artifacts, timestamps, devices, and benchmark status.
3. All accelerator tracks use the same schema before promotion review.

## Required Keywords For Validation

`colab`, `gh run watch`, `JSON schema`, `artifact ingestion`, `device metadata`, `CPU fallback`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
