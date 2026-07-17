# Track Specification: FPGA Physical Board Runtime Evidence

## Overview

Follow pre-silicon CI with real FPGA board runtime when hardware access exists.

This is a follow-through track in the FPGA/ASIC external gates lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

Board-specific toolchain, bitstream/runtime packet, CPU parity checks, and benchmark manifests.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Select a physical board target and document its toolchain, license, runner, and access assumptions.
2. Build or map the existing fixed-point EVPI-style kernel to the board runtime path.
3. Capture bitstream, runtime, CPU parity, timing, throughput, and workload hash evidence when hardware exists.
4. Record hardware-unavailable status as an external gate without reopening pre-silicon completion.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. Physical FPGA evidence exists or the specific board/access gate is documented.
2. Pre-silicon CI evidence remains preserved and linked as prerequisite evidence.
3. No production FPGA acceleration claim is made without board runtime and benchmark evidence.

## Required Keywords For Validation

`FPGA`, `physical board`, `bitstream`, `runtime packet`, `CPU parity`, `external gate`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
