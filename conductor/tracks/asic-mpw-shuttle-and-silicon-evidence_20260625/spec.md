# Track Specification: ASIC MPW Shuttle And Silicon Evidence

## Overview

Track Tiny Tapeout/SkyWater MPW application, shuttle result, fabricated silicon, and runtime evidence.

This is a follow-through track in the FPGA/ASIC external gates lane. It converts prior readiness, setup, fixture-backed, or visibility evidence into live evidence, stable promotion, production-speedup proof, or an explicit external gate. Completed readiness tracks remain complete unless this track finds a concrete inconsistency.

## Tooling And Execution Boundary

OpenROAD/OpenLane/SKY130 evidence, Tiny Tapeout or SkyWater MPW portals, and silicon runtime manifests.

GitHub Actions and `gh` are preferred for reproducible checks, workflow monitoring, PR tracking, and artifact retrieval. `colab` and `gcloud` may be used only where runtime, quota, billing, and authentication are available. Browser or Chrome automation is allowed only for external portals that require it, and the agent must pause before login-bound irreversible submissions or account actions.

## Functional Requirements

1. Prepare a shuttle submission packet from the existing RTL-to-GDS and pre-silicon evidence.
2. Pause before any Tiny Tapeout, SkyWater MPW, or account-bound portal submission.
3. Track application, review, acceptance, fabrication, delivery, bring-up, and runtime evidence separately.
4. Record fabricated-silicon absence as an external gate without implying production acceleration.

## Non-Functional Requirements

1. Preserve public API compatibility unless a downstream implementation track explicitly approves an additive change.
2. Keep repository-owned evidence separate from external registry, hardware, cloud-quota, or maintainer approval gates.
3. Use deterministic artifacts, hashes, timestamps, and evidence URLs wherever possible.
4. Do not claim stable method status, registry publication, HPC-native acceleration, FPGA runtime, ASIC runtime, or production speedup without direct evidence.
5. Maintain or improve the repository-wide 90 percent coverage gate for any code-bearing implementation slices.

## Acceptance Criteria

1. ASIC shuttle packet exists with links to RTL, GDS/DEF status, timing, area, and testbench evidence.
2. Shuttle and fabricated-silicon states are explicit and evidence-backed.
3. No production ASIC claim is made without silicon runtime and benchmark evidence.

## Required Keywords For Validation

`ASIC`, `Tiny Tapeout`, `SkyWater MPW`, `fabricated silicon`, `RTL-to-GDS`, `external gate`

## Out Of Scope

1. Reopening completed readiness/setup/pre-silicon tracks without a concrete inconsistency.
2. Performing irreversible external submissions, paid cloud actions, registry account changes, or hardware purchases without explicit user approval.
3. Treating blocked external gates as completed work.
4. Weakening existing CI, coverage, contract, or documentation gates.
