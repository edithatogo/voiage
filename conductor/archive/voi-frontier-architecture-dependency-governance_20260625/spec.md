# Track Specification: VOI Frontier Architecture And Dependency Governance

## Overview

Define the final frontier VOI architecture, backend boundaries, dependency policy, method maturity taxonomy, and non-conflicting implementation sequence.

This cross-cutting track belongs to the comprehensive VOI roadmap follow-through program. It should land before or alongside downstream runtime, registry, dataset, Rust, or HPC work that depends on its decisions.

## Tooling And Execution Boundary

pyproject, tox, Rust crate metadata, frontier schemas, dependency audit, and Conductor setup docs.

GitHub Actions and `gh` are preferred for repeatable verification. Heavy optional dependencies, cloud runners, Colab, and browser workflows must stay optional and evidence-gated.

## Functional Requirements

1. Audit Python, Rust, binding, frontend-docs, and accelerator dependencies for conflicts before implementation tracks widen.
2. Define the method maturity taxonomy and backend boundary for stable, experimental, fixture-backed, and runtime-complete methods.
3. Document how optional bleeding-edge dependencies avoid base-install conflicts.
4. Update Conductor setup and developer docs before downstream implementation changes.

## Non-Functional Requirements

1. Preserve public API compatibility unless an additive change is explicitly approved.
2. Avoid dependency conflicts in the base install; keep heavyweight or experimental backends optional.
3. Maintain the repository-wide 90 percent coverage gate and language-native binding quality gates.
4. Keep external, hardware, cloud, and experimental gates explicit until evidence exists.

## Acceptance Criteria

1. Architecture and dependency docs define a conflict-free implementation path.
2. Maturity labels and backend ownership rules are documented and testable.
3. Downstream tracks can proceed without choosing dependency policy ad hoc.

## Required Keywords For Validation

`architecture`, `dependency`, `maturity taxonomy`, `backend boundary`, `no conflicts`

## Out Of Scope

1. Irreversible external submissions, paid cloud actions, or hardware purchases without explicit user approval.
2. Weakening existing CI, coverage, contract, or docs gates.
3. Marking experimental or external-gated work complete without evidence.
