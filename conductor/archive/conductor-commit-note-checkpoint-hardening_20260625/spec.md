# Track Specification: Conductor Commit Note And Checkpoint Hardening

## Overview

Harden Conductor setup so every track task requires commits, commit notes, git notes, plan SHA updates, phase checkpoints, and GitHub Actions monitoring.

This cross-cutting track belongs to the comprehensive VOI roadmap follow-through program. It should land before or alongside downstream runtime, registry, dataset, Rust, or HPC work that depends on its decisions.

## Tooling And Execution Boundary

Conductor workflow docs, tests, git notes, gh run monitoring, and plan validation.

GitHub Actions and `gh` are preferred for repeatable verification. Heavy optional dependencies, cloud runners, Colab, and browser workflows must stay optional and evidence-gated.

## Functional Requirements

1. Update workflow wording to require task commits, git notes, plan SHA updates, and checkpoint commits.
2. Add tests that fail when generated tracks omit commit-note or phase-checkpoint tasks.
3. Document how blocked external evidence is recorded without marking a track complete.
4. Preserve non-interactive commands and GitHub Actions monitoring as default verification behavior.

## Non-Functional Requirements

1. Preserve public API compatibility unless an additive change is explicitly approved.
2. Avoid dependency conflicts in the base install; keep heavyweight or experimental backends optional.
3. Maintain the repository-wide 90 percent coverage gate and language-native binding quality gates.
4. Keep external, hardware, cloud, and experimental gates explicit until evidence exists.

## Acceptance Criteria

1. Workflow and tests enforce commit notes, git notes, plan SHA updates, and phase checkpoints.
2. Generated tracks follow the hardened template.
3. External gates cannot be closed without evidence and review.

## Required Keywords For Validation

`commit notes`, `git notes`, `short commit SHA`, `phase checkpoint`, `GitHub Actions`

## Out Of Scope

1. Irreversible external submissions, paid cloud actions, or hardware purchases without explicit user approval.
2. Weakening existing CI, coverage, contract, or docs gates.
3. Marking experimental or external-gated work complete without evidence.
