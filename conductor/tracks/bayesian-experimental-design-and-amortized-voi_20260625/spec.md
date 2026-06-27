# Track Specification: Bayesian Experimental Design And Amortized VOI

## Overview

Add recommended bleeding-edge VOI methods for expected information gain, Bayesian optimal experimental design, active learning, simulation-based inference, and amortized EVSI.

This cross-cutting track belongs to the comprehensive VOI roadmap follow-through program. It should land before or alongside downstream runtime, registry, dataset, Rust, or HPC work that depends on its decisions.

## Tooling And Execution Boundary

optional JAX/NumPyro-style backends, synthetic fixtures, property tests, profiling, docs, and dependency guards.

GitHub Actions and `gh` are preferred for repeatable verification. Heavy optional dependencies, cloud runners, Colab, and browser workflows must stay optional and evidence-gated.

## Functional Requirements

1. Define optional-backend interfaces for expected information gain and Bayesian optimal experimental design without adding heavy base dependencies.
2. Implement synthetic fixtures and examples that prove behavior before real-data examples widen.
3. Profile estimator variance, compile overhead, and approximation tradeoffs.
4. Document experimental maturity, dependency extras, and promotion gates clearly.

## Non-Functional Requirements

1. Preserve public API compatibility unless an additive change is explicitly approved.
2. Avoid dependency conflicts in the base install; keep heavyweight or experimental backends optional.
3. Maintain the repository-wide 90 percent coverage gate and language-native binding quality gates.
4. Keep external, hardware, cloud, and experimental gates explicit until evidence exists.

## Acceptance Criteria

1. Bleeding-edge methods are available behind explicit optional dependencies or experimental flags.
2. Examples and tests cover expected information gain, active learning, and amortized EVSI workflows.
3. Docs keep experimental status and promotion gates explicit.

## Required Keywords For Validation

`expected information gain`, `Bayesian optimal experimental design`, `active learning`, `amortized EVSI`, `simulation-based inference`

## Out Of Scope

1. Irreversible external submissions, paid cloud actions, or hardware purchases without explicit user approval.
2. Weakening existing CI, coverage, contract, or docs gates.
3. Marking experimental or external-gated work complete without evidence.
