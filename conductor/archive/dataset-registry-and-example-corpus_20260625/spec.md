# Track Specification: Dataset Registry And Example Corpus

## Overview

Create deterministic synthetic datasets and real open-data examples for every analysis type and method family.

This cross-cutting track belongs to the comprehensive VOI roadmap follow-through program. It should land before or alongside downstream runtime, registry, dataset, Rust, or HPC work that depends on its decisions.

## Tooling And Execution Boundary

dataset registry JSON, transform scripts, tiny snapshots, hashes, examples, docs, and scheduled refresh workflows.

GitHub Actions and `gh` are preferred for repeatable verification. Heavy optional dependencies, cloud runners, Colab, and browser workflows must stay optional and evidence-gated.

## Functional Requirements

1. Define a dataset registry with source URL, license, citation, transform command, snapshot hash, schema, and method tags.
2. Add deterministic synthetic datasets for unit, property, integration, and e2e examples.
3. Source real open datasets for healthcare, clinical trials, economics, environmental, and policy examples without requiring network in default tests.
4. Keep tiny committed snapshots separate from live refresh scripts and external data terms.

## Non-Functional Requirements

1. Preserve public API compatibility unless an additive change is explicitly approved.
2. Avoid dependency conflicts in the base install; keep heavyweight or experimental backends optional.
3. Maintain the repository-wide 90 percent coverage gate and language-native binding quality gates.
4. Keep external, hardware, cloud, and experimental gates explicit until evidence exists.

## Acceptance Criteria

1. Every analysis family has at least one synthetic example and a mapped real open-data source.
2. Default tests use committed snapshots and do not require live network access.
3. Docs explain data licenses, citations, transforms, and refresh policy.

## Required Keywords For Validation

`synthetic dataset`, `open dataset`, `NHANES`, `MEPS`, `ClinicalTrials.gov`, `World Bank`, `NOAA`, `EPA`

## Out Of Scope

1. Irreversible external submissions, paid cloud actions, or hardware purchases without explicit user approval.
2. Weakening existing CI, coverage, contract, or docs gates.
3. Marking experimental or external-gated work complete without evidence.
