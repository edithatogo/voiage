# Track Specification: Adjacent Frontier Extensions

## Overview

This track groups the remaining frontier VOI families that are not yet
represented as separate contracts:

- causal-identification, transportability, and external-validity VOI
- data-quality, measurement-error, data-acquisition, privacy, and linkage VOI
- computational VOI and value of model refinement
- expert-elicitation VOI
- evidence-synthesis design VOI

These are adjacent to the main frontier track but deserve explicit contract
scoping so they do not remain a vague backlog note.

## Functional Requirements

1. Define the contract boundaries for each adjacent frontier family.
2. Distinguish the decision value being measured from the operational or
   statistical mechanism that generates the uncertainty.
3. Require maturity metadata and diagnostics on every result payload.
4. Define how each family will graduate from planned to experimental and, later,
   fixture-backed status.
5. Keep the contracts compatible with the shared reporting model and the core
   API contract.

## Non-Functional Requirements

1. Keep each family separable so later implementation tracks can be created
   independently.
2. Keep the contract language-neutral and net-benefit-first.
3. Avoid collapsing different kinds of uncertainty into one generic bucket.

## Acceptance Criteria

1. The adjacent frontier families have explicit contract scopes and task
   breakdowns.
2. The roadmap and frontier track point at this track instead of leaving the
   families as prose-only triage.
3. The contract supports future schemas, fixtures, and maturity metadata.

## Out of Scope

1. Runtime implementations for any of the families.
2. Experimental public APIs before the contracts are stable.
