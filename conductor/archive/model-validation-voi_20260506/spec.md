# Track Specification: Model Validation VOI Implementation

## Overview

Implement the model-validation / model-discrepancy VOI frontier surface as a
real runtime feature, not just a schema scaffold. The implementation should
treat validation profiles as an explicit analysis dimension and expose the
value of external validation together with discrepancy-reduction summaries.

The track is anchored to `specs/frontier/validation/v1/` and must stay aligned
with the existing frontier registry, shared maturity conventions, and
deterministic fixture approach.

## Goals

- Add a runtime module for model-validation VOI.
- Expose a public `DecisionAnalysis` wrapper and curated exports.
- Compute profile-specific optima, discrepancy/regret summaries, and the value
  of external validation.
- Preserve deterministic behavior, explicit reporting metadata, and
  fixture-backed conformance.
- Keep the implementation parallelizable, testable, and easy to review in
  small slices.

## Functional Requirements

1. Accept validation-aware net-benefit surfaces shaped as
   `sample x strategy x validation_profile`.
2. Support explicit validation-profile metadata as well as convenience
   profile-name inputs.
3. Compute, at minimum:
   - profile-specific optimal strategies
   - expected net benefit by profile and strategy
   - cross-profile discrepancy or regret summaries
   - value of external validation
   - value of model-discrepancy reduction
   - consensus or robust strategy summaries under profile weights
   - Pareto or non-dominated strategy sets across profiles
4. Emit structured result objects with reporting metadata and maturity fields.
5. Expose the feature from `voiage.methods`, `voiage`, and `DecisionAnalysis`.
6. Add deterministic fixture-backed tests and exact payload comparisons.
7. Add CLI and docs coverage if the feature is surfaced as a user command.

## Non-Functional Requirements

- Keep the implementation fully type-hinted.
- Keep the public API explicit and stable once exported.
- Prefer deterministic calculations over stochastic shortcuts where a
  deterministic surface is possible.
- Keep contract and runtime shapes aligned with the existing frontier scaffold.

## Acceptance Criteria

- The validation VOI runtime module exists and is imported through the curated
  public API.
- The feature passes unit tests, export tests, and fixture-backed contract
  checks.
- The resulting payloads include the shared reporting envelope and maturity
  metadata.
- The roadmap, todo list, changelog, and Conductor registry describe the new
  track accurately.

## Out Of Scope

- Cross-language binding work for this feature family.
- New release automation.
- Non-deterministic approximate methods unless the spec explicitly adds them
  later.
- Reworking unrelated frontier families.
