# Track Specification: Threshold, Tipping-Point, And Robust VOI Implementation

## Overview

Implement the threshold, tipping-point, and robust / ambiguity-aware VOI
frontier surface as a real runtime feature. The implementation should treat
threshold profiles as an explicit analysis dimension and expose the value of
switching or reversing decisions under alternative threshold regimes.

The track is anchored to `specs/frontier/threshold/v1/` and must stay aligned
with the existing frontier registry, shared maturity conventions, and
deterministic fixture approach.

## Goals

- Add a runtime module for threshold and robust VOI.
- Expose a public `DecisionAnalysis` wrapper and curated exports.
- Compute profile-specific optima, threshold-crossing summaries, tipping-point
  behavior, and robust decision rules.
- Preserve deterministic behavior, explicit reporting metadata, and
  fixture-backed conformance.
- Keep the implementation parallelizable, testable, and easy to review in
  small slices.

## Functional Requirements

1. Accept threshold-aware net-benefit surfaces shaped as
   `sample x strategy x threshold_profile`.
2. Support explicit threshold-profile metadata as well as convenience
   profile-name inputs.
3. Compute, at minimum:
   - profile-specific optimal strategies
   - expected net benefit by profile and strategy
   - threshold-crossing probability summaries
   - tipping-point or decision-reversal matrices
   - robust strategy summaries under ambiguity weights
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

- The threshold / robust VOI runtime module exists and is imported through the
  curated public API.
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
