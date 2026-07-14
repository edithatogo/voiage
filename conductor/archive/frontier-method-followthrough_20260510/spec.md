# Track Specification: Frontier Method Followthrough

## Overview

This track completes the remaining frontier-method follow-through identified in
the roadmap. It is a coordination track for the remaining experimental and
planned VOI surfaces, with clear separation between stable follow-through and
new exploratory scope.

## Functional Requirements

1. Define the remaining frontier-method tasks that are still open.
2. Separate stable follow-through work from exploratory work.
3. Keep the experimental surfaces explicit about maturity and limitations.
4. Preserve the existing public behavior of completed frontier methods.
5. Keep the remaining work organized so it can be implemented in smaller child
   phases if needed.

## Non-Functional Requirements

1. Keep the track bounded so it does not absorb unrelated contract work.
2. Preserve deterministic test coverage for the surfaces it touches.
3. Avoid broad API churn unless a sub-track explicitly requires it.

## Acceptance Criteria

1. The open frontier-method items are explicitly enumerated.
2. The work is split into clear sub-phases or child tasks.
3. Existing completed frontier methods remain stable.
4. Documentation and tests reflect the remaining frontier scope.

## Out of Scope

1. Core contract and polyglot fixture work.
2. CLI stub cleanup unrelated to frontier methods.
3. Rewriting completed frontier implementations.
