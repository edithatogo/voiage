# Track Specification: Dynamic Real-Options VOI

## Overview

This track defines the VOI contract for decisions where information arrives
over time and the option to wait, stage, or exercise a policy has value.
Dynamic real-options VOI treats delay, irreversibility, and policy lock-in as
first-class features of the decision problem instead of as hidden modeling
assumptions.

## Functional Requirements

1. Define the input contract for staged decisions, including evidence arrival
   times, option-exercise rules, and timing of decisions.
2. Define how delay, irreversibility, and policy lock-in affect the value
   calculation.
3. Define result payloads for option value, waiting value, timing sensitivity,
   and policy-path comparisons.
4. Require diagnostics and maturity metadata so consumers can tell whether a
   result is experimental, approximate, or fixture-backed.
5. Add reproducible deterministic examples and fixtures once the contract is
   stable enough to validate.

## Non-Functional Requirements

1. Keep the contract net-benefit-first and compatible with the core API.
2. Keep the contract deterministic under seeded fixtures.
3. Avoid introducing heavy new dependencies in the base install.

## Acceptance Criteria

1. A versioned schema/fixture plan exists for dynamic real-options inputs and
   outputs.
2. The track decomposes the dynamic timing logic into explicit phases or
   tasks.
3. The contract records diagnostics, reproducibility, and maturity metadata.

## Out of Scope

1. Implementing the runtime algorithm in the same track if the contract must be
   stabilized first.
2. Replacing sequential VOI.
3. Broad general-purpose stochastic control tooling beyond VOI-focused option
   value.
