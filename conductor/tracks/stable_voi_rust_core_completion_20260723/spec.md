# Track Specification: Stable VOI Rust Core Completion

## Overview

Make Rust authoritative for the complete stable decision-VOI surface.

## Requirements

1. Implement generalized/net benefit, expected loss/opportunity loss,
   EVPI/population EVPI, EVPPI, EVSI, ENBS, optimal design, CEAC/CEAF,
   dominance/extended dominance, ICER, and cost-effectiveness summaries.
2. Support validated nested, regression/metamodel, non-parametric,
   moment-matching, and importance-sampling estimators where the registry
   assigns them stable status.
3. Define RNG, convergence, interval, tolerance, tie, scaling, diagnostic, and
   fallback policy in versioned contracts.
4. Publish a supported Rust facade and remove silent Python numerical
   divergence.

## Compatibility and failure policy

Preserve v1 semantics through additive envelopes and adapters. Non-finite,
mis-shaped, unsupported, non-convergent, or invalid invariant requests fail
with typed diagnostics; panics never cross a public boundary.

## Acceptance criteria

Every stable path executes Rust or visibly reports a compatibility path, passes
analytical/property/differential tests and mutation/performance budgets, and
owns versioned fixtures.

## Out of scope

Promotion of experimental estimators without scientific and numerical evidence.

