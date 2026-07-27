# Implementation Plan

## Phase 1: Baseline and transformation contract [checkpoint: 2a9941c0]

- [x] Capture and classify the exact 223-error baseline in a machine-readable
  audit. (`2a9941c0`)
- [x] Add failing regression tests for normalization, idempotence, state
  preservation, and final full validation. (`2a9941c0`)
- [x] Complete automated review and validation checkpoint for Phase 1.
  (`2a9941c0`)

## Phase 2: Mechanical historical normalization [checkpoint: 2a9941c0]

- [x] Implement the deterministic normalizer and apply it to all registered
  active and archived tracks. (`2a9941c0`)
- [x] Register every orphaned archive directory without deleting or merging
  historical records. (`2a9941c0`)
- [x] Preserve superseded and external follow-ups without false completion
  claims. (`2a9941c0`)
- [x] Complete automated review and validation checkpoint for Phase 2.
  (`2a9941c0`)

## Phase 3: Reconciliation and closeout

- [x] Reconcile the registry, audit, metadata, evidence, task list, and
  changelog. (`2a9941c0`)
- [x] Run focused and full repository validation and confirm the bundled
  Conductor validator reports zero errors. (`2a9941c0`)
- [~] Complete final automated review, archive this track, and retain any
  genuinely ambiguous records as explicit evidence rather than guessed state.
