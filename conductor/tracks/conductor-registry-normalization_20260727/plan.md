# Implementation Plan

## Phase 1: Baseline and transformation contract

- [ ] Capture and classify the exact 223-error baseline in a machine-readable
  audit.
- [ ] Add failing regression tests for normalization, idempotence, state
  preservation, and final full validation.
- [ ] Complete automated review and validation checkpoint for Phase 1.

## Phase 2: Mechanical historical normalization

- [ ] Implement the deterministic normalizer and apply it to all registered
  active and archived tracks.
- [ ] Register every orphaned archive directory without deleting or merging
  historical records.
- [ ] Preserve superseded and external follow-ups without false completion
  claims.
- [ ] Complete automated review and validation checkpoint for Phase 2.

## Phase 3: Reconciliation and closeout

- [ ] Reconcile the registry, audit, metadata, evidence, task list, and
  changelog.
- [ ] Run focused and full repository validation and confirm the bundled
  Conductor validator reports zero errors.
- [ ] Complete final automated review, archive this track, and retain any
  genuinely ambiguous records as explicit evidence rather than guessed state.
