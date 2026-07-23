# Implementation Plan

## Phase 1: Inventory and evidence

- [x] Inventory all active, archived, and proposed tracks and preserve legacy
  metadata IDs as aliases where they differ from directory IDs.
- [x] Derive pull-request associations from recorded commits, explicit track
  references, and track-path Git history.

## Phase 2: GitHub reconciliation

- [x] Create one idempotently marked issue for every completed track.
- [x] Add every completed issue to Project 28, attach its native parent, and
  close it only after the hierarchy link succeeds.
- [x] Reuse the existing active registry issue hierarchy and ingestion epic.

## Phase 3: Repository contract

- [x] Add the central cross-reference manifest, per-track metadata/index links,
  and automated validator tests.
- [x] Run focused tests, repository harness checks, full `tox`, and Conductor
  validation; retain the pre-existing 222-error full-validator baseline.
- [x] Self-review the complete diff and verify the live GitHub hierarchy and
  Project 28 state.

## Phase 4: Hosted handoff

- [~] Push the signed branch, open a pull request, and record it in the
  manifest.
- [ ] Monitor required hosted checks to a terminal state.
- [ ] Keep issue closure and merge as explicit post-merge boundaries.
