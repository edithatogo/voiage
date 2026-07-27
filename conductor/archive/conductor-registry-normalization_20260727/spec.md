# Conductor Registry Normalization

## Overview

Normalize historical Conductor records to the current repository schema without
rewriting implementation history or converting external, superseded, or
follow-up work into false completion claims.

The authoritative baseline is the bundled Conductor validator at
`/Users/doughnut/.codex/skills/conductor/scripts/validate_conductor.py`, run in
`full` mode against repository revision
`d514c3b98ccf6187e5360519e73656fcb5fed39c`. It reports 223 errors and zero
warnings.

## Requirements

1. Capture every baseline finding in a machine-readable audit.
2. Apply deterministic transformations for current metadata keys, supported
   status/type values, UTC timestamps, directory-aligned track IDs, required
   index links, and complete archive registration.
3. Preserve unchecked historical or external follow-ups as explicit
   non-acceptance prose. Do not mark the underlying external activity complete.
4. Preserve superseded outcomes explicitly while using the current lifecycle
   status vocabulary.
5. Add regression tests for idempotence, state-boundary preservation, archive
   coverage, and a zero-error final validator result.
6. Reconcile the central registry, Conductor plan, metadata, evidence, task
   list, and changelog.

## Acceptance criteria

- The bundled full validator reports zero errors.
- Every active and archived track directory is registered exactly once.
- Re-running the normalizer produces no file changes.
- No external submission, publication, review, hardware, speedup, or human
  approval is newly represented as completed.
- Every transformed file remains valid UTF-8, Markdown, or JSON as applicable.
- Focused tests, repository Conductor tests, the repository harness, and prose
  lint pass.

## Non-functional constraints

- Transformations must be deterministic and reviewable.
- Existing implementation evidence, links, prose, and unknown metadata fields
  must be preserved.
- No historical track directory may be deleted.
- The programme must not alter software runtime behaviour.

## External gates

None. This programme normalizes repository records only. Any external gates
already recorded by historical tracks remain external.

## Out of scope

- Reopening or implementing legacy product work.
- Deciding whether an external registry, publication, hardware, or adoption
  outcome has occurred.
- Reconstructing absent historical commit notes or evidence.
