# Specification: Conductor GitHub Cross-Reference Reconciliation

## Overview

Make Conductor-to-GitHub traceability complete, bidirectional, and
machine-verifiable for every VOIAGE track. Historical grouping ledgers remain
the parent records, while each track receives its own repository issue,
Project 28 item, and evidence-based pull-request references.

## Requirements

1. Every directory under `conductor/tracks/` and `conductor/archive/` must have
   one unique GitHub issue.
2. The proposed standardized-ingestion track in PR #334 must remain visible
   while it is not yet present on `main`.
3. Completed archive issues must be closed, appear as Done in Project 28, and
   be native sub-issues of the applicable historical or programme parent.
4. Active track issues must remain open and retain their existing native
   sub-issue hierarchy.
5. Pull requests may be associated only when evidenced by a commit recorded in
   the track, Git history for the track path, or an explicit track reference.
6. A completed track with no provable pull request must say so explicitly;
   the reconciliation must not guess from similar titles.
7. Repository metadata, track indexes, and a central machine-readable manifest
   must agree.
8. Automated tests must fail when a local track is missing, an issue is reused,
   an archive issue is not closed, or a proposed track lacks its source PR.

## Acceptance criteria

- All 126 archived tracks, all active tracks, and the proposed ingestion track
  have unique issue records in Project 28.
- The 122 historical archives are native children of ledgers
  `vop_poc_nz#29` through `#33`; the four later assurance/release archives use
  their current programme parents.
- Each completed track records all pull requests supported by the stated
  evidence policy, or records that no such pull request was found.
- `python scripts/validate_conductor_github_cross_references.py .` and its
  focused tests pass.
- The original dirty manuscript worktree remains untouched.

## Non-functional constraints

- GitHub mutations must be idempotent through a stable hidden track marker.
- Repository paths and URLs must be portable and must target `main` unless a
  proposed track explicitly records another source PR.
- External, hardware, registry, publication, and human gates retained by old
  tracks must not be reclassified as complete.

## Out of scope

- Rewriting legacy track plans or changing their scientific acceptance gates.
- Claiming a pull request association without commit/path evidence.
- Merging PR #334 or this reconciliation PR.

## Authoritative inputs

- `AGENTS.md`
- `conductor/tracks.md`
- Every directory under `conductor/tracks/` and `conductor/archive/`
- Git commit history for each track
- GitHub Project 28 and native issue hierarchy
- User direction recorded on 2026-07-24
