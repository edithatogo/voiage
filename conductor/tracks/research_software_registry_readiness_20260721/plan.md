# Implementation Plan

## Phase 1: Readiness and prerequisites

- [x] Confirm scope, rights, licensing, metadata, release, and persistence prerequisites in the parent issue. (`483fe29c`; live readiness audit recorded in the subsequent handoff commit)
- [x] Capture repository-specific validation commands and baseline results. (see `handoff/registry-readiness.json`)

## Phase 2: Registry deliverables

- [ ] [Issue #297](https://github.com/edithatogo/voiage/issues/297)
- [ ] [Issue #298](https://github.com/edithatogo/voiage/issues/298)
- [ ] [Issue #299](https://github.com/edithatogo/voiage/issues/299)

## Phase 3: Reconciliation and closeout

- [ ] Reconcile Conductor status, issue state, project state, and external evidence.
- [ ] Run the repository's documented validation workflow.
- [ ] Archive this track only after all automatable work is complete and every remaining external gate is explicit.

## Current evidence boundary

- Repository readiness audit: complete at 2026-07-22T00:41:31Z.
- Signed public release evidence: complete; `v1.0.0` was published at
  https://github.com/edithatogo/voiage/releases/tag/v1.0.0 on 2026-07-22T06:35:22Z.
  The release includes `SHA256SUMS`, source, and macOS, Linux, and Windows
  wheels; PyPI mirrors the source release at https://pypi.org/project/voiage/1.0.0/.
- Software Heritage origin lookup: HTTP 404; no pre-v1 ingestion requested.
- RRID route: SciCrunch General Resource registration; assignment and curation external.
- JOSS route: paper/author/impact readiness remains pending; editorial review external.
- Signed v1.0 release: complete at https://github.com/edithatogo/voiage/releases/tag/v1.0.0; live archival, identifier, submission, review, and indexing gates remain external.
- Software Heritage archival request: accepted as request `2397350` for visit
  `1`; the request is scheduled and currently has no snapshot SWHID. Continue
  polling the request before recording archival completion.
- JOSS submission package: draft-ready with `paper.md`, `paper.bib`,
  `codemeta.json`, `CITATION.cff`, and `docs/joss-submission-readiness.md`.
  Author affiliations/ORCID, funding/conflict declarations, and concrete
  research-impact evidence still require author confirmation.
- JOSS permits an arXiv preprint before, during, or after JOSS submission;
  arXiv timing is therefore not a JOSS blocker.
