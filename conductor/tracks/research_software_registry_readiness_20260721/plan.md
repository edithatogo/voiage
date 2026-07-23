# Implementation Plan

## Phase 1: Readiness and prerequisites

- [x] Confirm scope, rights, licensing, metadata, release, and persistence prerequisites in the parent issue. (`483fe29c`; live readiness audit recorded in the subsequent handoff commit)
- [x] Capture repository-specific validation commands and baseline results. (see `handoff/registry-readiness.json`)

## Phase 2: Registry deliverables

- [x] [Issue #297](https://github.com/edithatogo/voiage/issues/297) — Software
  Heritage snapshot verified as
  `swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`.
- [~] [Issue #298](https://github.com/edithatogo/voiage/issues/298) — SciCrunch
  registration prepared; RRID assignment and curation remain external.
- [~] [Issue #299](https://github.com/edithatogo/voiage/issues/299) — JOSS
  adaptation prepared, but submission intentionally deferred until the
  arXiv-first author gate is complete.
  - [~] [Issue #312](https://github.com/edithatogo/voiage/issues/312) — final
    manuscript review is in PR #311; author approval, metadata decisions, and
    authenticated arXiv submission remain human or external gates.

## Phase 3: Reconciliation and closeout

- [x] Reconcile Conductor status, release evidence, and external-gate boundaries.
- [ ] Run the repository's documented validation workflow.
- [ ] Archive this track only after all automatable work is complete and every remaining external gate is explicit.

## Current evidence boundary

- Repository readiness audit: complete at 2026-07-22T00:41:31Z.
- Signed public release evidence: complete; `v1.0.0` was published at
  https://github.com/edithatogo/voiage/releases/tag/v1.0.0 on 2026-07-22T06:35:22Z.
  The release includes `SHA256SUMS`, source, and macOS, Linux, and Windows
  wheels; PyPI mirrors the source release at https://pypi.org/project/voiage/1.0.0/.
- Software Heritage archival: complete with request `2397350`, full visit `1`,
  and snapshot
  `swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`.
- RRID route: SciCrunch General Resource registration; assignment and curation external.
- JOSS route: the canonical arXiv LaTeX preprint and JOSS adaptation are
  repository-ready; author/impact confirmation and editorial review remain
  external.
- Signed v1.0 release: complete at https://github.com/edithatogo/voiage/releases/tag/v1.0.0; live archival, identifier, submission, review, and indexing gates remain external.
- GitHub work hierarchy: #296 is the registry parent; #297--#299 are native
  registry subissues; #312 is the native arXiv subissue of #299 and is present
  in GitHub Project 28.
- JOSS submission package: draft-ready with `paper.md`, `paper.bib`,
  `codemeta.json`, `CITATION.cff`, and `docs/joss-submission-readiness.md`.
  Author affiliations/ORCID, funding/conflict declarations, and concrete
  research-impact evidence still require author confirmation.
- arXiv preprint package: canonical authored source is `paper/main.tex`; the
  deterministic, non-submitting readiness pipeline validates TeX Live
  2023/2025, source hygiene, PDF/font integrity, semantic HTML, and independent
  cleaner/collector variants. Category, license, endorsement, and authenticated
  upload remain human gates.
- JOSS submission is explicitly deferred for this execution; no JOSS submission
  or editorial action is claimed.
- JOSS permits an arXiv preprint before, during, or after JOSS submission;
  arXiv timing is therefore not a JOSS blocker.
