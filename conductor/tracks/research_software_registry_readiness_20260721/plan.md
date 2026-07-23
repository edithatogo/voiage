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
  adaptation prepared; authenticated submission, editorial review, acceptance,
  and DOI assignment remain human or external gates.
  - [~] [Issue #312](https://github.com/edithatogo/voiage/issues/312) —
    authenticated arXiv submission `7861466` is complete; announcement and a
    permanent arXiv identifier remain external gates.

## Phase 2A: JOSS readiness after arXiv submission [checkpoint: 80af0da]

- [x] Reconcile the JOSS manuscript with the current JOSS paper, screening,
  design-thinking, research-impact, AI-disclosure, and archive requirements.
  (`80af0da`)
- [x] Add a pinned, least-privilege Open Journals/Inara draft build and a
  repository-owned fail-closed JOSS manuscript validator. (`80af0da`)
- [x] Audit reviewer-facing installation, packaging, documentation, examples,
  tests, and release evidence for the Python, Rust, R, and Julia surfaces,
  including the pyOpenSci/rOpenSci partner routes. (`80af0da`)
- [x] Record the submitted arXiv draft as authoritative submission evidence and
  retain announcement, permanent identifier, JOSS submission, review, and
  acceptance as external gates. (`80af0da`)

## Review fixes

- [x] Apply strict Ruff formatting and performance-rule fixes to the JOSS
  validator after the Conductor phase review. (`80af0da`)

## Phase 3: Reconciliation and closeout

- [x] Reconcile Conductor status, release evidence, and external-gate boundaries.
- [x] Run the repository's documented JOSS and package validation workflow.
  (`80af0da`)
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
  repository-ready; funding confirmation, submission-route selection,
  authenticated submission, and editorial review remain human or external.
- Signed v1.0 release: complete at https://github.com/edithatogo/voiage/releases/tag/v1.0.0; live archival, identifier, submission, review, and indexing gates remain external.
- GitHub work hierarchy: #296 is the registry parent; #297--#299 are native
  registry subissues; #312 is the native arXiv subissue of #299 and is present
  in GitHub Project 28.
- JOSS submission package: draft-ready with `paper.md`, `paper.bib`,
  `codemeta.json`, `CITATION.cff`, and `docs/joss-submission-readiness.md`.
  Author affiliations and ORCID are recorded; the funding statement still
  requires author confirmation. Concrete developer-led research use is stated
  without claiming independent adoption.
- arXiv preprint package: canonical authored source is `paper/main.tex`; the
  deterministic, non-submitting readiness pipeline validates TeX Live
  2023/2025, source hygiene, PDF/font integrity, semantic HTML, and independent
  cleaner/collector variants. Authenticated submission `7861466` is complete;
  announcement and the permanent arXiv identifier remain external.
- JOSS submission is explicitly deferred for this execution; no JOSS submission
  or editorial action is claimed.
- JOSS permits an arXiv preprint before, during, or after JOSS submission;
  arXiv timing is therefore not a JOSS blocker.
