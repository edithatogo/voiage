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
- [x] Reconcile registry-readiness contract tests with the expanded command
  evidence and completed authenticated arXiv submission. (`f40c2da`)
- [x] Replace mismatched foundational bibliography records with authoritative
  source metadata and reject placeholder author lists in JOSS preflight.
  (`c37c78e`)
- [x] Select direct JOSS review for the Rust-centred polyglot package and
  publish a bounded independent-validation protocol. (issue #471)
- [x] Extend the JOSS preflight to reconcile CFF and CodeMeta release metadata
  and trigger hosted JOSS validation for either metadata file. (`2ba6e854`)
- [~] Obtain attributable human community engagement, external use, or
  collaborative-input evidence before direct JOSS submission. ([Issue
  #471](https://github.com/edithatogo/voiage/issues/471); human participant
  required; agents, bots, and same-author repositories do not qualify)
- [ ] Obtain the author's explicit JOSS AI-policy affirmation and bind every
  tool/model version that can be established without guessing.
- [ ] Publish, verify, and archive the exact v2 release described by `paper.md`,
  then replace prospective availability wording with observed release,
  provenance, SBOM, digest, clean-installation, and Software Heritage evidence.

## Phase 2B: Independent simulated JOSS editorial review

- [x] Define a fail-closed 1,000-point JOSS manuscript rubric and run documented,
  independent editor-in-chief, handling-editor, domain, software, reproducibility,
  numerical, accessibility, and sentence-level reviews of `paper.md`.
- [x] Add the explicit 2026 JOSS article contract, deterministic 1,600 ±2%
  target, structured metadata and section validation, claim ledger,
  SourceRight queued-reference sidecars, selected Authentext checks, and
  review-only Textstat workflow.
- [~] Reproduce and remediate the panel's scientific EVSI finding: move the
  validated normal-normal study model into the public package, correct the
  built-in estimator so current, predictive, and posterior calculations use
  one coherent fitted Gaussian prior; retain correlations through a numerically
  guarded joint update; add explicit custom two-loop callbacks; and remove
  stable scientific claims from generic estimators that lack complete
  method-specific validation.
- [x] Synthesize and prioritize the panel findings from manuscript purpose and
  structure through paragraph, claim, citation, and sentence-level changes.
- [x] Implement the evidence-supported revisions and rerun the repository-owned
  JOSS, citation-provenance, prose, readability, and rendered-PDF checks.
  The current local contract passes at 1,628 words with 19/19 citations
  reconciled; the prior hosted Open Journals artifact passed visual review, and
  the current editorial revision is awaiting a fresh hosted build and
  inspection; human source review remains pending.
- [x] Return the revision to the full panel, remediate every supported
  manuscript finding, and record snapshot scores as diagnostic evidence rather
  than artificial acceptance thresholds; report external JOSS screening gates
  separately.

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
- JOSS route: direct JOSS is selected for the Rust-centred polyglot package.
  The canonical arXiv LaTeX preprint and JOSS adaptation are repository-ready;
  exact v2 publication, human engagement, author-confirmed AI attestation,
  authenticated submission, and editorial review remain human or external.
  The author confirmed funding, competing-interest, affiliation, and ORCID
  metadata on 24 July 2026.
- Signed v1.0 release: complete at https://github.com/edithatogo/voiage/releases/tag/v1.0.0; live archival, identifier, submission, review, and indexing gates remain external.
- GitHub work hierarchy: #296 is the registry parent; #297--#299 are native
  registry subissues; #312 is the native arXiv subissue of #299 and is present
  in GitHub Project 28; #471 is the native independent-validation subissue of
  #299.
- JOSS submission package: under active review in PR #480 with `paper.md`,
  `paper.bib`, `paper/health-example-methods.md`,
  `paper/reproduction-manifest.json`, `codemeta.json`, `CITATION.cff`, and
  `docs/release/joss-submission-readiness.md`. The fixed-seed example and
  independent benchmarks provide specific reproducible near-term-significance
  evidence without claiming independent adoption. Issue #471 records the
  separate single-author human-engagement gate.
- arXiv preprint package: canonical authored source is `paper/main.tex`; the
  deterministic, non-submitting readiness pipeline validates TeX Live
  2023/2025, source hygiene, PDF/font integrity, semantic HTML, and independent
  cleaner/collector variants. Authenticated submission `7861466` is complete;
  announcement and the permanent arXiv identifier remain external.
- Direct JOSS submission is authorised by the author but remains unperformed
  until issue #471 contains genuine human engagement evidence, the exact v2
  release is archived, the AI attestation is confirmed, the final PDF is
  reviewed, and the author-preferred arXiv announcement/permanent-identifier
  boundary is resolved.
- JOSS permits an arXiv preprint before, during, or after JOSS submission;
  arXiv timing is therefore not a JOSS blocker.
