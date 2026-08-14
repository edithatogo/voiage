# Implementation Plan

## Merged governance reconciliation

- [x] Record merged scientific contract PR #480, Julia publication-readiness
  PR #561, and governance PR #813 with full head and merge SHAs while
  preserving external registry decisions as pending.

## Phase 1: Readiness and prerequisites

- [x] Confirm scope, rights, licensing, metadata, release, and persistence prerequisites in the parent issue. (`483fe29c`; live readiness audit recorded in the subsequent handoff commit)
- [x] Capture repository-specific validation commands and baseline results. (see `handoff/registry-readiness.json`)

## Phase 2: Registry deliverables

- [x] [Issue #297](https://github.com/edithatogo/voiage/issues/297) — Software
  Heritage snapshot verified as
  `swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`.
- [x] [Issue #298](https://github.com/edithatogo/voiage/issues/298) — SciCrunch
  registration was submitted at 2026-07-27T06:07:06Z after the portal found no
  similar resource and the account holder confirmed accuracy and current terms.
  SciCrunch curation, RRID assignment, and resolver indexing remain external.
- [x] [Issue #299](https://github.com/edithatogo/voiage/issues/299) — JOSS
  adaptation and repository submission package prepared; authenticated
  submission, editorial review, acceptance, and DOI assignment remain human or
  external gates.
  - [x] [Issue #312](https://github.com/edithatogo/voiage/issues/312) —
    prior submission `7861466` is absent from the authenticated dashboard.
    Replacement submission `7870358` exists only as an incomplete start-stage
    draft expiring 9 August 2026; author selection of category and licence,
    file upload, completion, moderation, announcement, and a permanent arXiv
    identifier are explicitly handed off as human or external gates.

## Phase 2A: JOSS readiness after arXiv submission [checkpoint: 80af0da]

- [x] Reconcile the JOSS manuscript with the current JOSS paper, screening,
  design-thinking, research-impact, AI-disclosure, and archive requirements.
  (`80af0da`)
- [x] Add a pinned, least-privilege Open Journals/Inara draft build and a
  repository-owned fail-closed JOSS manuscript validator. (`80af0da`)
- [x] Audit reviewer-facing installation, packaging, documentation, examples,
  tests, and release evidence for the Python, Rust, R, and Julia surfaces,
  including the pyOpenSci/rOpenSci partner routes. (`80af0da`)
- [x] Record the historical submitted arXiv state and preserve later
  authenticated disposition checks without treating a submission number as a
  permanent identifier. (`80af0da`; refreshed 2026-07-26)

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
- [x] Document completed developer research-workflow use through the released
  package and preserve attributable human community engagement, external use,
  or collaborative input as a separate pre-submission gate. The latter remains
  a detailed-review criterion, strong positive pre-review signal, and
  author-selected prerequisite. ([Issue
  #471](https://github.com/edithatogo/voiage/issues/471); human participant
  required; agents, bots, and same-author repositories do not qualify)
- [x] Reconcile the author's explicit JOSS AI-policy affirmation with the
  best-available tool/model inventory without guessing unavailable historical
  identifiers. (`paper/joss-editorial-assurance.json`; `paper.md`)
- [x] Publish, verify, and archive the exact v2 release described by `paper.md`,
  then replace prospective availability wording with observed release,
  provenance, SBOM, digest, clean-installation, and Software Heritage evidence.
  (`cdc40de`; `v2.0.0`; release run `30200134119`; supply-chain run
  `30200921515`; Software Heritage request `2399846`)
- [x] Build and visually inspect the release-bound Open Journals PDF from the
  exact v2 evidence revision. (run `30202496481`; artifact `8632098142`;
  six-page PDF SHA-256
  `132af479c9d76091478459652ff12091d04bd3dd426ef5e90265ec1e4bab3e71`)
- [x] Submit the v2.0.0 conda-forge staged recipe and obtain green lint,
  Linux, macOS, and Windows build evidence. ([conda-forge PR
  #34308](https://github.com/conda-forge/staged-recipes/pull/34308);
  maintainer review and merge remain external)
- [x] Expand the compiled conda recipe from its initial Python 3.12-only
  artifacts to the supported Python 3.12+ build matrix, add installed
  Rust-core and numerical smoke evidence, and verify the revised hosted
  matrix before requesting conda-forge review. (`581fcafb`; staged-recipes
  build `1558265` passes lint and current Python 3.12/3.13 variants on Linux,
  macOS, and Windows; external review requested)
- [x] Submit the Rust C ABI recipe through BinaryBuilder/Yggdrasil and prepare
  the Julia binding, deferred Registrator command, and collision-free TagBot
  automation. Yggdrasil PR #14292 remains open and green; JLL creation,
  Registrator execution, General registry review, and registry acceptance are
  sequential external gates.
- [x] Maintain the locally validated Spack/EasyBuild packaging handoff in
  https://github.com/edithatogo/voiage/issues/622 before any HPC registry or
  curation decision; registry acceptance remains external.

## Phase 2B: Independent simulated JOSS editorial review

- [x] Define a fail-closed 1,000-point JOSS manuscript rubric and run documented,
  independent editor-in-chief, handling-editor, domain, software, reproducibility,
  numerical, accessibility, and sentence-level reviews of `paper.md`.
- [x] Add the explicit 2026 JOSS article contract, deterministic 1,600 ±2%
  target, structured metadata and section validation, claim ledger,
  SourceRight queued-reference sidecars, selected Authentext checks, and
  review-only Textstat workflow.
- [x] Reproduce and remediate the panel's scientific EVSI finding: move the
  validated normal-normal study model into the public package, correct the
  built-in estimator so current, predictive, and posterior calculations use
  one coherent fitted Gaussian prior; retain correlations through a numerically
  guarded joint update; add explicit custom two-loop callbacks; and remove
  stable scientific claims from generic estimators that lack complete
  method-specific validation. Implemented and published in merged v2
  contract PR #480 (`8a49bcf3`), with focused scientific and changed-coverage
  tests passing; remaining JOSS, adoption, AI-attestation, and registry gates
  are external.
- [x] Synthesize and prioritize the panel findings from manuscript purpose and
  structure through paragraph, claim, citation, and sentence-level changes.
- [x] Implement the evidence-supported revisions and rerun the repository-owned
  JOSS, citation-provenance, prose, readability, and rendered-PDF checks.
  The round-nine source passes its local contract at 1,583 words with 18/18
  citations reconciled. PR #522 passed the hosted Open Journals build,
  Textstat report, and six-page visual review. The retained decision-maker
  wording passed its release-bound rebuild and six-page visual review in PR
  #529; human source review remains pending.
- [x] Return the revision to the full panel, remediate every supported
  manuscript finding, and record snapshot scores as diagnostic evidence rather
  than artificial acceptance thresholds; report external JOSS screening gates
  separately.

## Phase 3: Reconciliation and closeout

- [x] Reconcile Conductor status, release evidence, and external-gate boundaries.
- [x] Run the repository's documented JOSS and package validation workflow.
  (`80af0da`)
- [x] Archive this track after all automatable work is complete and every
  remaining external gate is explicit. Repository completion does not close
  issues or claim registry, journal, archive, or publication acceptance.

## Final review fixes

- [x] Synchronize the v1 programme active/archive baseline and the
  changelog-bound distributional evidence digest after archival. (`3923309e`)
- [x] Record merged archive-delivery PR #880 across every Conductor projection,
  refresh the live issue, Project 28, conda-forge, and Yggdrasil evidence, and
  reject cross-reference paths outside the repository track roots. (`c0eb4664`)

## Current evidence boundary

- Repository readiness audit: complete at 2026-07-22T00:41:31Z.
- Signed public release evidence: complete; `v2.0.0` was published at
  https://github.com/edithatogo/voiage/releases/tag/v2.0.0 on 2026-07-26T11:41:47Z.
  The release includes `SHA256SUMS`, source, and macOS, Linux, and Windows
  wheels; PyPI and TestPyPI publish version 2.0.0, and the four public Rust
  crates publish version 2.0.0.
- Software Heritage archival: complete with request `2399846`, a full visit,
  and snapshot
  `swh:1:snp:31f89375852737bb9eb62ebc03fadfbc7ff70c2d`.
- RRID route: the SciCrunch General Resource registration answers, optional
  metadata, release/archive evidence, authoritative no-match duplicate check,
  account declarations, final submission, confirmation page, and fail-closed
  validator are complete. Curation, RRID assignment, and resolver indexing
  remain external.
- JOSS route: direct JOSS is selected for the Rust-centred polyglot package.
  The canonical arXiv LaTeX preprint and JOSS adaptation are repository-ready;
  developer research use and the author-confirmed AI attestation are recorded.
  Non-author engagement, authenticated submission, and editorial review remain
  human or external.
  The author confirmed funding, competing-interest, affiliation, and ORCID
  metadata on 24 July 2026.
- Signed v2.0 release and matching archive: complete; remaining identifier,
  submission, review, adoption, and indexing gates remain external.
- GitHub work hierarchy: #296 is the registry parent; #297--#299 are native
  registry subissues; #312 is the native arXiv subissue of #299 and is present
  in GitHub Project 28; #471 is the native independent-validation subissue of
  #299.
- JOSS submission package: delivered by merged PR #480 with `paper.md`,
  `paper.bib`, `paper/health-example-methods.md`,
  `paper/reproduction-manifest.json`, `codemeta.json`, `CITATION.cff`, and
  `docs/release/joss-submission-readiness.md`. The fixed-seed example and
  independent benchmarks provide specific reproducible near-term-significance
  evidence without claiming independent adoption. Issue #471 records the
  separate non-author engagement risk.
- arXiv preprint package: canonical authored source is `paper/main.tex`; the
  deterministic, non-submitting readiness pipeline validates TeX Live
  2023/2025, source hygiene, PDF/font integrity, semantic HTML, and independent
  cleaner/collector variants. Prior submission `7861466` is absent from the
  authenticated dashboard. Replacement submission `7870358` is incomplete at
  the start stage and expires on 9 August 2026; it is not submission evidence.
- Direct JOSS submission is authorised by the author but remains unperformed
  until issue #471 contains genuine human engagement evidence and the
  author-preferred arXiv announcement/permanent-identifier boundary is
  resolved. The exact v2 release, AI affirmation, developer research use, and
  release-bound PDF review are recorded.
- JOSS permits an arXiv preprint before, during, or after JOSS submission;
  arXiv timing is therefore not a JOSS blocker.
