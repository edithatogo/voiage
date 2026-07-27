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
    prior submission `7861466` is absent from the authenticated dashboard.
    Replacement submission `7870358` exists only as an incomplete start-stage
    draft expiring 9 August 2026; author selection of category and licence,
    file upload, completion, moderation, announcement, and a permanent arXiv
    identifier remain human or external gates.

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
- [~] Document completed research-workflow use and obtain attributable human
  community engagement, external use, or collaborative-input evidence before
  direct JOSS submission. The former is a hard pre-review gate; the latter is
  a detailed-review criterion, strong positive pre-review signal, and
  author-selected prerequisite. ([Issue
  #471](https://github.com/edithatogo/voiage/issues/471); human participant
  required; agents, bots, and same-author repositories do not qualify)
- [ ] Obtain the author's explicit JOSS AI-policy affirmation and bind every
  tool/model version that can be established without guessing.
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
- [~] Publish the Rust C ABI through BinaryBuilder/Yggdrasil, consume the
  generated JLL from the Julia binding, and initiate subdirectory registration
  in Julia General with collision-free TagBot automation. BinaryBuilder,
  General registry bots, and registry maintainers remain external acceptance
  gates.
- [~] Maintain the successful r-universe publication and complete the CRAN
  submission path. r-universe publishes `voiageR` 2.0.0 and its hosted source,
  Linux, macOS, Windows, and WebAssembly checks pass. A fresh CRAN source
  bundle passes `R CMD check --as-cran` with 0 errors, 0 warnings, and the
  expected new-submission and clock-verification notes. The author supplied a
  valid confirmation address; record upload, email confirmation, reviewer
  feedback, and acceptance as separate evidence states.

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
  The round-nine source passes its local contract at 1,583 words with 18/18
  citations reconciled. PR #522 passed the hosted Open Journals build,
  Textstat report, and six-page visual review. The retained decision-maker
  wording passed its release-bound rebuild and six-page visual review in PR
  #529; human source review remains pending.
- [x] Return the revision to the full panel, remediate every supported
  manuscript finding, and record snapshot scores as diagnostic evidence rather
  than artificial acceptance thresholds; report external JOSS screening gates
  separately.

## Phase 2C: Optional community software-review routes

- [x] Add the authoritative cross-venue contract, fail-closed validator,
  focused tests, tox/task entry points, hosted CI gate, governance baseline,
  pull-request reminder, and developer documentation. ([Issue
  #614](https://github.com/edithatogo/voiage/issues/614))
- [x] Create native subissues under #296 and add them to Project 28 for the
  pyOpenSci, rOpenSci, and distinct post-JOSS decision lanes. ([Issues
  #615](https://github.com/edithatogo/voiage/issues/615),
  [#616](https://github.com/edithatogo/voiage/issues/616), and
  [#617](https://github.com/edithatogo/voiage/issues/617))
- [ ] After the direct JOSS outcome, refresh the official pyOpenSci criteria
  and decide whether to open a Python-package pre-submission inquiry. Maintain
  the criteria-to-evidence matrix locally first; an inquiry or submission
  requires a separate explicit author decision. ([Issue
  #616](https://github.com/edithatogo/voiage/issues/616))
- [ ] Reconsider rOpenSci only after `voiageR` has a self-contained,
  reviewer-reproducible installation path, a deliberately bounded R API,
  confirmed distribution evidence, and a documented mapping to the applicable
  rOpenSci statistical-software standards. Do not run this review concurrently
  with an associated manuscript review. ([Issue
  #615](https://github.com/edithatogo/voiage/issues/615))
- [ ] At the same decision point, assess whether the R Journal, Journal of
  Statistical Software, or a sustainability affiliation such as NumFOCUS would
  add a distinct community or scholarly outcome rather than duplicate JOSS.
  Treat RRID, Software Heritage, Zenodo, and research-software directories as
  discoverability or archival routes, not peer-review substitutes. ([Issue
  #617](https://github.com/edithatogo/voiage/issues/617))

## Phase 2D: Repository-controlled cross-venue closure plan

The following ordered work closes every currently identifiable
repository-controlled submission requirement. Each item remains open until its
issue records the named evidence. Author, participant, maintainer, editor,
curator, reviewer, and registry decisions are excluded from completion and
remain explicit gates in the target contract.

- [x] [Issue #614](https://github.com/edithatogo/voiage/issues/614) — refresh
  every official criterion at the decision point; keep each target's evidence,
  authority, status, and execution-lane assignment valid; reconcile specialised
  JOSS, arXiv, registry, and binding evidence after each release.
  (`23261ce`; `criteria-refresh-2026-07-27`; full tox validation passed)
- [x] [Issue #616](https://github.com/edithatogo/voiage/issues/616) — complete
  and test a pyOpenSci criteria-to-evidence matrix for the maintained Python
  API, installation, documentation, support, governance, prior art, methods,
  release provenance, and AI-use disclosure. Then leave maintenance commitment
  and any inquiry to the author after the JOSS decision point.
  (`46ba8fe1`; `pyopensci-evidence.json`; full tox validation passed)
- [~] [Issue #615](https://github.com/edithatogo/voiage/issues/615) — make
  `voiageR` self-contained to install and test, define its bounded public API,
  and add a claim-by-claim rOpenSci statistical-software standards matrix with
  reproducibility, non-finite-input, error-condition, and reference-comparison
  evidence. Use the result to determine R Journal eligibility without drafting
  a duplicate manuscript. (`DESCRIPTION` author/maintainer repair; clean
  built-source `R CMD check --no-manual` passed locally; self-contained runtime
  and `pkgcheck` evidence remain outstanding; `ropensci-evidence.json` maps
  the current bounded R API, documentation, test, numerical-reference,
  input/error, and seed evidence.)
- [ ] [Issue #622](https://github.com/edithatogo/voiage/issues/622) — select a
  retained HPC source/native-build strategy, add locally tested Spack and (only
  if distinct) EasyBuild recipes, and document CPU-fallback and numerical-smoke
  evidence. Keep HPSF/E4S curation conditional on adoption and governance.
- [ ] [Issue #617](https://github.com/edithatogo/voiage/issues/617) — record a
  venue-by-venue non-duplication decision for JSS, NumFOCUS, and Zenodo before
  creating any new manuscript, affiliation, or deposition material.
- [ ] [Issues #299](https://github.com/edithatogo/voiage/issues/299),
  [#312](https://github.com/edithatogo/voiage/issues/312), and
  [#471](https://github.com/edithatogo/voiage/issues/471) — maintain the
  repository-validated manuscript and metadata package only. Author
  attestations, human research-use/community evidence, category/licence choice,
  upload, submission, and editorial outcomes are not repository tasks.

## Phase 3: Reconciliation and closeout

- [x] Reconcile Conductor status, release evidence, and external-gate boundaries.
- [x] Run the repository's documented JOSS and package validation workflow.
  (`80af0da`)
- [ ] Archive this track only after all automatable work is complete and every remaining external gate is explicit.

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
- RRID route: SciCrunch General Resource registration; assignment and curation external.
- JOSS route: direct JOSS is selected for the Rust-centred polyglot package.
  The canonical arXiv LaTeX preprint and JOSS adaptation are repository-ready;
  demonstrated research use, human engagement, author-confirmed AI attestation,
  authenticated submission, and editorial review remain human or external.
  The author confirmed funding, competing-interest, affiliation, and ORCID
  metadata on 24 July 2026.
- Signed v2.0 release and matching archive: complete; remaining identifier,
  submission, review, adoption, and indexing gates remain external.
- Cross-venue contract: `specs/submission-readiness/targets.json` covers 22
  current and potential destinations and is enforced locally and in hosted CI.
  It records readiness and evidence boundaries but authorizes no external
  action.
- Execution lanes: every retained target is linked to a repository-controlled
  issue in `targets.json`; #614 is the universal contract lane, #615 the R
  lane, #616 the Python community-review lane, #617 the distinct-outcome lane,
  and #622 the HPC packaging lane. Their completion cannot establish any
  external registry, review, curation, or acceptance state.
- R publication: r-universe publishes `voiageR` 2.0.0 with green hosted
  builds. The CRAN-quality bundle passes strict local checks and has an
  author-controlled maintainer email; upload, confirmation, review, and
  acceptance remain separate external states.
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
  separate demonstrated-use gate and single-author engagement risk.
- arXiv preprint package: canonical authored source is `paper/main.tex`; the
  deterministic, non-submitting readiness pipeline validates TeX Live
  2023/2025, source hygiene, PDF/font integrity, semantic HTML, and independent
  cleaner/collector variants. Prior submission `7861466` is absent from the
  authenticated dashboard. Replacement submission `7870358` is incomplete at
  the start stage and expires on 9 August 2026; it is not submission evidence.
- Direct JOSS submission is authorised by the author but remains unperformed
  until issue #471 contains genuine human engagement evidence, the exact v2
  release is archived, the AI attestation is confirmed, and the
  author-preferred arXiv announcement/permanent-identifier boundary is
  resolved. The release-bound PDF has been built and reviewed.
- JOSS permits an arXiv preprint before, during, or after JOSS submission;
  arXiv timing is therefore not a JOSS blocker.
