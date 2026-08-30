# Implementation Plan: v2.2 Release and Venue Submissions

## Phase 0 — Track and release contract [checkpoint: 710aca9]

- [x] **R0 / AC-01:** Initialize the authorized track, repair the active-track
  handshake, validate Conductor, and record the approval and baseline. (`de89950`)
- [x] **R1 / AC-01, AC-02:** Reconcile live release, tag, registry, venue, issue,
  and workflow state against the hardened candidate. (`be33223`)
- [x] **R2 / AC-01:** Run automated phase review and validation checkpoint.
  (`710aca9`)

## Phase 1 — v2.2.0 release candidate [checkpoint: cf35bc9]

- [x] **R3 / AC-02:** Add failing synchronization/readiness assertions for the
  v2.2.0 candidate and stale v2.1.0/v1.0.0 submission bindings. (`a9a8534`)
- [x] **R4 / AC-02:** Synchronize Rust, Python, R, Julia, lockfiles, release notes,
  manuscript metadata, and venue packets to v2.2.0. (`9004ba4`)
- [x] **R5 / AC-02:** Run ABI, packaging, manuscript, submission, binding, and full
  tox validation; repair all in-scope findings. (`5cdb1dd`)
- [x] **R6 / AC-02:** Open the protected release-candidate PR, resolve review
  findings, wait for exact-head checks, and squash-merge with exact tree equality.
  PR #1038 merged after 68 passing checks, five governed skips, and zero
  unresolved threads; the verified merge exactly matches the checked tree.
  (`cf35bc9`)
- [x] **R6a / AC-02 — Review fixes:** Bind the verified upstream template
  revision, preserve the public-release boundary, and reject fabricated or
  deleted prepublication evidence fields. (`00edb35`)
- [x] **R6b / AC-02 — Hosted review fixes:** Derive new decision-card producer
  versions from installed metadata, avoid inventing missing historical lineage,
  retain last-published citation metadata until publication is evidenced, and
  read frozen hardening evidence from reachable squash-merge history in CI.
  All 15 tox environments passed with 4,516 tests and 95.16% coverage. (`d351914`)
- [x] **R6c / AC-02 — CodeQL test cleanup:** Use direct monkeypatch target
  paths instead of importing the decision-card module twice; retain the same
  behavior assertions, correct the candidate's stale citation-description
  sentence, and pass full local validation. Fresh hosted CodeQL remains part
  of the R6 merge gate. (`90b3cc2`)
- [x] **R6d / AC-02 — Bound installed-consumer validation:** Retain the
  observed cold-download delay, add a named opt-in writable dependency cache
  without reusing the consumer environment or skipping wheel installation,
  enforce subprocess timeouts, and verify default isolation and full tests.
  All 15 tox environments passed: 4,522 tests, 16 skips, 95.16% coverage.
  (`bfc1d79`)
- [x] **R7 / AC-02:** Run automated phase review and validation checkpoint.
  Use the reachable source squash merge (`cf35bc9`) as the next phase's diff
  boundary. The separate validation receipt (`ca7c5d7`) records the reviewed
  evidence-only checkpoint; it is not a main-history ancestor after squash.
- [x] **R7a / AC-02 — Security-update integration:** Include the already merged
  pnpm security update (#1039) in PR #1046, validate the combined tree, and
  retain its final squash merge as the release-tag target. Runtime code is
  unchanged from the checked #1038 source candidate.
  PR #1046 merged with exact tree equality after 35 passing checks and four
  governed skips; documentation passed locally with pnpm 10.34.4. (`7af563c8`)

## Phase 2 — Signed staged and public release [checkpoint: 4c47b241]

- [x] **R8 / AC-03:** Create and push the signed annotated v2.2.0 tag on the exact
  merged release candidate, including R7a's security update and evidence fixes.
  Verified tag object `6f42d26d5a20d4c1e47221f01daff219edc88a59` targets
  `7af563c8cb373057d30662650b3f332f39e05b83`.
- [x] **R9 / AC-03:** Wait for the private draft, download and verify the attested
  payload, and record the reviewed wheel and sdist SHA-256 values.
  All eight exact-source provenance/SBOM verifications passed; the four
  distributions match the private release, manifest and checksum file.
- [x] **R10 / AC-03:** Invoke the hash-bound publish workflow and verify GitHub,
  TestPyPI, PyPI, provenance, SBOM, and clean-install receipts.
  Immutable public release, exact registry digests, four PyPI attestations and
  fresh macOS installation verified after successful run 33303294302. (`fe79e1a`)
- [x] **R11 / AC-03:** Reconcile any Rust, R, Julia, documentation, archive, and
  registry workflows triggered by the tag without overstating external outcomes.
  Exact-source documentation and tag-bound supply-chain jobs passed; no extra
  binding tags or registry submissions were triggered. The new Software
  Heritage snapshot remains pending. (`70198b2`)
- [x] **R12 / AC-03:** Run automated phase review and validation checkpoint.
  PR #1047 merged after 36 passing checks, three configured skips, one neutral
  CodeQL result and zero unresolved review threads. Verified signed merge
  `4c47b241` exactly matches checked tree `287d239a`; use this reachable merge
  as the next phase boundary. The tagged release source remains `7af563c8`.
- [x] **R12a / AC-03 — Publication projection review repair:** Update current
  citation, roadmap and candidate projections in this publication PR, bind
  the successful immutable receipt, retain prepublication rejection tests,
  and reject mismatched or missing public evidence. PR #1047 review identified
  the contradiction in deferring these projections to the next phase.
  Final full tox passed all 15 environments: 4,530 tests, 16 skips and 95.16
  percent coverage. Non-JSON receipt bypass is covered by a regression.
  (`0d1c2812`)

## Phase 3 — pyOpenSci-first submission

- [x] **R13 / AC-04:** Refresh the current official template and finalize the
  submission body against R12a's public v2.2.0 citation and candidate bindings;
  complete known JOSS paper/use-record and rOpenSci packet repairs before the
  first venue submission, preserving any missing human evidence as a gate.
  Repository packet repairs are complete in PR #1051; R14 retains all personal
  declarations and external eligibility. Exact-head CI remains the PR merge gate.
- [x] **R13a / AC-04 — Research-use environment boundary:** Refresh the
  same-author VOP research-use evidence using separate supported environments
  and a hash-bound data hand-off. VOP requires pandas 3 and SciPy 1.18, outside
  voiage's stable bounds; do not force a combined installation or describe
  the bounded exchange as full in-process integration or independent adoption.
  The published-wheel replay reproduced all 500 rows and EVPI; 30 focused
  tests, 97 percent script coverage and all 15 tox environments passed.
  Historical human-use evidence is preserved, not recertified. (`8785c6de`)
- [x] **R13b / AC-04 — Version-specific manuscript claims:** Align the JOSS
  language-surface description and claim map with native EVPI/ENBS in both R
  and Julia, and distinguish the historical v2.0.0 Software Heritage snapshot
  from the verified v2.2.0 release without inventing a new archive or DOI.
  Describe R's separate bundled Rust kernel rather than claiming one physical
  Rust implementation is used by all language surfaces.
  Also repair stale v1.0.0 and EVPI-only claims in the canonical LaTeX
  preprint, distinguish Python decision records from scalar R/Julia surfaces,
  and rebuild/review both manuscripts without extending historical human
  attestations to new edits. Correct the dated AI-transparency projections.
  Local source/PDF/variant audits and all 15 tox environments passed at
  `7e762eb7`. All six hosted JOSS pages and the complete HTML figure were
  inspected; the later LaTeX spacing fix is covered by R13d. (`a5b3668e`)
- [x] **R13b1 / AC-04 — HTML review fix:** Reject missing or empty manuscript
  images even when LaTeXML returns zero. The local converter lacks its Perl
  image-processing module; retain that failure and require a complete hosted
  semantic-HTML artifact, including its figure, before merge.
  The complete initial-head artifact passed and its figure was inspected;
  fresh exact-head hosted validation remains required. (`7e762eb7`)
- [x] **R13c / AC-04 — Submission preflight review fix:** Recognize successful
  manuscript status values, reject missing required gates, and enforce the
  selected JOSS partner route and current human declarations. Automated tests
  use hypothetical evidence only, never repository attestations. All nine
  focused regressions and full tox passed. (`7e762eb7`)
- [x] **R13d / AC-04 — Hosted provenance false positive:** Keep secret scanning
  enabled while narrowly allowing the pinned public Authentext gitlink only
  in its exact audit provenance line and file. Verify mutated values and other
  paths are not covered by the exception.
  Also correct the hosted LaTeX intersentence-spacing warning after ENBS;
  preserve both initial failures and rerun the unchanged gates.
  Five scope-mutation tests, full local history scanning, exact LaTeX lint and
  all 15 tox environments passed. (`a5b3668e`)
- [x] **R13e / AC-04 — Hosted research-evidence review:** Bind evaluation to a
  reviewed public wheel digest and compare installed payload bytes before use;
  reject same-version local builds or modifications. Commit the exact CSV so
  a fresh clone can evaluate the retained export receipt. Preserve the initial
  receipt and record a new strengthened evaluation, not a human attestation.
  Verified 192 installed payload files, retained the exact CSV, and passed
  38 focused handoff tests plus full tox: 4,600 passed, 16 skipped and
  95.16 percent coverage. (`a5b3668e`)
- [ ] **R14 / AC-04:** Collect or record the maintainer-only Code of Conduct,
  maintenance, guide, survey, reviewer-contact, and partnership declarations;
  resolve contact-capacity eligibility against existing pyOpenSci issues #271
  and #272 and obtain human review of the submission communication.
- [ ] **R15 / AC-04:** Create the authenticated pyOpenSci submission and retain its
  issue URL, timestamp, exact body hash, and observed initial venue state.
- [ ] **R16 / AC-04:** Run automated phase review and validation checkpoint.

## Phase 4 — JOSS partner fast-track and rOpenSci

- [ ] **R17 / AC-05:** When pyOpenSci eligibility is evidenced, refresh the JOSS
  paper and declarations against v2.2.0 and initiate the partner fast-track.
- [ ] **R18 / AC-06:** After honoring pyOpenSci-first ordering and avoiding
  concurrent venue review, refresh `voiageR` inquiry/submission material and
  initiate rOpenSci review with exact package evidence.
- [ ] **R19 / AC-05, AC-06:** Record venue URLs and observed states; do not mark
  review, acceptance, DOI, or indexing complete without authoritative receipts.
- [ ] **R20 / AC-05, AC-06:** Run automated phase review and validation checkpoint.

## Phase 5 — Outcome reconciliation and closeout

- [ ] **R21 / AC-07:** Reconcile release and venue outcomes, repository records,
  roadmap, todo, changelog, issues, and machine-readable evidence.
- [ ] **R22 / AC-07:** Complete whole-track review and full validation.
- [ ] **R23 / AC-07:** Mark complete and archive only after AC-01 through AC-07
  have authoritative evidence; otherwise retain the exact external waiting state.
