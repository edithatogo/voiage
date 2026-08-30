# Implementation Plan: v2.2 Release and Venue Submissions

## Phase 0 — Track and release contract [checkpoint: 710aca9]

- [x] **R0 / AC-01:** Initialize the authorized track, repair the active-track
  handshake, validate Conductor, and record the approval and baseline. (`de89950`)
- [x] **R1 / AC-01, AC-02:** Reconcile live release, tag, registry, venue, issue,
  and workflow state against the hardened candidate. (`be33223`)
- [x] **R2 / AC-01:** Run automated phase review and validation checkpoint.
  (`710aca9`)

## Phase 1 — v2.2.0 release candidate

- [x] **R3 / AC-02:** Add failing synchronization/readiness assertions for the
  v2.2.0 candidate and stale v2.1.0/v1.0.0 submission bindings. (`a9a8534`)
- [x] **R4 / AC-02:** Synchronize Rust, Python, R, Julia, lockfiles, release notes,
  manuscript metadata, and venue packets to v2.2.0. (`9004ba4`)
- [x] **R5 / AC-02:** Run ABI, packaging, manuscript, submission, binding, and full
  tox validation; repair all in-scope findings. (`5cdb1dd`)
- [~] **R6 / AC-02:** Open the protected release-candidate PR, resolve review
  findings, wait for exact-head checks, and squash-merge with exact tree equality.
- [x] **R6a / AC-02 — Review fixes:** Bind the verified upstream template
  revision, preserve the public-release boundary, and reject fabricated or
  deleted prepublication evidence fields. (`00edb35`)
- [ ] **R7 / AC-02:** Run automated phase review and validation checkpoint.

## Phase 2 — Signed staged and public release

- [ ] **R8 / AC-03:** Create and push the signed annotated v2.2.0 tag on the exact
  merged release candidate.
- [ ] **R9 / AC-03:** Wait for the private draft, download and verify the attested
  payload, and record the reviewed wheel and sdist SHA-256 values.
- [ ] **R10 / AC-03:** Invoke the hash-bound publish workflow and verify GitHub,
  TestPyPI, PyPI, provenance, SBOM, and clean-install receipts.
- [ ] **R11 / AC-03:** Reconcile any Rust, R, Julia, documentation, archive, and
  registry workflows triggered by the tag without overstating external outcomes.
- [ ] **R12 / AC-03:** Run automated phase review and validation checkpoint.

## Phase 3 — pyOpenSci-first submission

- [ ] **R13 / AC-04:** Refresh the current official template and bind the final
  submission body to the public v2.2.0 release.
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
