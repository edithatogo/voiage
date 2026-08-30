# Release Candidate Review

Functional candidate: `00edb35bdfd05c3c011a546a6a49460567481d05`.
Scope: Phase 1 after checkpoint `710aca9`; review is not a phase-completion
receipt. R6 remains in progress until exact-head hosted checks and merge.

## Findings and repairs

- Corrected upstream pyOpenSci template provenance to the observed commit
  `df24b7c63a589ff5d82a30e42f1d11b8aa1b5927` and its exact blob. Added a
  baseline-binding test supporting both active and archived track locations.
- Preserved v2.1.0 as the latest public README release. The v2.2.0 source
  candidate has no tag, artifact, publication, submission, or acceptance claim.
- Required nullable release-identity fields to remain present and null before
  publication. Negative tests reject rebound publication dates, fabricated
  artifact digests, and deleted receipt fields.
- Made the JOSS drift test mutate structured metadata, not a release literal.
- Split release authorization from actual publication in track gates and
  restored the active release task in the root backlog and roadmap.

## Local assurance

The full 15-environment tox matrix passed at `5cdb1dd9` before the subsequent
self-review fixes. The fresh `tox -p 3 --parallel-live` run at the functional
candidate passed 14 environments but failed the coverage environment: 4,510
tests passed, 16 skipped, and the import-speed assertion measured 6.84 seconds
against its five-second budget. Combined coverage was 95.16 percent. The
unchanged import test passed in isolation. A bounded full retry serializes tox
environments with `-p 1`, retaining six pytest workers and all existing timing
and coverage limits. The complete retry passed all 15 environments: 4,511
tests passed, 16 skipped, 95.16 percent combined coverage, and 1,191.55 seconds
total elapsed time. No implementation, timing limit, or coverage threshold was
changed between those runs. Subsequent edits only record this evidence.

- Rust workspace and fuzz checks, standalone R release-profile compilation,
  version synchronization, and the pyOpenSci staging validator passed.
- The v2.1.0 ABI comparison passed: nine symbols retained and the additive
  `voiage_v1_enbs_r` symbol does not break the baseline.
- Pinned pkgcheck passed for the v2.2.0 R source package: zero R CMD check
  errors or warnings, 89.5 percent R coverage, and applicable statistical
  standards accepted. Advisory naming, linter and example notes are retained;
  this is not rOpenSci acceptance. The temporary source archive had SHA-256
  `50438350e09cfc02186019b0e415474dfcd65fd1c855240f7ad6c0ecb2fc6a72`;
  the runner removed its temporary directory after checking.
- Canonical LaTeX build and PDF audit passed: 15 pages, 26 embedded fonts.
  The reviewed source archive SHA-256 was
  `818fd081e09f5dd76947fc603317d791a09067baa1c40084a244934567b82016`.
  Textstat 0.7.13 recorded 4,160 words and 281 sentences without warnings;
  scores remain editorial evidence, not scientific-quality thresholds.
- Local LaTeXML exited zero but reported conversion status 2 because its Perl
  image-processing module was unavailable. The HTML structure check passed,
  but this does not establish figure conversion. Hosted semantic-HTML
  assurance remains required before merge.

## Interrupted validation and recovery

An earlier final-head tox attempt lost its worker processes and left a wrapper
with a revoked filesystem descriptor. A volume interruption is suspected, not
confirmed. That attempt was terminated and is not counted as a pass.

Git connectivity passed with commit-graph acceleration disabled. A stale
derived split commit-graph referred to missing old commits; after a verified
recovery bundle was created, the exact cache directory was moved to a
recoverable temporary location and regenerated. Default Git connectivity then
passed. No tracked source loss was observed. Full validation was restarted
with live output captured locally.

## Review boundaries

The current [pyOpenSci policies](https://www.pyopensci.org/software-peer-review/our-process/policies.html#submission-volume-and-maintainer-overlap)
allow one submission under review per point of contact. On 2026-08-30,
[@edithatogo's ee_trd issue #272](https://github.com/pyOpenSci/software-submission/issues/272)
was open with `on-hold`, and
[pyMARS issue #271](https://github.com/pyOpenSci/software-submission/issues/271)
was open with presubmission labels. These labels alone do not establish
whether a new voiage submission is allowed. R14 now requires clarification;
no editor contact, withdrawal, issue closure, or replacement contact was
authorized or performed. The venue policy also prefers human-written review
communications and prohibits concurrent review without an applicable editorial
exception. Submission preparation must preserve those boundaries.

Live pre-PR security readback on 2026-08-30 found no open dependency alerts
and no Critical or High security-severity code-scanning alerts. There were
96 CodeQL code-quality alerts without a security-severity classification and
two Scorecard alerts. This is not a claim that all advisory findings are fixed.
Scorecard #854 is Medium: its last update on 2026-08-01 reported SAST on
26 of 30 sampled commits, not an exploitable-code finding. Its bounded release
disposition requires successful current-head CodeQL and the normal protected
checks; it is not dismissed or bypassed. Scorecard #851 is Low and concerns
the best-practices badge. Hosted assurance remains pending.

Python style: Pass under the repository-authoritative Ruff, typing and
NumPy-docstring rules; these take precedence over the generic guide's pylint
and Google-docstring suggestions. Mobile UI and platform-specific deployment
guides: not applicable to this metadata and release-contract change.

No new analytical implementation, dependency promotion, arXiv upload, personal
declaration, or external venue outcome is included. The post-publication
packet refresh and eligible JOSS handoff remain later tasks.
