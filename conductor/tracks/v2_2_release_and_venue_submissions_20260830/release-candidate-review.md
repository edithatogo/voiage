# Release Candidate Review

Initial functional candidate: `00edb35bdfd05c3c011a546a6a49460567481d05`.
Scope: Phase 1 after checkpoint `710aca9`. The final readback below supersedes
preparation-time pending-check notes, which are retained as historical context.

## Final source-phase readback

PR #1038 merged on 2026-08-30 at `cf35bc998b936ede8eb3eea57d5bb2eec40124c0`.
Its verified merge tree, `c525533c75735f1efabd5f8d40e4d01d1817710a`, exactly
matches checked head `dbb089fe5bfbc57dd897d2cca7d0b1bec4268d5e`. All 68 hosted
checks passed, five were skipped under their configured conditions, and all
three review threads were resolved. Fresh CodeQL marked alert #1280 fixed.
The complete [merge receipt](release-candidate-merge-receipt-20260830.json)
retains individual job links and the distinct local-validation evidence.

The review covered all phase changes since checkpoint `710aca9`, including
Conductor records. Decision-card behavior, installed-consumer isolation,
submission guards and version synchronization have direct regression coverage.
The R runner, canonical LaTeX, package/binding metadata and source-history
contracts have their dedicated tests and hosted build evidence. No new
Critical or High finding remains; retained advisory dispositions and external
venue gates below are not presented as completed outcomes. The final checkpoint
revalidates the evidence-only delta; it does not substitute for the full local
matrix or exact-head hosted checks already completed for the source candidate.

## Hosted review follow-up

PR #1038 head `addcbc826d21366adcc36d001f3133e72274c6d3` completed 66
successful checks and five governed skips, with two failures and two unresolved
review threads. It was not merged. R6b owns the following repairs:

- P1: New decision cards derive producer identity from installed package
  metadata; absent distribution metadata is `unknown`. Deserialization
  preserves explicit historical versions and uses `unknown` for omitted
  historical provenance, never the reader's current version.
- P2: CITATION.cff and CodeMeta retain verified v2.1.0 release metadata rather
  than announcing an unpublished v2.2.0 release. R13 updates them only after
  public-artifact verification. The source and venue candidate remain v2.2.0.
- Both failed CI jobs reached the same historical-evidence test. The original
  pre-squash commit was not reachable in hosted Git history, so its reader
  substituted current files and compared v2.2.0 manifests with v2.1.0 hashes.
  The repaired test reads the recorded squash merge, verifies its actual tree,
  retains every historical digest, and fails if Git evidence is unavailable.
- Focused type checking exposed a nested governance-dictionary inference
  error. An explicit typed dictionary preserves the serialized data and hash.

The review regression red phase reproduced five lineage/citation failures and
two historical-reader failures. All 37 focused regression tests passed. The
fresh complete `uv run tox -p 1 --parallel-live` run then passed all 15
environments: 4,516 tests passed, 16 skipped, 95.16 percent combined coverage,
and 1,564.10 seconds elapsed. Its retained log SHA-256 is
`7f567625f56a142b4d14ade6b10de5a6b43fbc6284c7e3143f7af1ceb32d7f19`.
No timing limit or coverage threshold was weakened. Fresh hosted checks for
these repairs remain pending; earlier hosted results do not cover changed code.

The required dependency-frontier probe passed. Its incidental Hypothesis,
linkify-it-py and websocket-client updates were not retained in this frozen
release candidate; the unchanged lockfile also passed the strict audit. No
generated Conductor context pack was present, so the canonical index-linked
project context and this active track were used.

Stale ignored `voiage.egg-info` from July reported `0.2.2.dev102` and shadowed
the development distribution. It was moved recoverably to a temporary backup,
and `maturin develop --release --locked` installed verified 2.2.0 metadata.

Hosted LaTeXML reported `Status:conversion:0`, including successful graphics
processing, on the initial PR head. Both TeX Live versions and source/PDF
assurance also passed. This supplies the hosted evidence missing locally;
fresh-head checks remain required after the review fixes.

All hosted checks on `680f86b73b2726caa5df273631f613f4fd3442d6` completed:
68 passed, five skipped, and none failed. Both previously failing coverage
jobs passed. CodeQL nevertheless reported a mixed
module-import style in the added lineage tests (alert #1280). R6c removes the
duplicate module import and uses direct monkeypatch target paths. The runtime
implementation and every assertion remain unchanged. This warning is being
repaired, not dismissed; the final local result is recorded below and fresh
hosted validation remains pending.

The R6c full retry retained a maximum-version Hypothesis timing failure:
`test_any_unique_nullable_suffix_is_compatible` measured 292.99 ms against
its unchanged 200 ms deadline, then 0.07 ms on replay. Its coverage run later
waited on a cold dependency install observed still running at 11 minutes
43 seconds, with further download progress afterward. That exact owned
install subprocess and the superseded tox wrapper were stopped; unrelated
work was untouched. The attempt exited 1 after 3,113.47 seconds, with 13
environments passed, one failed, and coverage interrupted rather than passed.
Its log SHA-256 is
`3aa63bab9acff5270480f84da9774eef62017eda76352c6736b58a56dfedd1c0`.

R6d adds an explicit writable download-cache option while retaining default
cache isolation, fresh consumer environments, dependency resolution, real
wheel installation, copied files, and an outside-checkout SDK probe. Every
build/install/probe now has a 600-second timeout. Five new red cases preceded
the repair; six helper cases and the real consumer test then passed. The
complete consumer call took 160.08 seconds with cache reuse. This is a local
observation against an interrupted cold-install attempt, not a controlled
hosted speedup claim. No cache files were manually changed. The original
Hypothesis and import-performance deadlines remain unchanged. The complete
matrix retry with the explicit cache option passed all 15 environments in
1,775.65 seconds; the previously failing maximum-version environment passed
unchanged. The coverage suite passed 4,522 tests with 16 skips and 95.16 percent
combined coverage. The complete log SHA-256 is
`a719c5d43503fc6423814f1aca654704cb236fc8e48c89fb9978c0e13147a1b6`;
the focused cache-test log SHA-256 is
`2f69bbda2262901ce707857883cf6ca3bc2f2e77ab94c984747dc0afa8274215`.
Fresh exact-head hosted validation remains required before merge.

Phase review also found and corrected the candidate rationale's stale claim
that citation metadata already selected v2.2.0. Its staging digest and the
root backlog now agree that citation metadata retains public v2.1.0 until
publication is verified.

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
changed between those runs. This pass predates the R6b code fixes above, which
require their own complete validation.

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
