# Implementation Plan: Remaining Backlog Delivery

Current disposition: [24 November reconciliation](../../reconciliation-20261124.md).
Spack passed on Colab. The [HPC roadmap](../../../roadmap.md#deferred-hpc-qualification)
owns the remaining native matrix. Repository closeout proceeds with B12–B14.

## Phase 1 — Audit and recovery

- [x] B1: Reconcile every open PR and issue with current hosted evidence (AC1).
- [x] B2: Verify independent recovery and prune retired branch tips with leases
  while preserving active work (AC5).
- [x] B3: Review inventory and recovery receipts; validate Conductor and
  GitHub cross-references before checkpointing (AC1, AC5).

## Phase 2 — Repository repairs

- [x] B4: Challenge historical-lock, dependency-policy and coverage-upload
  contracts, then repair #656 controls without rewriting old receipts (AC2).
  Evidence: PR #1057 (515529e3), closed #656, and
  [delivery checkpoint](./delivery-checkpoint-20260901.md).
- [x] B5: Test invalidation and plugin incompatibilities, then implement and
  measure opt-in serial test acceleration for #1028 (AC2).
  Later evidence: PR #1059 (79072731), closed #1028, and
  [delivery checkpoint](./delivery-checkpoint-20260901.md).
- [x] B6: Deliver and validate the current repository source and recipes for
  #1025, and retain the passing installed Colab Spack evidence (AC2).
  - [x] B6.1: Qualify an immutable current source archive with provenance,
    runtime/dependency and actual runtime-file integrity witnesses; see the
    2026-10-28 continuation audit and handoff. Accepted in PR #1212 at
    `7f0feacc`; reviewed and merged trees match.
  - [x] B6.2: Bind current recipes to the qualified source and retain the
    delivered Linux graph, source/vendor, compiler and installed Spack receipts.
    The [installed Spack receipt](../hpc_source_build_qualification_20261029/installed-spack-colab-20261124.md)
    records the passing Spack cell. The remaining compiler, capacity,
    EasyBuild and upstream work is tracked on the HPC roadmap.
    Polars frontend/runtime source declarations were delivered in PR #1214.
    Dated Rust source/bootstrap bindings and stable compiler-role separation
    were accepted in PR #1215 (`e336cd9f`), ahead of the later graph and build work.
    Source slices: stable compiler roles `07c556ec`, dated source/bootstrap
    binding `1d9be54d`; upstream licensing retained separately. See the dated
    compiler receipt and review. All 15 local tox environments pass; protected
    PR #1215 delivery is accepted; this is not native build acceptance.
    Later merged checkpoints: PR #1260 resolves all three Linux graphs;
    PR #1261 qualifies the bounded source/vendor closure; PR #1262 builds
    and probes both baseline source Rust/LLVM roles. The 3 November review
    verified 953 retained file bindings, including declared decompressed bytes.
    Compiler binaries were not retained. See the
    [3 November review](./review-reconciliation-20261103.json) for that checkpoint's
    open effective settings and installed acceptance requirements.
    The 22 November Colab run resolved the Spack graph and fetched 132
    archives before the pinned catalogue rejected Python 3.15.2 without
    a digest. The Spack overlay now extends the pinned upstream Python
    recipe with the same checksum-bound CPython archive used by both
    EasyBuild generations. A targeted Spack fetch verified the archive.
    EasyBuild 2024a's retained test output identifies the Perl 5.42.0
    Pod::Simple corpus test failure. The test excluded every fixture under
    the hosted runner's `_temp` directory. The shared Perl recipe pins the
    [upstream filename-filter fix](https://github.com/perl-pod/pod-simple/commit/f6c4e5d7485d0c0c8230b2d2c3d6bfaada015488).
    The original test exited 255 under `_temp`. After the patch, all 19
    fixtures passed with the same results as the ordinary-path control.
    The corrected 2023a job installed Perl and built mimalloc 3.4.1. Its
    default EasyBuild sanity check expected a `bin` directory. Both allocator
    recipes now check the installed header, shared and static libraries,
    and CMake package metadata directly.
    Both generations then reached CPython's profile-guided build, where
    `test_float_pow` raised `OverflowError` for a result that should underflow
    to zero. Both Python recipes use EasyBuild's `precise` setting to preserve
    libm errno. On GCC 13.3.0, the default flags reproduced the failure and
    the precise flags completed the 43-file PGO workload and passed
    `test_float` on the final LTO build.
    [Guarded comparison](https://github.com/edithatogo/voiage/actions/runs/45015070842).
    The next 2024a run completed the profile-guided build and reached the
    installed Python sanity check. EasyBuild 5.4.2 imports setuptools there.
    Python's `ensurepip` provides pip. Both recipes now install the already
    pinned setuptools 84.0.0 source as a Python extension with pip offline.
    The Rust build-support bundles consume it through their Python dependency.
    [2024a receipt](https://github.com/edithatogo/voiage/actions/runs/45016568149/job/137190046465).
    An isolated CPython 3.15.2 replay then exposed the missing Flit backend
    for packaging. The bundles install pinned flit-core before packaging
    and the existing setuptools-scm pin before pluggy. All 13 sources
    in the corrected Python and Rust support sequence built offline and
    `pip check` passed.
    The subsequent 2024a run installed Python and found that the older
    Hatchling bundle could not build pathspec without Flit. Both generations
    now use the existing Rust build-support bundle for Python and Arrow
    support. A network-denied Python 3.15.2 replay reproduced the failure
    and installed all 33 pinned sources from the shared and consumer bundles.
    Both consumer prefixes passed `pip check`.
    [2024a failure receipt](https://github.com/edithatogo/voiage/actions/runs/45034305228/job/137244514082).
- [x] B7: Complete remaining R, governance, identifier and venue preparation;
  retain external acceptance requirements (AC2, AC4).
- [x] B8: Review repository changes, run full tox and relevant language-native checks,
  and record failures and subsequent verification without weakening gates (AC2).
  Repository and source-binding verification is recorded in the
  [22 November checkpoint](engineering-closeout-20261122.md). The exact tree
  merged in #1386 passed all 15 local tox environments and required hosted
  checks. The Spack evidence in #1403 also passed all
  [15 tox environments](https://github.com/edithatogo/voiage/actions/runs/45270206429)
  and all 17 required checks. Each later PR retains its normal verification gates.
  The [3 November full-gate attempts](./evidence/local-gate-20261103/manifest.json)
  retain the interpreter-selection failure and subsequent free-space stop.

## Phase 3 — Protected delivery and issue reconciliation

- [x] B9: Merge repaired PRs sequentially after exact-head checks, signatures
  and resolved-thread verification; record merge evidence (AC3).
- [x] B10: Refresh all issue bodies and evidence links, closing only satisfied
  acceptance criteria and retaining genuine human/external gates (AC4).
- [x] B11: Review post-merge hosted evidence and validate all track records
  before checkpointing delivery (AC3, AC4).

## Phase 4 — Final cleanup and audit

- [~] B12: Preserve and remove integrated local branches and clean worktrees;
  retain unique work, active branches and maintained release lines (AC5).
  The independently restored 14 November bundle covered two tree-identical
  retired branch tips. The 22 November restore verified four more integrated
  branch tips before retirement. The three Spack repair tips for #1400–#1402
  were independently restored and retired. Settle the disposition of remaining
  refs and worktrees. Retain active or owner-held work with its ownership recorded.
- [~] B13: Re-query final inventories and document all unfinished requirements
  before assessing the complete user goal (AC6).
  The [22 November inventory](b13-inventory-20261122.json) binds open PRs,
  issues and refs after scoped branch retirement.
  The [24 November reconciliation](../../reconciliation-20261124.md) records
  the latest status review. Capture the terminal inventory after the remaining
  repository delivery and eligible cleanup.
- [~] B14: Review final receipts and run Conductor/cross-reference validation;
  archive only when this track's acceptance criteria are satisfied (AC6).
  Registry and cross-reference validation passed for the current checkpoint.
  The final receipt awaits the repository inventory and B12 dispositions.
  The deferred HPC track retains its own native acceptance and archival gate.

## Review fixes — current source qualification

- [x] Require declared build inputs and validate every distribution dependency,
  including marked requirements and extras; preserve failing regression evidence
  and rerun the affected source and package checks before protected delivery.
  Fix: `dffb1114`; 24 qualifier tests and 33 affected integration tests passed.
  Final-head hosted checks and protected delivery remain required for B6.1.

## Review fixes — dated compiler providers

- [x] Replace wildcard imports, correct the class-method binding and preserve
  the failed external OpenSSL probe fallback; regression rejected the original
  import style and all 62 affected tests now pass. Private Spack class loading
  passes on Python 3.15.1. Current-head hosted checks and native build gates
  remain required; this does not complete B6.2.

## Arrow prerequisite source slices

- [x] B6.2-LibCST: Freeze and challenge the Python 3.15 LibCST source, marker,
  licence and compiler contract; implement the bounded readiness packet, then
  qualify the actual provider without inferring the full Arrow/native matrix.
  Source/declaration slice: `227f1583`; 32 affected tests, authenticated source
  and actual private Spack class load passed. PyArrow built with the installed
  LibCST 1.9.0 provider in the passing Spack cell.

## Build capacity follow-up

The [capacity plan](./capacity-plan.md) retains the B6-CAP1–B6-CAP4 support
decision and pilot evidence. The [HPC roadmap](../../../roadmap.md#deferred-hpc-qualification)
owns the remaining terminal capacity packet, source staging and measured build
budgets. B12 retains preservation-backed repository cleanup.

## B8 verification checkpoint — 3 November 2026

The interpreter-selection fix uses `[testenv] runner = virtualenv` and the
explicit `VOIAGE_TOX_PYTHON` repo-local executable. All 15 environments verified
Python 3.15.1. The full run at `a3972c810` passed 13 environments and exposed
formatting, metadata encoding and a missing selected-cache-directory failure.
At corrected candidate `56f603e0d`, all 33 affected tests passed, followed by
successful lint and the complete coverage lane: 5,803 passed, 27 skipped,
94.83% coverage. The other 13 unchanged environments retain their passing proof.
The first failing full run remains retained; it is not retrospectively a pass.

[Verification manifest](./evidence/tox315-20261103/manifest.json.gz) and terminal
resource receipts bind both sources, configurations, commands and raw artifact
hashes. Verification used a standard Git worktree on the external SSD; the
8 GiB reserve was preserved, with over 15 GiB minimum free on the final rerun.
At that checkpoint the installed matrix, hosted delivery and final audit were
open. The current status is recorded in the B8 task above.

## Native continuation — 5 November 2026

PRs #1263–#1267 merged after current required checks and review resolution.
The last coverage lane passed 5,855 tests with 20 skips and 94.84% coverage.
Actual installed Spack and both EasyBuild attempts reached their four-hour
stage limits with complete cleanup; native B6.2 acceptance remains open.
The linked HPC track owns the isolated-loopback regression, platform admission
and bounded continuation. Colab initially rejected runner identity mappings; a user-authorized disposable
runtime prerequisite now permits exact mappings. Colab and WSL admission are
being qualified separately. Source/control success does not substitute for
installed proof.
