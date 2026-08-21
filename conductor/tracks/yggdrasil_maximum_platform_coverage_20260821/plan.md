# Yggdrasil Maximum Platform Coverage Implementation Plan

Every task cites the acceptance criteria it advances. Tests precede
implementation, and each phase ends with automated review and validation.

## Phase 1 — Freeze the platform-coverage contract [checkpoint: a84eef8a]

- [x] **YMC1-T1 (AC1, AC3): Write failing contract tests.** (`ac929f1c`)
  - [x] Assert that the schema requires catalogue and candidate revisions,
    exact platform identities, lifecycle states, evidence boundaries, and
    aggregate reconciliation.
  - [x] Add adversarial cases for missing platforms, duplicates, unexplained or
    over-broad exclusions, missing reconsideration triggers, stale totals, and
    runtime claims based only on builds.
  - [x] Run the focused suite and record the expected red result before adding
    implementation artifacts.
- [x] **YMC1-T2 (AC1, AC3): Implement the versioned contract.** (`bb517571`)
  - [x] Add the JSON Schema, initial manifest, and fail-closed validator under
    the existing binding/release contract boundary.
  - [x] Model build, product, ABI-smoke, and runtime validation independently.
  - [x] Pin the observed Yggdrasil catalogue, release source, recipe, and
    candidate identities.
- [x] **YMC1-R1 (AC1, AC3): Apply Phase 1 automated review fixes.** (`5792e1fd`)
  - [x] Correct the release and BinaryBuilderBase source revisions to the exact
    authoritative commits resolved by the candidate and Yggdrasil manifest.
  - [x] Reject broad additional predicates and non-authoritative placeholder
    evidence before an evidence-filtered stage can validate.
- [x] **YMC1-T3 (AC1, AC6): Complete Phase 1 review and validation.** (`546a2a2b`)
  - [x] Run focused positive, negative, property, and pathological tests.
  - [x] Run Ruff and type checking for new Python validation code.
  - [x] Run automated Conductor review; apply Critical/High fixes and rerun.
  - [x] Run full Conductor and cross-reference validation and record the phase
    checkpoint.

## Phase 2 — Expand the recipe by negative filtering [checkpoint: 6f083205]

- [x] **YMC2-T1 (AC2): Write recipe-policy tests first.** (`569e9f61`)
  - [x] Require `platforms = supported_platforms()`.
  - [x] Require exactly the initial FreeBSD `aarch64` and `riscv64` filters,
    including adjacent reasons, before the first expanded run.
  - [x] Reject the former seven-platform positive allowlist and uncontracted
    negative predicates.
- [x] **YMC2-T2 (AC2, AC5): Update the repository recipe mirror.** (`4e838afe`)
  - [x] Apply the inclusive universe and two initial filters.
  - [x] Preserve locked Cargo build, musl shared-link flags, product naming,
    installation paths, and release-source pinning.
  - [x] Parse/lint the recipe locally and update the initial manifest snapshot.
- [x] **YMC2-T3 (AC2, AC6, AC7): Refresh the external PR candidate.** (`a32188d9`)
  - [x] Re-query PR #14292 and its base immediately before mutation.
  - [x] Rebase or refresh the owner-controlled branch without overwriting
    unrelated upstream work.
  - [x] Apply the repository-validated recipe change, commit, push, and respond
    to the existing review thread with the exact candidate head.
- [x] **YMC2-T4 (AC2, AC6): Complete Phase 2 review and validation.** (`66b51eb6`)
  - [x] Run recipe-policy tests, contract validation, and diff hygiene.
  - [x] Run automated Conductor review; apply Critical/High fixes and rerun.
  - [x] Run full Conductor and cross-reference validation and record the phase
    checkpoint.

## Phase 3 — Undertake maximum-coverage hosted filtering [checkpoint: dd0c572d]

- [x] **YMC3-T1 (AC3, AC4, AC6): Capture the expanded hosted matrix.** (`c3413a56`)
  - [x] Wait for the exact-head Buildkite matrix to reach terminal states.
  - [x] Record every catalogue member, including passed, failed, skipped, or
    not-scheduled states, without collapsing build and runtime evidence.
  - [x] Preserve the initial expanded-run receipt even if remediation is needed.
- [x] **YMC3-T2 (AC3, AC4): Triage every non-pass result.** (`d81c0f69`)
  - [x] Classify recipe/source/linker/product/ABI failures as actionable and fix
    them while retaining the platform.
  - [x] Classify infrastructure failures as transient and rerun them.
  - [x] For a genuine upstream toolchain or architecture gap, collect primary
    evidence, define the narrowest predicate, and state the retest trigger.
  - [x] Attempt Windows `i686`, Linux `i686`, ARM, PowerPC, FreeBSD `x86_64`,
    musl, and other catalogue members unless their own evidence fails the
    contract; do not copy exclusions from unrelated recipes.
- [x] **YMC3-T3 (AC3, AC4): Iterate to a terminal maximum-coverage candidate.** (`d7513968`)
  - [x] Add only contract-valid negative filters and corresponding tests.
  - [x] Rerun the full matrix after each recipe revision.
  - [x] Reconcile final included/excluded counts and preserve superseded runs.
- [x] **YMC3-T4 (AC3, AC4, AC6): Complete Phase 3 review and validation.** (`685ada0f`)
  - [x] Run the schema, manifest, recipe-policy, evidence-integrity, and
    pathological suites.
  - [x] Run automated Conductor review; apply Critical/High fixes and rerun.
  - [x] Run full Conductor and cross-reference validation and record the exact
    final Yggdrasil candidate head and hosted run.

## Phase 4 — Product, ABI, and downstream evidence [checkpoint: da878c15]

- [x] **YMC4-T1 (AC5): Verify per-platform product evidence.** (`8aef2e8d`)
  - [x] Confirm each green target produced the declared shared-library product
    with the correct platform filename and installation path.
  - [x] Mark cross targets as build/product validated only unless executable
    evidence exists.
- [x] **YMC4-T2 (AC5): Run executable smoke evidence where available.** (`8aef2e8d`)
  - [x] Verify exported C ABI and native version agreement.
  - [x] Execute the deterministic EVPI shared fixture on runnable generated
    artifacts or record why a target is not executable in the available lane.
  - [x] Keep Julia clean-depot JLL execution pending until a registered JLL is
    authoritative and available.
- [x] **YMC4-T3 (AC5, AC6): Complete Phase 4 review and validation.** (`c65127a7`)
  - [x] Run product/ABI tests and verify that evidence labels cannot overclaim
    runtime coverage.
  - [x] Run automated Conductor review; apply Critical/High fixes and rerun.
  - [x] Run full Conductor and cross-reference validation and record the phase
    checkpoint.

## Phase 5 — Reconcile, validate, and hand off external gates

- [x] **YMC5-T1 (AC6, AC7): Reconcile repository records.** (`32d98e36`)
  - [x] Update issue #555 and issue #614 only with exact repository and hosted
    evidence; do not infer external merge or registry state.
  - [x] Refresh the archived registry-readiness handoff and release-candidate
    receipt through an append-only successor artifact.
  - [x] Cross-reference the final PR head, Buildkite run, Conductor track, and
    any VOIAGE implementation PR.
- [ ] **YMC5-T2 (AC6): Run the repository-owned final gate.**
  - [ ] Run focused tests, Ruff, type checks, relevant tox environments,
    repository harness, full Conductor validation, cross-reference validation,
    and `git diff --check`.
  - [ ] Run automated final Conductor review; apply Critical/High fixes and
    repeat the full gate.
- [ ] **YMC5-T3 (AC7): Record external handoff without premature closure.**
  - [ ] Re-query PR #14292 immediately before recording its state.
  - [ ] Keep Yggdrasil merge, JLL generation, clean-depot JLL smoke, Julia
    General merge, and indexing pending unless authoritative receipts exist.
  - [ ] Archive the track only when AC1–AC7 have repository evidence and every
    remaining external gate is explicit.
