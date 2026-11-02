# Implementation plan

Status: in progress; user-authorized scientific ecosystem delivery includes this
linked track. See the compiler requirement review in [the handoff](handoff.md). Sequential dependencies are P1 -> P2 -> P3 -> P4 ->
P5 -> P6. P2 source controls must pass before P3 fetch/build dispatch.

## P1 — Reconcile ownership and freeze the graph
- [x] Task: Review the packaging requirement for every source-built compiler.
  - [x] Record the supported matrix cell, required compiler capability, why a
        qualified binary toolchain is insufficient, and distinct source stages.
  - [x] Obtain agent correctness/security review before bootstrap dispatch.
  - [x] Record any proposed matrix change as an explicit scope decision; retain
        the promised matrix until that decision is accepted and documented.
- [ ] Task: Bind current integrator/provider/capacity handoffs to this track.
  - [ ] Record exact owned paths, worktrees, active jobs and evidence revisions.
  - [ ] Map each parent HPC requirement to AC1–AC6 without duplicate execution.
  - [ ] Freeze the Linux Spack and two EasyBuild matrix cells, compiler roles,
        features, source digests and tool versions in a machine-readable manifest.
- [x] Task: Resolve supported combined-root compiler constraints.
  - [x] Write and reproduce failing graph-contract regression cases.
  - [x] Implement the smallest supported provider/graph correction through owner.
  - [x] Pass the full unmodified Linux root solve and persist the concrete graph.
- [ ] Task: Phase verification, agent review and checkpoint per workflow.md.

## P2 — Qualify fetching and measurement
- [~] Task: Repair and qualify source resolution and offline execution.
  - [ ] Write failing cases for shadowed paths, incorrect digest, missing input,
        forbidden fetch and unsupported CLI options.
  - [ ] Implement supported pre-fetch restrictions and resolved-source checks.
  - [ ] Run focused tests and exact-version package-manager probes on Linux.
- [~] Task: Bind a safe per-stage observation runner.
  - [ ] Test descendants in distinct process groups, sampler loss, timeout/OOM,
        failed stage, interruption, cleanup and receipt completeness.
  - [ ] Reuse accepted observation tools; add only missing counters and provenance.
  - [x] Verify owned-process termination and fail-closed terminal reports.
        Linux run `41435528252` passes all 51 combined controls, including
        detached/orphaned children, SIGTERM, a fork during cleanup and vendor
        contract/shared-lock boundaries.
  - [x] Freeze runner envelope, concurrency and storage headroom before dispatch.
        Four build jobs, one LLVM link job, 12 GiB sampled RSS, 8 GiB minimum
        free space, four hours per stage and 5.5 hours aggregate driver budget.
- [ ] Task: Phase verification, agent review and checkpoint per workflow.md.

## P3 — Measure Rust and LLVM source bootstraps
- [x] Task: Run bounded hosted feasibility builds for the two baseline roles.
  - [x] Record stage0 exception, explicit source stages, LLVM source build,
        targets, debug settings, vendoring and source authentication.
  - [x] Retain per-stage process/CPU/I/O/disk/time measurements and complete logs.
  - [x] Verify installed compiler identities and actual compile/link operation.
  - [x] If stopped, preserve terminal evidence and select a supported execution
        route within available authority; never classify a stop as a pass.
- [~] Task: Qualify the remaining distinct compiler configurations.
  - [ ] Reuse identical accepted compiler output only with exact provenance.
  - [x] Build the selected stable and dated-nightly baseline configurations.
        Runs `41436471133` and `41436475314` passed all 12 stages at `24ae6cb6`.
  - [ ] Bind these baselines to actual package-manager/generation flags and
        qualify differing generation configurations before installed acceptance.
- [ ] Task: Phase verification, agent review and checkpoint per workflow.md.

## P4 — Diagnose overhead and set a measured budget
- [ ] Task: Attribute the costly stages and decide whether design changes help.
  - [ ] Capture focused CPU/compiler timings; record capability gaps and overhead.
  - [ ] Inspect duplicate LLVM/bootstrap work, targets, debug symbols, Arrow
        static/shared output, nested jobs and cache/source-copy amplification.
  - [ ] Write regression tests before any provider or budget-policy change.
  - [ ] Compare justified changes against frozen baseline with parity and linkage
        checks; retain a no-change rationale when existing work is necessary.
  - [ ] Publish per-cell resource envelope, headroom, retention and cleanup policy
        through the capacity owner; do not silently raise the shared guard.
- [ ] Task: Phase verification, agent review and checkpoint per workflow.md.

## P5 — Execute the full HPC matrix
- [ ] Task: Qualify the current Linux Spack graph.
  - [ ] Complete source build/install using the qualified compiler graph.
  - [ ] Pass installed imports, numerical fixtures, Arrow/Polars interchange and
        actual dynamic-library linkage outside the source checkout.
- [ ] Task: Qualify EasyBuild foss/2023a.
  - [ ] Complete full source build/install and all installed-consumer checks.
- [ ] Task: Qualify EasyBuild foss/2024a.
  - [ ] Complete full source build/install and all installed-consumer checks.
- [ ] Task: Verify generation isolation and complete matrix accounting.
  - [ ] Prove loaded modules and linked libraries come from the selected prefix.
  - [ ] Check no missing cells/features and retain every terminal receipt.
- [ ] Task: Phase verification, agent review and checkpoint per workflow.md.

## P6 — Review, integrate and archive
- [ ] Task: Perform correctness and security/automation review with agents.
  - [ ] Apply findings and run affected checks; reuse unchanged exact-input proof.
  - [ ] Run required final local gate, coverage and hosted checks at a coherent
        milestone; preserve all native matrix gates separately.
- [ ] Task: Deliver one scoped terminal packet to the repository integrator.
  - [ ] Validate receipt hashes, durable artifacts, links and AC1–AC6 coverage.
  - [ ] Integrator reruns read-only integration preflight at the decision point,
        reconciles parent status, opens/updates the PR and merges after gates.
- [ ] Task: Phase verification and final acceptance checkpoint per workflow.md.
- [ ] Task: Archive only after native acceptance and verified integration.

## Execution conventions

Use the verified repo-local Python 3.15.1 and normal hooks. TDD applies to code
changes, not mechanical document checks. Select affected tests during iteration;
run the full local tox gate once per coherent candidate and invalidate evidence
when relevant inputs change. Exact commands are resolved from installed tools
at dispatch. Use the verified edithatogo/voiage remote. `origin` in the current isolated
source/vendor checkout is that repository. Earlier scientific worktree remote
assignments remain historical. No new skill installation is
needed from the planning catalog. Paid resources require separate user direction.

Current release evidence and available-compute boundary are recorded in
[the 2 November reconciliation](../scientific_ecosystem_alignment_20261029/reconciliation-20261102.md).
Existing unchecked native acceptance tasks remain open.


## Linux graph checkpoint — 2 November 2026

All three module/package graph cells passed in [run 41386711954](https://github.com/edithatogo/voiage/actions/runs/41386711954).
The [final receipt manifest](evidence/linux-graph-20261102/final/manifest.json)
binds source, input Git objects, artifact hashes and terminal resource measurements.
Spack uses explicit stable and Polars compiler/bootstrap package identities.
Both EasyBuild generations resolve only Python 3.15.1 and authenticate the actual
candidate sdist before root evaluation. DOT files retain their dependency edges.

This completes the bounded graph checkpoint. Source fetching/vendor closure,
compiler/library builds, installed modules, generation-isolation execution and
scientific consumer acceptance remain P2–P6 work. At this graph checkpoint the new Cargo providers
still had unqualified Git-source archive bytes. The subsequent source/vendor
checkpoint below records their qualification.

The graph checkpoint writer was chat 01a195fd-b970-7391-83a9-9796bbf98ae3 in
/Users/doughnut/Library/Caches/innovate-arrow26-qual.Q7IJ07/voiage on
codex/linux-graph-qualification. Scope is the graph workflow, current provider
recipes, source-contract tests and these receipts. No native build is active.
Previous ownership entries remain historical.


## Linux source/vendor checkpoint — 2 November 2026

The bounded Cargo/compiler-source checkpoint is recorded in
[evidence/linux-source-vendor-20261102/manifest.json](evidence/linux-source-vendor-20261102/manifest.json).
It authenticates six Cargo-backed provider sdists, the frozen Voiage HPC
candidate, both compiler source archives and their bootstrap SDK inputs. Nine
Cargo cells retain locked vendor inventories and all-feature offline metadata.
The independent EasyBuild fetch-only cell proves reproducible bytes for Polars'
three fixed Git sources and verifies their hashes in both current generations.

- [x] Task: Implement source authentication, locked vendor replay and negative
  controls `86d5cce`, with Polars lock/Git-source repairs `d57db19`, `91fdfb6`.
- [x] Task: Review, apply fixes and qualify the source/compiler cells `f44fda7`.
  The review exposed partial-download cleanup and a legitimate synchronized local
  workspace/registry name overlap. Fourteen focused Python 3.15.1 controls pass.
- [x] Task: Retain exact source/lock/vendor/graph/resource identities and terminal
  failure history. Large metadata and command logs are losslessly compressed.
  The manifest hashes both stored and original bytes.

This is a P2 sub-checkpoint. It does not close package-manager fetch/install
acceptance for the complete graph, the observation-runner cases, Rust/LLVM source
compilation, installed scientific probes or generation isolation. Bootstrap SDK
archives are authenticated input, never source-built output. Native acceptance
and P3-P6 remain open. The local full tox attempt was blocked at the disk-space
preflight and is retained as a stop, not a successful gate. Hosted PR/main gates
remain required for the integrated revision.

Current ownership and review disposition are in
[handoff.md](handoff.md) and
[review-source-vendor-20261102.md](review-source-vendor-20261102.md).
