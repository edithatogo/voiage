# Implementation plan

Status: planned; user-authorized scientific ecosystem delivery includes this
linked track. No tasks below have been executed. Sequential dependencies are P1 -> P2 -> P3 -> P4 ->
P5 -> P6. P2 source controls must pass before P3 fetch/build dispatch.

## P1 — Reconcile ownership and freeze the graph
- [ ] Task: Review the packaging requirement for every source-built compiler.
  - [ ] Record the supported matrix cell, required compiler capability, why a
        qualified binary toolchain is insufficient, and distinct source stages.
  - [ ] Obtain agent correctness/security review before bootstrap dispatch.
  - [ ] Record any proposed matrix change as an explicit scope decision; retain
        the promised matrix until that decision is accepted and documented.
- [ ] Task: Bind current integrator/provider/capacity handoffs to this track.
  - [ ] Record exact owned paths, worktrees, active jobs and evidence revisions.
  - [ ] Map each parent HPC requirement to AC1–AC6 without duplicate execution.
  - [ ] Freeze the Linux Spack and two EasyBuild matrix cells, compiler roles,
        features, source digests and tool versions in a machine-readable manifest.
- [ ] Task: Resolve supported combined-root compiler constraints.
  - [ ] Write and reproduce failing graph-contract regression cases.
  - [ ] Implement the smallest supported provider/graph correction through owner.
  - [ ] Pass the full unmodified Linux root solve and persist the concrete graph.
- [ ] Task: Phase verification, agent review and checkpoint per workflow.md.

## P2 — Qualify fetching and measurement
- [ ] Task: Repair and qualify source resolution and offline execution.
  - [ ] Write failing cases for shadowed paths, incorrect digest, missing input,
        forbidden fetch and unsupported CLI options.
  - [ ] Implement supported pre-fetch restrictions and resolved-source checks.
  - [ ] Run focused tests and exact-version package-manager probes on Linux.
- [ ] Task: Bind a safe per-stage observation runner.
  - [ ] Test descendants in distinct process groups, sampler loss, timeout/OOM,
        failed stage, interruption, cleanup and receipt completeness.
  - [ ] Reuse accepted observation tools; add only missing counters and provenance.
  - [ ] Verify owned-process termination and fail-closed terminal reports.
  - [ ] Freeze runner envelope, concurrency and storage headroom before dispatch.
- [ ] Task: Phase verification, agent review and checkpoint per workflow.md.

## P3 — Measure Rust and LLVM source bootstraps
- [ ] Task: Run a bounded hosted feasibility build for one frozen configuration.
  - [ ] Record stage0 exception, explicit source stages, LLVM source build,
        targets, debug settings, vendoring and source authentication.
  - [ ] Retain per-stage process/CPU/I/O/disk/time measurements and complete logs.
  - [ ] Verify installed compiler identities and actual compile/link operation.
  - [ ] If stopped, preserve terminal evidence and select a supported execution
        route within available authority; never classify a stop as a pass.
- [ ] Task: Qualify the remaining distinct compiler configurations.
  - [ ] Reuse identical accepted compiler output only with exact provenance.
  - [ ] Build differing stable/nightly or generation configurations and verify.
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
at dispatch. Use the verified `voiage` remote for edithatogo/voiage; `origin`
in this checkout points to another repository. No new skill installation is
needed from the planning catalog. Paid resources require separate user direction.
