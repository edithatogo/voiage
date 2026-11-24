# Implementation plan — tobac consumer qualification

The original planning-only hold was superseded by the user's 29 October
direction to implement, review and archive this track. The [installed M2 retail
workflow](evidence/installed-m2-retail-20261124/README.md) passed on 24 November.
Earlier candidate results are in the [November qualification packet](evidence/installed-20261101/README.md).
[Specification](spec.md). [Repository workflow](../../workflow.md).

## Phase 1 — Consumer reconciliation

- [x] Task P1.1: Refresh producer/master revisions, ownership and current contracts.
  The [current master record](https://github.com/edithatogo/tobac/tree/fe06ef5958ba0f218317a738fe8de95973300a12/conductor/archive/scientific_ecosystem_20261025)
  retains tobac ownership of the domain model. Its first Voiage use is the
  frozen M2 retail case through the published 3.1.0 optional backend.
- [x] Task P1.2: Map consumer requirements to existing behavior and evidenced gaps.
- [ ] Task P1.3: Write failing consumer and independent-oracle tests before changes.
  - [ ] Map master R11.1–R11.5 to existing parameter/model/execution contracts; write round-trip, invalid-input, scenario-isolation, input-mutation and I/O-independent calculation cases.
- [ ] Task P1.4: Phase verification and checkpoint under the repository workflow.

## Phase 2 — Bounded producer support

- [ ] Task P2.1: Qualify required Python/native dependencies and record genuine blockers.
- [ ] Task P2.2: Implement only demonstrated reusable gaps within existing core/API ownership.
  - [ ] Close only demonstrated parameter/model/execution separation gaps through existing typed/native APIs, preserving structural invariants and domain ownership.
- [ ] Task P2.3: Qualify optional backend/interchange paths against exact references and measured value.
- [ ] Task P2.4: Phase verification and checkpoint under the repository workflow.

## Phase 3 — Installed consumer and evidence

- [x] Task P3.1: Build and install independent artifacts; verify actual import/artifact identities.
- [~] Task P3.2: Execute consumer contracts, negative cases, precision and scientific invariants.
  - [x] Qualify the [installed frozen M2 retail EVPI workflow](evidence/installed-m2-retail-20261124/README.md)
    with both wheels, the frozen reference, reversed state order and error controls.
  - [ ] Prove a bounded extension preserves reference scenarios and record resolved parameters, provenance, model/schema identity and execution settings separately (R11.1–R11.5).
- [ ] Task P3.3: Review existing engineering/context/security controls; add only justified gaps.
- [ ] Task P3.4: Phase verification and checkpoint under the repository workflow.

## Phase 4 — Handoff

- [ ] Task P4.1: Execute full relevant native, clean-package, coverage, security and hosted gates.
- [ ] Task P4.2: Reconcile main and master track, publish only within authority, verify receiving refs/files.
- [ ] Task P4.3: Deliver exact artifact/API/schema receipts and rollback notes to tobac.
- [ ] Task P4.4: Phase verification and checkpoint including repository review requirements.

Current release evidence and available-compute boundary are recorded in
[the 2 November reconciliation](../scientific_ecosystem_alignment_20261029/reconciliation-20261102.md).
Existing unchecked native acceptance tasks remain open.
