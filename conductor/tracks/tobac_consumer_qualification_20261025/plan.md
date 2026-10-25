# Approved plan — implementation hold

All tasks are future work. Do not start or auto-progress without a new user instruction.
[Specification](spec.md). [Repository workflow](../../workflow.md).

## Phase 1 — Consumer reconciliation

- [ ] Task P1.1: Refresh producer/master revisions, ownership and current contracts.
- [ ] Task P1.2: Map consumer requirements to existing behavior and evidenced gaps.
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

- [ ] Task P3.1: Build and install independent artifacts; verify actual import/artifact identities.
- [ ] Task P3.2: Execute consumer contracts, negative cases, precision and scientific invariants.
  - [ ] Prove a bounded extension preserves reference scenarios and record resolved parameters, provenance, model/schema identity and execution settings separately (R11.1–R11.5).
- [ ] Task P3.3: Review existing engineering/context/security controls; add only justified gaps.
- [ ] Task P3.4: Phase verification and checkpoint under the repository workflow.

## Phase 4 — Handoff

- [ ] Task P4.1: Execute full relevant native, clean-package, coverage, security and hosted gates.
- [ ] Task P4.2: Reconcile main and master track, publish only within authority, verify receiving refs/files.
- [ ] Task P4.3: Deliver exact artifact/API/schema receipts and rollback notes to tobac.
- [ ] Task P4.4: Phase verification and checkpoint including repository review requirements.
