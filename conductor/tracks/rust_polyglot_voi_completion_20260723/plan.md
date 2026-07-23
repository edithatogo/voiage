# Track Implementation Plan: Comprehensive Rust-First Polyglot VOIAGE Completion

## Phase 1: Governance foundation

- [x] Test the intended issue, native-subissue, Project 28, and Conductor
  topology.
- [x] Create GitHub parent #313 and native subissues #314--#323.
- [x] Populate live Project 28 fields using its supported option vocabulary.
- [x] Add and run repository governance validation.
- [x] Commit the governance artifacts, attach a git note, record the short
  commit SHA, and commit the plan update. (`c576ad14`, rebased)
- [x] Automated Conductor review and validation checkpoint.
- [x] Review fix: isolate per-track validator errors so one malformed track
  cannot suppress inspection of later tracks. (`9cf3849d`, rebased)
- [x] Integration fix: make the frozen v1 baseline require its recorded tracks
  as a subset, rather than prohibit later separately governed programmes.
- [x] Python 3.12 full suite: 2,032 passed and 16 skipped.
- [ ] Conductor - User Manual Verification 'Phase 1: Governance foundation'
  (Protocol in workflow.md).

## Phase 2: Contract and implementation programme

- [ ] Complete the method and external-library censuses.
- [ ] Freeze v1.1 stable method, numerical, serialization, and ABI contracts.
- [ ] Complete the stable Rust and Value of Perspective tracks.
- [ ] Complete frontier, ML/LLM/agent, and binding tracks.
- [ ] Commit each functional task, attach a git note, record its short commit
  SHA, and commit the plan update.
- [ ] Automated Conductor review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Contract and implementation programme'
  (Protocol in workflow.md).

## Phase 3: Evidence, releases, and closeout

- [ ] Complete datasets, worked examples, contribution records, and automation.
- [ ] Run the full tox, Rust, binding, docs, provenance, and governance gates.
- [ ] Reconcile v1.1, v1.2, and v1.3 release evidence without inferring external
  acceptance.
- [ ] Commit closeout evidence, attach a git note, record the short commit SHA,
  and commit the plan update.
- [ ] Final Conductor review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Evidence, releases, and closeout'
  (Protocol in workflow.md).

## Closure boundary

Green planning or one release lane does not complete the programme. All
repository-owned child tasks must be complete and remaining external gates
must be explicit.
