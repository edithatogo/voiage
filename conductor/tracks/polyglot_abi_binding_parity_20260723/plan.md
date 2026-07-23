# Track Implementation Plan: Polyglot ABI And Binding Parity

## Phase 1: ABI and conformance tests

- [ ] Add failing symbol, layout, ownership, error, lifetime, capability,
  install, and differential tests.
- [ ] Freeze additive ABI v1 types and code-generation inputs.
- [ ] Define per-language public and packaging contracts.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 1: ABI and conformance tests'
  (Protocol in workflow.md).

## Phase 2: Binding implementation

- [ ] Expand Rust facade and C ABI with panic containment.
- [ ] Complete Python, direct R, Julia Artifacts/JLL, and Mojo packages.
- [ ] Generate capabilities, headers, docs, and migration adapters.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Binding implementation'
  (Protocol in workflow.md).

## Phase 3: Installed parity

- [ ] Run clean install/unload/concurrency/error and shared-fixture matrices.
- [ ] Run Miri, sanitizers, fuzzing, semver, ABI, docs, and full quality gates.
- [ ] Reconcile packaging and external registry readiness.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Installed parity'
  (Protocol in workflow.md).

