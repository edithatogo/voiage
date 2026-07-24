# Track Implementation Plan: Stable VOI Rust Core Completion

## Phase 1: Numerical contract tests

- [~] Add failing analytical, property, error, RNG, and serialization tests.
- [ ] Freeze stable estimator, diagnostic, tolerance, tie, and fallback policy.
- [ ] Freeze bias/variance and Monte Carlo error reporting, convergence,
  effective-sample diagnostics, RNG identity, replication, budget, stopping,
  and numerical-error envelopes by estimator family.
- [ ] Record Python/Rust baseline differences and performance budgets.
- [ ] Define deterministic parallel reduction, splittable RNG streams,
  streaming/out-of-core behavior, and memory/latency/energy profiles.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 1: Numerical contract tests'
  (Protocol in workflow.md).

## Phase 2: Rust implementation

- [ ] Implement the missing stable kernels and public Rust facade.
- [ ] Route Python stable APIs through Rust with explicit compatibility paths.
- [ ] Add properties, mutation tests, benchmarks, diagnostics, and fixtures.
- [ ] Validate analytical oracles, independent references, and metamorphic
  invariants in addition to differential fixtures.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Rust implementation'
  (Protocol in workflow.md).

## Phase 3: Stable-core evidence

- [ ] Run Rust, Python, fixture, mutation, benchmark, and full tox gates.
- [ ] Reconcile docs, roadmap, changelog, capabilities, and maturity.
- [ ] Record v1.1 promotion evidence and unresolved external gates.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Stable-core evidence'
  (Protocol in workflow.md).
