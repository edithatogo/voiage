# Track Implementation Plan: Stable VOI Rust Core Completion

## Phase 1: Numerical contract tests

- [x] Add failing analytical, property, error, RNG, and serialization tests.
  (`ebd32c6b`; net-benefit is deterministic and array-returning, so RNG and
  result-envelope requirements are inapplicable; analytical, property,
  dimension, non-finite, overflow, adapter, and PyO3 tests cover its contract)
- [x] Freeze stable estimator, diagnostic, tolerance, tie, and fallback policy.
  (`a62e79d9`; the schema-validated v1.1 registry covers every approved stable
  method through an executable profile or explicit delegation and freezes
  comparison tolerances, ties, clipping, failures, diagnostics, and fallbacks)
- [x] Freeze bias/variance and Monte Carlo error reporting, convergence,
  effective-sample diagnostics, RNG identity, replication, budget, stopping,
  and numerical-error envelopes by estimator family.
  (`e4e0f5c1`; a schema-validated family policy plus portable runtime envelope
  separates statistical, approximation, and floating-point error and makes
  replay, convergence, resource, and stopping evidence explicit)
- [x] Record Python/Rust baseline differences and performance budgets.
  (`25658925`; the versioned contract binds existing Rust direct-kernel and
  Python NumPy-oracle budgets to their executable sources and prohibits
  unmatched cross-language speedup claims)
- [x] Define deterministic parallel reduction, splittable RNG streams,
  streaming/out-of-core behavior, and memory/latency/energy profiles.
  (`d0167c9e`; stable reductions are fixed-order sequential, EVSI streams are
  indexed and splittable by contract, unsupported parallel/out-of-core modes
  fail closed, and every stable method has bounded resource claims)
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 1: Numerical contract tests'
  (Protocol in workflow.md).

## Phase 2: Rust implementation

- [x] Implement the missing stable kernels and public Rust facade. (`ebd32c6b`;
  net-benefit was the sole stable authority gap)
- [x] Route Python stable APIs through Rust with explicit compatibility paths.
  (`ebd32c6b`; both array and scalar helpers use PyO3, with warned v1
  elementwise inference and explicit `thresholds`/`elementwise` policies)
- [x] Add properties, mutation tests, benchmarks, diagnostics, and fixtures.
  (`f89e70c2`; analytical/property/error coverage, the CI-gated foundational
  benchmark, shared Python/Rust normal-edge-invalid fixtures, and a fail-closed
  bounded Rust mutation audit are complete)
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
