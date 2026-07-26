# Track Implementation Plan: Stable VOI Rust Core Completion

## Phase 1: Numerical contract tests [checkpoint: c5a7ad0e]

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
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update. (`25658925`, `d0167c9e`, and Conductor evidence commits
  `fa7217a9` and `f71224a7`)
- [x] Review Fixes. (`6cf10968`; refreshed the reviewed ecosystem workflow
  digest after the Phase 1 CI mutation job was added; no scientific or runtime
  contract changed)
- [x] Automated review and validation checkpoint. (`c5a7ad0e`; 2101 Python
  tests passed with 16 documented skips, the Rust workspace and warnings-denied
  clippy passed, all tox environments passed with 90.99% coverage, Astro built
  708 pages with valid links, and local/live GitHub reconciliation passed)
- [ ] Conductor - User Manual Verification 'Phase 1: Numerical contract tests'
  (Protocol in workflow.md).

## Phase 2: Rust implementation [checkpoint: 46d86e8c]

- [x] Implement the missing stable kernels and public Rust facade. (`ebd32c6b`;
  net-benefit was the sole stable authority gap)
- [x] Route Python stable APIs through Rust with explicit compatibility paths.
  (`ebd32c6b`; both array and scalar helpers use PyO3, with warned v1
  elementwise inference and explicit `thresholds`/`elementwise` policies)
- [x] Add properties, mutation tests, benchmarks, diagnostics, and fixtures.
  (`f89e70c2`; analytical/property/error coverage, the CI-gated foundational
  benchmark, shared Python/Rust normal-edge-invalid fixtures, and a fail-closed
  bounded Rust mutation audit are complete)
- [x] Validate analytical oracles, independent references, and metamorphic
  invariants in addition to differential fixtures. (`2cd23b7b`; a
  schema-validated map covers all 11 stable profiles, adds exact and affine
  two-loop Python evidence plus structural Rust properties, treats
  differential tests as supplementary, and exposes the two-loop Rust-authority
  gap)
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update. (`2cd23b7b`; Conductor evidence reconciled in `1df1525d`)
- [x] Review Fixes. (`99ebb054`; synchronized the normative Astro API
  reference with the corrected EVSI Rust/Python implementation boundary)
- [x] Automated review and validation checkpoint. (`46d86e8c`; 2108 top-level
  Python tests, the Rust workspace, warnings-denied clippy, every tox
  environment, 92.46% coverage, a 708-page Astro build, and local/live GitHub
  reconciliation passed)
- [ ] Conductor - User Manual Verification 'Phase 2: Rust implementation'
  (Protocol in workflow.md).

## Phase 3: Stable-core evidence [checkpoint: d501ef95]

- [x] Run Rust, Python, fixture, mutation, benchmark, and full tox gates.
  (`2b6ca058`; 2108 top-level tests, the Rust workspace, all tox environments,
  92.46% coverage, ten hosted checks, four-of-four bounded mutation kills, and
  native benchmarks within declared budgets passed)
- [x] Reconcile docs, roadmap, changelog, capabilities, and maturity.
  (`7cfd9525`; a generated status registry now joins the normative stable-core
  contracts and keeps API stability distinct from method assurance and
  aggregate v1.1 release readiness)
- [x] Record v1.1 promotion evidence and unresolved external gates.
  (`07589392`; the hash-bound candidate remains blocked and separates
  repository, maintainer, and Python/R/Julia/Mojo/Rust distribution gates
  without extending the scientific-freeze approval)
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update. (`7cfd9525`, `07589392`; Conductor evidence reconciled in
  `223fdd66` and `a8a4b063`)
- [x] Final review and validation checkpoint. (`d501ef95`; no High or Critical
  findings, 2,116 Python tests with 16 documented skips, the Rust workspace,
  warnings-denied Clippy, all tox environments with 92.46% coverage, 708 Astro
  pages, generated-contract drift checks, and live reconciliation passed)
- [ ] Conductor - User Manual Verification 'Phase 3: Stable-core evidence'
  (Protocol in workflow.md).

## Phase 4: Runtime statistical assurance

- [x] Add Rust-owned sample-average assurance to expected loss and propagate it
  through the typed Python result and canonical serialization. (`41253fc9`;
  Rust owns variance and Monte Carlo standard error, the typed portable
  envelope is serialized canonically, and unavailable replication,
  convergence, ESS, and RNG evidence remains explicit)
- [x] Integrate fail-closed single-fit envelopes for linear and callback
  regression EVSI and moment-based EVSI, preserving unavailable variance,
  convergence, ESS, and RNG evidence as null. (`f975ea2b`)
- [x] Add Rust-owned resample variance, Monte Carlo error, interval, indexed
  RNG identity, and budget assurance to seeded-bootstrap EVSI. (`dd1e132b`)
- [ ] Integrate statistically valid envelopes for the remaining stable
  sample-average, EVPPI regression, CEAF, and structural methods, plus
  replicated convergence evidence for EVSI estimators.
- [ ] Regenerate stable-core maturity and promotion evidence from runtime
  assurance coverage.
- [ ] Commit, attach git notes, record task SHAs, and reconcile evidence.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification
  'Phase 4: Runtime statistical assurance' (Protocol in workflow.md).
