# Track Implementation Plan: Polyglot ABI And Binding Parity

## Phase 1: ABI and conformance tests

- [ ] Add failing symbol, layout, ownership, error, lifetime, capability,
  install, and differential tests.
  - [x] Generate a caller-owned C-ABI capability document from the canonical
    stable-core status and fail on generated-artifact drift. (`71c9d15f`)
  - [x] Add a fixed-width expected-loss result with caller-owned per-strategy
    arrays and strict ownership, alignment, and capacity validation.
    (`b68b105c`)
  - [x] Add Rust-authoritative ENBS to the versioned C ABI with typed
    validation and panic containment. (`7d54581c`)
  - [x] Add deterministic dominance, frontier, and ICER results with
    caller-owned classification, index, and transition arrays. (`6928b47b`)
  - [x] Add threshold-aligned CEAF probabilities, intervals, expected net
    benefit, and assurance through caller-owned arrays. (`f2d0d5b6`)
  - [x] Add typed structural EVPI and structural EVPPI results with explicit
    model probabilities, selected-structure indices, and sampling assurance.
    (`feeb342c`)
  - [x] Add the stable full-sample linear-regression EVPPI result with fit
    dimensions and an explicit incomplete-assurance state. (`4013003f`)
  - [x] Add typed results for the promoted Rust-native linear-regression and
    moment-matching EVSI estimators without advertising the distinct Python
    two-loop compatibility path as Rust-native. (`719d3d49`)
- [ ] Freeze additive ABI v1 types and code-generation inputs.
  - [x] Add the fixed-width v1.1 typed EVPI assurance result and retain all
    scalar v1.0 entry points.
- [ ] Include the canonical Decision Problem and estimator-assurance envelopes
  in Arrow, JSON, C ABI, and language-native representations.
  - [x] Carry EVPI sample-average variance and Monte Carlo error through the C
    ABI typed result.
- [ ] Define per-language public and packaging contracts.
  - [x] Freeze and validate the public Rust facade/package contract.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 1: ABI and conformance tests'
  (Protocol in workflow.md).

## Phase 2: Binding implementation

- [ ] Expand Rust facade and C ABI with panic containment.
  - [x] Add the publishable, module-qualified `voiage` Rust facade without
    binding-layer dependencies.
  - [ ] Expand the typed C ABI and retain panic containment across every new
    entry point.
- [ ] Complete Python, direct R, Julia Artifacts/JLL, and Mojo packages.
- [ ] Generate capabilities, headers, docs, and migration adapters.
- [ ] Generate deterministic unsupported-method responses and fail CI when a
  binding or document advertises capabilities absent from the registry.
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
