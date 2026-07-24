# Track Implementation Plan: External VOI Library Feature Parity

## Phase 1: Landscape contract

- [x] Add failing registry, freshness, license, and traceability tests.
  (`2630e39`)
- [x] Define discovery, inclusion, feature, parity, and exclusion schemas.
  (`2630e39`)
- [x] Pin initial package/tool candidates and authoritative sources.
  (`2630e39`)
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update. (`2630e39`)
- [x] Automated review and validation checkpoint. (`c4c4fab`; registry,
  governance, Ruff, and live GitHub reconciliation passed)
- [ ] Conductor - User Manual Verification 'Phase 1: Landscape contract'
  (Protocol in workflow.md).

## Phase 2: Feature census and parity

- [x] Inventory source, tests, docs, examples, vignettes, schemas, and
  releases. Every observed external feature now has a versioned artifact
  record; GitHub source, test, and example evidence is pinned to an exact
  reviewed commit, and absent artifact classes are explicit limitations.
  The refreshed search also added seven previously omitted external records,
  including the source-rights-blocked BayesCal-VOI supplement, and corrected
  `decision-security`, decisionSupport, and TRD CEA Toolkit provenance.
  (`5dadf91`, `7443253`, `ae77dc1`)
- [x] Build independent fixtures and feature-to-method mappings. (`6e3ebb1`)
- [~] Implement missing justified features without competitor runtime imports.
  - [x] Add the Rust-authoritative expected opportunity-loss foundation required
    by BCEA, dampack, and expected-loss reporting. (`95beb20`)
- [ ] Add optional migration adapters and reviewed exclusions.
- [ ] Preserve archived or unavailable tools with unique features, maintenance
  state, last verifiable behavior, and closest supported workflow.
- [ ] Normalize licenses to SPDX where possible and prohibit reference-fixture
  copying until source rights are recorded.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Feature census and parity'
  (Protocol in workflow.md).

## Phase 3: Public evidence

- [x] Generate the test-linked comparison. (`2630e39`; clean competitor-absent
  isolation tests remain part of the implementation phase)
- [ ] Run license, provenance, docs, and complete quality gates.
- [x] Schedule quarterly and pre-minor-release refresh checks. (`6e3ebb1`)
- [ ] Include registry, toolchain, lockfile, action, and source-pinned
  documentation-plugin drift in evidence-preserving refresh proposals.
- [~] Produce a machine-readable gap report and bounded GitHub triage updates;
  never generate duplicate issues or overwrite human notes.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Public evidence'
  (Protocol in workflow.md).

## Review fixes

- [x] Add a separately versioned method-evidence registry so software feature
  links cannot be mistaken for complete scientific support. (`c4c4fab`)
- [x] Require every positive external parity claim to name competitor-free
  fixtures and tests after checkpoint review. (`6e3ebb1`)
- [x] Replace mutable upstream `HEAD` evidence links with exact commit-pinned
  source, test, and example records and expose extraction limitations in the
  generated matrix and routed gap report. (`ae77dc1`)
