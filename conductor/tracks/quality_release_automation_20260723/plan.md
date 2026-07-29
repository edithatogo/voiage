# Track Implementation Plan: Quality, Security, Release, And Registry Automation

## Phase 1: Assurance contracts

- [~] Add failing workflow, matrix, provenance, drift, dependency-automation,
  GitHub-posture, and release-claim tests.
- [ ] Define required versus scheduled/manual/external lanes and budgets.
- [ ] Define reproducibility, artifact identity, and registry dry-run policy.
- [ ] Define numerical error, memory, latency, energy, estimator-assurance, and
  deterministic-parallelism budgets by release tier.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 1: Assurance contracts'
  (Protocol in workflow.md).

## Phase 2: Automation implementation

- [ ] Implement cross-platform, Rust, binding, example, ML, provenance, and
  freshness workflows.
- [x] Consolidate the repository's duplicate Python security linting into
  Ruff, publish a machine-readable quality-tool disposition registry, and
  retain non-overlapping dead-code, type, dataflow, dependency, orchestration,
  mutation, and profiling controls until explicit parity evidence supports
  further consolidation.
- [x] Remove the redundant `tomli` backport now that Python 3.12 is the
  supported floor, update stale local validation references, and record the
  retained dependency-audit boundary. (`74e4be05`)
- [x] Make `ty` the fast routine typing gate and move BasedPyright to explicit
  scheduled/manual strict-assurance lanes, with a documented release-review
  boundary. (`8f03ed09`)
- [x] Clear high-confidence Vulture findings and make the whole-program
  dead-code check blocking, while preserving public parameter names and
  recording the reviewed ecosystem-drift baseline. (`59c61fe9`)
- [x] Make Renovate the sole version-update bot; validate its configuration,
  dependency dashboard, supported managers, vulnerability bypass, Action
  pinning, stability checks, grouping, concurrency, human-review boundaries,
  lock maintenance, and source-pinned submodule updates.
- [ ] Verify the Renovate GitHub App is installed and authorized, observe its
  dashboard and a test PR with required checks, then disable Dependabot
  security updates while retaining GitHub dependency-graph and Dependabot
  alerts. Until that evidence exists, keep security updates enabled to avoid a
  remediation gap.
- [~] Add Codecov, CodeQL, dependency review at moderate severity, Scorecard,
  secret scanning, push protection, private vulnerability reporting,
  non-provider and validity checks where supported, SBOM, provenance,
  attestation, license, malware/OSV, and release dry-run gates.
  - [x] Replace the long-lived crates.io publication secret with the official
    pinned OIDC trusted-publishing action and a named deployment environment.
    (`5e5091d`)
  - [~] Reconcile the live default-branch code-scanning queue. The four
    crates.io trusted-publishing findings have a tested fix in `5e5091d`; the
    Scorecard SAST-coverage finding and remaining quality findings require
    default-branch rescan evidence or bounded reviewed dispositions.
    - [x] Enable exact-ref manual and merge-queue CodeQL analysis (`30693e2`)
      so candidate commits can be scanned before they reach the default branch.
    - [x] Remove all pull-request-blocking CodeQL errors (`9f4275b`,
      `89c7962`, `ece829e`); exact-branch open findings fell from 65 to 26,
      and run 30069242255 plus the PR-attached CodeQL check passed.
    - [x] Remove the high-risk pre-publication `contents: write` token flagged
      by Scorecard alert #1095. (`6dfb97da`; staging uses immutable same-run
      Actions artifacts and reviewed digests; only final GitHub publication
      retains narrowly scoped write permission after TestPyPI and PyPI)
    - [ ] Reconcile the default-branch queue after merge and disposition the
      remaining non-security quality notes without weakening lazy imports or
      masking genuine scientific defects.
- [x] Upgrade JupyterLab to 4.6.2 and close Dependabot alerts #64--#68.
  GitHub's live Dependabot API returned zero open alerts on 2026-07-26, and
  local Python and production-documentation dependency audits reported no
  known vulnerabilities.
- [x] Make dependency-frontier verification lock-aware: preserve declared
  compatibility floors, compare the upgraded `uv.lock` resolution with the
  newest stable release admitted by each range, and disclose newer releases
  outside that range without silently widening it.
- [x] Require the reproducible-build/SBOM job in the active main ruleset and
  reconcile required contexts against actual hosted check names without
  weakening signed commits, linear history, thread resolution, or strict
  up-to-date checks.
- [ ] Add scheduled and pre-release live posture reconciliation for open
  dependency, code-scanning, secret-scanning, and workflow-audit findings,
  security settings, ruleset drift, Renovate activity, and artifact retention.
- [x] Prepare a fail-closed OpenSSF Best Practices badge evidence map and
  application protocol without claiming submission or award.
  (`82af3f7b`; maintainer attestations, external submission, and the public
  badge URL remain an external gate)
- [x] Add deterministic generated-artifact and clean-install gates. The SBOM
  workflow rebuilds distributions from a frozen checkout, installs the wheel
  into an isolated environment, retains reproducible evidence, and fails on
  missing artifacts; regression coverage protects the contract.
- [~] Add registry-to-code-to-binding-to-doc claim conformance, ADR and
  deprecation-ledger validation, adversarial ML/agent fixtures, and controlled
  ecosystem-drift proposals.
  - [x] Protect the existing landscape freshness projections for method,
    implementation, upstream-feature, and feature-matrix claims.
  - [x] Protect the versioned compatibility/deprecation policy contract and
    its architecture/versioning documentation cross-references.
  - [ ] Add explicit ADR/deprecation-ledger cross-reference validation.
  - [x] Add deterministic adversarial ML/agent fixture coverage; full utility
    estimators and provider/model validation remain owned by the ML/LLM track.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Automation implementation'
  (Protocol in workflow.md).

### Phase 2 implementation evidence

- [x] Commit `b03d9a85` removes Dependabot version-update configuration,
  validates the Renovate policy and all detected managers, upgrades JupyterLab
  to 4.6.2, strengthens dependency review to moderate severity, and records
  the required SBOM ruleset context.
- [x] Review fix in `b03d9a85`: constrain Ruff below the unreviewed 0.16
  breaking frontier after the upgrade rehearsal exposed new lint semantics.
- [x] Review fix in `b03d9a85`: normalize the Astro `/voiage` deployment base
  in documentation-link validation and add regression coverage.
- [x] Dependabot alerts #64--#68 are closed; the live repository API returned
  zero open alerts on 2026-07-26 after the JupyterLab 4.6.2 remediation.
- [ ] Renovate App activation, its dependency dashboard, and one checked test
  PR remain external GitHub evidence before Dependabot security updates can be
  disabled.
- [x] Commit `5e5091d` replaces the Rust release workflow's long-lived
  crates.io token with the official commit-pinned OIDC authentication action,
  job-scoped `id-token: write`, and the `crates-io` environment. Registry-side
  trusted-publisher registration remains an external human gate.
- [x] Consolidated duplicate Bandit source checks into Ruff's selected `S`
  rules, removed Bandit and its transitive `stevedore` dependency, and added a
  tested quality-tool registry that records why whole-program dead-code,
  independent typing, CodeQL dataflow, dependency, orchestration, mutation,
  profiling, and prose controls remain distinct.
- [x] Commit `74e4be05` removes the obsolete `tomli` backport, retires the
  Safety/pip-tools temporary-resolution lane in favour of the pinned
  pip-audit/SBOM path, and updates the supplementary local validator to the
  active toolchain. The complete tox matrix passed with 91.07% coverage.
- [x] Commit `8f03ed09` makes `ty` the fast local and pull-request typing
  gate, with BasedPyright retained as an explicit scheduled/manual strict
  assurance lane and release-review input.
- [x] Commit `59c61fe9` clears all high-confidence Vulture findings, makes the
  CI dead-code check blocking, adds a regression guard, and refreshes the
  reviewed workflow-drift baseline. The full tox matrix passed with Python
  3.12--3.14, minimum and maximum dependency lanes, Astro/polyglot docs,
  repository harness, and coverage.
- [x] Added regression coverage for the existing frozen reproducible-build,
  isolated SBOM clean-install, and fail-closed artifact-retention workflow.
- [x] Added regression coverage for the landscape freshness workflow's four
  registry-to-public-claim projections; ADR/deprecation-ledger cross-links and
  adversarial ML/agent fixtures remain open subitems.
- [x] Added explicit checks linking the core architecture decision, normative
  compatibility/deprecation policy, and versioning documentation.
- [x] Added a deterministic offline adversarial ML/agent fixture covering
  prompt injection, retrieval poisoning, correlated judge failure, provider
  drift, and human escalation, with explicit decision loss and review cost.

## Phase 3: Staged release evidence

- [ ] Run local and hosted required gates for v1.1, v1.2, and v1.3 candidates.
- [ ] Require zero unresolved critical/high dependency or secret findings;
  require each moderate finding to be fixed or carry a time-bounded,
  maintainer-confirmed risk record with compensating controls and review date.
- [ ] Reproduce artifacts and reconcile external registry/publication states.
- [ ] Complete release documentation without performing unauthorized publish.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Staged release evidence'
  (Protocol in workflow.md).
