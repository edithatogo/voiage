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
- [~] Upgrade or constrain JupyterLab to a patched release and close open
  Dependabot alerts #64--#68; all currently resolve at JupyterLab 4.6.2, and
  the two high-severity XSS alerts block release.
- [x] Require the reproducible-build/SBOM job in the active main ruleset and
  reconcile required contexts against actual hosted check names without
  weakening signed commits, linear history, thread resolution, or strict
  up-to-date checks.
- [ ] Add scheduled and pre-release live posture reconciliation for open
  dependency, code-scanning, secret-scanning, and workflow-audit findings,
  security settings, ruleset drift, Renovate activity, and artifact retention.
- [ ] Add deterministic generated-artifact and clean-install gates.
- [ ] Add registry-to-code-to-binding-to-doc claim conformance, ADR and
  deprecation-ledger validation, adversarial ML/agent fixtures, and controlled
  ecosystem-drift proposals.
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
- [~] Dependabot alerts #64--#68 close only after `b03d9a85` reaches the
  default branch and GitHub rescans the patched JupyterLab lock.
- [ ] Renovate App activation, its dependency dashboard, and one checked test
  PR remain external GitHub evidence before Dependabot security updates can be
  disabled.
- [x] Commit `5e5091d` replaces the Rust release workflow's long-lived
  crates.io token with the official commit-pinned OIDC authentication action,
  job-scoped `id-token: write`, and the `crates-io` environment. Registry-side
  trusted-publisher registration remains an external human gate.

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
