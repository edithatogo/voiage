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
- [~] Make Renovate the sole version-update bot; validate its configuration,
  dependency dashboard, supported managers, vulnerability bypass, Action
  pinning, stability checks, grouping, concurrency, human-review boundaries,
  lock maintenance, and source-pinned submodule updates.
- [ ] Verify the Renovate GitHub App is installed and authorized, observe its
  dashboard and a test PR with required checks, then disable Dependabot
  security updates while retaining GitHub dependency-graph and Dependabot
  alerts. Until that evidence exists, keep security updates enabled to avoid a
  remediation gap.
- [ ] Add Codecov, CodeQL, dependency review at moderate severity, Scorecard,
  secret scanning, push protection, private vulnerability reporting,
  non-provider and validity checks where supported, SBOM, provenance,
  attestation, license, malware/OSV, and release dry-run gates.
- [ ] Upgrade or constrain JupyterLab to a patched release and close open
  Dependabot alerts #64--#68; all currently resolve at JupyterLab 4.6.2, and
  the two high-severity XSS alerts block release.
- [ ] Require the reproducible-build/SBOM job in the active main ruleset and
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
