# Track Implementation Plan: Quality, Security, Release, And Registry Automation

## Phase 1: Assurance contracts

- [~] Add failing workflow, matrix, provenance, drift, dependency-automation,
  GitHub-posture, and release-claim tests.
- [ ] Define required versus scheduled/manual/external lanes and budgets.
- [ ] Define reproducibility, artifact identity, and registry dry-run policy.
- [ ] Define numerical error, memory, latency, energy, estimator-assurance, and
  deterministic-parallelism budgets by release tier.
- [ ] Freeze `decision-registry-cards` under
  [#580](https://github.com/edithatogo/voiage/issues/580): versioned decision,
  ownership, alternatives, utility/risk/constraints, uncertainty/information,
  sources, perspectives, costs/time/implementation, assumptions/provenance/
  rights, method/capability, result/assurance, limitations, review, audit and
  deployment/monitoring fields.
- [ ] Freeze `local-decision-studio-reporting` under
  [#581](https://github.com/edithatogo/voiage/issues/581): local-first guided
  modelling, validation, scenario/sensitivity, break-even/policy-switch,
  portfolio, stakeholder/constraint, report/export, accessibility, redaction
  and audit behavior over the public API.
- [ ] Freeze `enterprise-integration-adapters` under
  [#583](https://github.com/edithatogo/voiage/issues/583): adapter classes,
  canonical mappings, idempotency, lineage, auth, tenancy, privacy/rights,
  retry/pagination/rate-limit/offline and capability negotiation.
- [ ] Freeze `decision-correctness-industry-assurance` under
  [#584](https://github.com/edithatogo/voiage/issues/584): counterfactual,
  policy-recovery, regret, invariance, constraint, causal, calibration,
  optimization, estimator, shift, provenance, snapshot and human-review gates.
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
- [ ] Add deterministic generated-artifact and clean-install gates.
- [ ] Add registry-to-code-to-binding-to-doc claim conformance, ADR and
  deprecation-ledger validation, adversarial ML/agent fixtures, and controlled
  ecosystem-drift proposals.
- [ ] Implement #580 as a schema-validated registry with deterministic cards,
  immutable revisions, explicit draft/review/approved/deployed/retired states,
  redaction and portable export.
- [ ] Implement #581 as an optional local application that composes public
  contracts and estimators, never duplicates numerical engines, and produces
  accessible machine-readable and human-readable decision reports.
- [ ] Implement #583 as separately packaged thin adapters with contract tests,
  recorded lineage and failure behavior, and no default network dependency.
- [ ] Implement #584 as a generated assurance matrix binding every industry
  method, template, example, binding, adapter and report claim to executable
  evidence or an explicit gate.
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

## Phase 3: Staged release evidence

- [ ] Run local and hosted required gates for v1.1, v1.2, and v1.3 candidates.
- [ ] Run clean registry/studio/adapters/assurance matrices for #580, #581,
  #583 and #584, including accessibility and named human-review evidence where
  required.
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
