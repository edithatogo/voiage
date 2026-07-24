# Track Specification: Quality, Security, Release, And Registry Automation

## Overview

Extend the existing assurance system to cover the complete polyglot programme
and prevent unsupported release claims.

## Requirements

Add cross-platform/arm64 matrices; Rust properties, mutation, fuzz, Miri,
sanitizers, semver, ABI, and performance; clean package installs; randomized
and golden differential fixtures; executed examples; ML determinism,
calibration, drift, utility, stopping, privacy, and fallback; landscape and
literature freshness; dataset rights and hashes; docs, citation, CRediT,
Authentext, and AI disclosure; Codecov, Renovate, CodeQL, dependency review,
Scorecard, SBOM, provenance, secrets, and license gates.

Also validate registry-to-code-to-binding-to-documentation claims, ADR and
deprecation ledgers, numerical and resource budgets, deterministic parallel
execution, adversarial ML/agent fixtures, and reviewed ecosystem-drift
proposals.

Renovate is the sole version-update bot. GitHub's dependency graph and
Dependabot alerts remain enabled as advisory inputs, but `dependabot.yml` is
absent. Renovate must cover Python/PEP 621, Cargo, npm, Pixi, pre-commit,
Docker Compose, GitHub Actions, lockfiles, and git submodules; bypass normal
schedules for vulnerability repairs; expose a dependency dashboard; pin
Actions and container images; apply release-age and artifact checks; and
require human review for security, major, numerical, executable-hook,
environment, container, submodule, and lock-maintenance changes. Dependabot
security updates remain temporarily enabled only until the Renovate App has
produced a verified dashboard and test PR, preventing an alert-remediation
gap.

Repository security posture must be reconciled live: active maximal-quality
ruleset, required SBOM and dependency-review checks, CodeQL security-and-quality
queries, secret scanning and push protection, non-provider patterns and
validity checks where supported, private vulnerability reporting, action SHA
pinning, open vulnerability and secret alert inventories, and signed release
provenance.

Dry-run PyPI, crates.io, CRAN-compatible, Julia General, and Mojo packaging.
Produce reproducible signed artifacts and checksums only through authorized
release workflows.

## Failure and external policy

Flaky numerics, hidden network access, unsupported runners, nondeterministic
artifacts, stale generation, unresolved critical/high vulnerabilities,
unresolved secrets, inactive dependency automation, or optional-lane overclaim
block release. Moderate vulnerabilities require remediation or a
time-bounded, owner-confirmed risk decision before release.
Credentials, signing, publication, and registry acceptance remain external.

## Acceptance criteria

Required clean matrices pass, artifacts reproduce, evidence is linked, and
every external gate is explicit for staged v1.1--v1.3 releases.
