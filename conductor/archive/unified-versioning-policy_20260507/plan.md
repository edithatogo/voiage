# Unified Versioning And Release Synchronization - Implementation Plan

## Phase 1: Define Canonical Version Policy

- [x] Task: Document the canonical version source and release rules.
    - [x] Record that `pyproject.toml` is the canonical version source for the current release line.
    - [x] Clarify that binding manifests must mirror the same SemVer release.
    - [x] Describe the tag-driven release trigger model for package publishing.
- [x] Task: Add the versioning policy docs.
    - [x] Create a developer-guide page that explains the version source of truth and manifest alignment rules.
    - [x] Update the release/binding docs to point to the policy page.
    - [x] Refresh the roadmap wording so version synchronization is treated as an explicit implemented policy, not a vague future concern.
- [x] Task: Conductor - Automated Review and Checkpoint 'Define Canonical Version Policy' (Protocol in workflow.md)

## Phase 2: Synchronize Package Manifests

- [x] Task: Align all binding package versions to the canonical repo version.
    - [x] Update the TypeScript package manifest version.
    - [x] Update the Julia package manifest version.
    - [x] Update the Rust crate version.
    - [x] Update the .NET package version.
    - [x] Update the R package version.
- [x] Task: Add a version-alignment validator and tests.
    - [x] Implement a repository-local validation script that reads the canonical version and checks all binding manifests.
    - [x] Add tests that cover the validator success path and at least one drift/failure path.
    - [x] Ensure the validator reports actionable errors when a manifest diverges.
- [x] Task: Conductor - Automated Review and Checkpoint 'Synchronize Package Manifests' (Protocol in workflow.md)

## Phase 3: Wire Validation Into CI And Release Docs

- [x] Task: Add the versioning gate to CI and local automation.
    - [x] Add a dedicated tox environment or equivalent command for the version validator.
    - [x] Add a CI job that runs the validator on every push and pull request.
    - [x] Keep the release workflow aligned with the same canonical version assumptions.
- [x] Task: Update backlog and changelog references.
    - [x] Record the versioning policy in `todo.md` as completed track work.
    - [x] Add a concise changelog note for the version synchronization policy and validator.
- [x] Task: Conductor - Automated Review and Checkpoint 'Wire Validation Into CI And Release Docs' (Protocol in workflow.md)
