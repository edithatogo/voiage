# Unified Versioning And Release Synchronization

## Overview

`voiage` already uses semantic versioning and tag-driven publishing, but the
current version fields are split across package manifests. Python's
`pyproject.toml` is the effective canonical version source today, while the
binding manifests for R, Julia, TypeScript, Rust, and .NET still need to stay
in lockstep with that release version. This track makes that policy explicit,
adds validation for drift, and documents the release contract for maintainers
and downstream consumers.

## Goals

1. Treat the Python project version in `pyproject.toml` as the canonical
   repository release version for this track.
2. Synchronize the published package version fields for the external bindings
   to the same SemVer release number.
3. Add an automated validator that fails CI if package manifests diverge from
   the canonical repo version.
4. Document the release/versioning policy clearly in the developer and release
   docs so contributors know where version changes originate.

## Functional Requirements

1. The repo must have one documented canonical version source for the current
   release line.
2. Binding package manifests must match that release version:
   - `bindings/typescript/package.json`
   - `bindings/julia/Project.toml`
   - `bindings/rust/Cargo.toml`
   - `bindings/dotnet/src/Voiage.Core/Voiage.Core.csproj`
   - `r-package/voiageR/DESCRIPTION`
3. A repository-local validator must compare the canonical version against the
   manifest versions and report clear failures when drift appears.
4. CI must run the version validator as a dedicated gate.
5. Release and developer documentation must describe:
   - the canonical version source
   - the tag/release trigger model
   - how binding package versions are expected to stay aligned

## Non-Functional Requirements

1. The validator must be deterministic and non-interactive.
2. The version policy must remain ecosystem-appropriate:
   - tag-driven release events stay in place
   - each registry still uses its native publishing mechanism
3. The implementation must not introduce a new release-time dependency on an
   external service.

## Acceptance Criteria

1. The binding manifests all report the same release version as the canonical
   repo version.
2. The version validator passes in CI and fails when a manifest drifts.
3. The versioning policy is documented in the developer guide and release docs.
4. The roadmap and backlog reflect the versioning policy as implemented work.

## Out of Scope

1. Introducing a new root `VERSION` file or changing the canonical source away
   from `pyproject.toml` in this track.
2. Changing the registry targets or package-manager publish channels.
3. Reworking the release commit/tag semantics beyond the current SemVer flow.
