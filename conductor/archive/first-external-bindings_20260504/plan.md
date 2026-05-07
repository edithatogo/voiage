# Track Implementation Plan: First External Bindings Release Matrix

## Phase 1: Define The Binding Release Matrix [checkpoint: complete]

- [x] Task: Write the binding release matrix document for the first external
  bindings.
  - [x] Record the package manager / registry target for Python, R, Julia,
    TypeScript, Go, Rust, and .NET.
  - [x] Record the release channel for each language, including early
    distribution channels where mature registries are not yet appropriate.
  - [x] Record the .NET target framework requirement as `net11.0`.
- [x] Task: Specify which languages are in scope for the initial release
  contract and which remain deferred.
- [x] Task: Specify the minimum package metadata each binding must publish.
- [x] Task: Conductor - Automated Review and Checkpoint 'Binding Release
  Matrix' (Protocol in workflow.md)

## Phase 2: Define CI/CD Gates And Dry-Run Validation [checkpoint: complete]

- [x] Task: Specify the binding CI/CD gate set.
  - [x] Build
  - [x] Lint / format
  - [x] Type or static analysis where applicable
  - [x] Unit tests
  - [x] Docs checks
  - [x] Shared conformance-fixture validation
  - [x] Package dry-run validation on pull requests
- [x] Task: Define how each binding should invoke the shared conformance
  fixtures before release.
- [x] Task: Define how local and CI dry runs should fail when package metadata or
  registry targets are invalid.
- [x] Task: Conductor - Automated Review and Checkpoint 'Binding CI/CD Gates'
  (Protocol in workflow.md)

## Phase 3: Define Trusted Publishing And Release Provenance [checkpoint: complete]

- [x] Task: Specify the trusted publishing or token-scoped release trigger for
  each language.
  - [x] PyPI / TestPyPI / Conda-forge
  - [x] CRAN / r-universe / GitHub Releases
  - [x] Julia General registry
  - [x] npm
  - [x] Go module proxy / GitHub Releases
  - [x] crates.io
  - [x] NuGet
- [x] Task: Define release-note and changelog generation expectations.
- [x] Task: Define rollback guidance and provenance fields for released
  artifacts.
- [x] Task: Conductor - Automated Review and Checkpoint 'Trusted Publishing And
  Release Provenance' (Protocol in workflow.md)

## Phase 4: Document Handoff To Binding-Specific Tracks [checkpoint: complete]

- [x] Task: Add roadmap and todo references to the new binding release matrix
  track.
- [x] Task: Add a short note in the binding-release contract about how future
  language-specific implementation tracks should inherit the matrix.
- [x] Task: Validate the track artifacts, registry entry, and cross-links.
- [x] Task: Conductor - Automated Review and Checkpoint 'Binding Release Track
  Handoff' (Protocol in workflow.md)

## Execution Notes

- This track defines the release contract, not binding code.
- Keep the release model broad enough for future binding-specific tracks to
  inherit without narrowing it prematurely.
- Do not invent package-manager policies that conflict with the roadmap.
- The release automation is now implemented; this track is kept as the
  canonical contract for release targets and the remaining external registry
  dependencies (conda-forge feedstock approval, CRAN/r-universe, and the Julia
  General registry).
