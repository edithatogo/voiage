# Track Implementation Plan: First External Bindings Release Matrix

## Phase 1: Define The Binding Release Matrix [checkpoint: ]

- [ ] Task: Write the binding release matrix document for the first external
  bindings.
  - [ ] Record the package manager / registry target for Python, R, Julia,
    TypeScript, Go, Rust, and .NET.
  - [ ] Record the release channel for each language, including early
    distribution channels where mature registries are not yet appropriate.
  - [ ] Record the .NET target framework requirement as `net11.0`.
- [ ] Task: Specify which languages are in scope for the initial release
  contract and which remain deferred.
- [ ] Task: Specify the minimum package metadata each binding must publish.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Binding Release
  Matrix' (Protocol in workflow.md)

## Phase 2: Define CI/CD Gates And Dry-Run Validation [checkpoint: ]

- [ ] Task: Specify the binding CI/CD gate set.
  - [ ] Build
  - [ ] Lint / format
  - [ ] Type or static analysis where applicable
  - [ ] Unit tests
  - [ ] Docs checks
  - [ ] Shared conformance-fixture validation
  - [ ] Package dry-run validation on pull requests
- [ ] Task: Define how each binding should invoke the shared conformance
  fixtures before release.
- [ ] Task: Define how local and CI dry runs should fail when package metadata or
  registry targets are invalid.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Binding CI/CD Gates'
  (Protocol in workflow.md)

## Phase 3: Define Trusted Publishing And Release Provenance [checkpoint: ]

- [ ] Task: Specify the trusted publishing or token-scoped release trigger for
  each language.
  - [ ] PyPI / TestPyPI / Conda-forge
  - [ ] CRAN / r-universe / GitHub Releases
  - [ ] Julia General registry
  - [ ] npm
  - [ ] Go module proxy / GitHub Releases
  - [ ] crates.io
  - [ ] NuGet
- [ ] Task: Define release-note and changelog generation expectations.
- [ ] Task: Define rollback guidance and provenance fields for released
  artifacts.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Trusted Publishing And
  Release Provenance' (Protocol in workflow.md)

## Phase 4: Document Handoff To Binding-Specific Tracks [checkpoint: ]

- [ ] Task: Add roadmap and todo references to the new binding release matrix
  track.
- [ ] Task: Add a short note in the binding-release contract about how future
  language-specific implementation tracks should inherit the matrix.
- [ ] Task: Validate the track artifacts, registry entry, and cross-links.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Binding Release Track
  Handoff' (Protocol in workflow.md)

## Execution Notes

- This track defines the release contract, not binding code.
- Keep the release model broad enough for future binding-specific tracks to
  inherit without narrowing it prematurely.
- Do not invent package-manager policies that conflict with the roadmap.
