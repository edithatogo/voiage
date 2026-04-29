# Track Implementation Plan: Cross-Language Conformance Fixtures

## Phase 1: Create the Fixture Catalog and Layout [checkpoint: ]

- [ ] Task: Create the versioned fixture layout under `specs/core-api/fixtures/v1/`.
- [ ] Task: Define the manifest format that enumerates every normative fixture, its method family, its expected output artifact, and its tolerance policy.
- [ ] Task: Create the deterministic input fixtures for the stable v1 core scope.
  - [ ] EVPI
  - [ ] EVPPI
  - [ ] EVSI
  - [ ] ENBS
  - [ ] population/sample-size cases if they are stable in v1

## Phase 2: Create Canonical Outputs and Tolerance Envelopes [checkpoint: ]

- [ ] Task: Generate canonical expected outputs for each normative fixture case.
- [ ] Task: Store fixture artifacts in the agreed language-neutral formats, using Arrow/Parquet where that improves cross-language interchange and JSON where readability is more important.
- [ ] Task: Attach provenance, seed, and tolerance metadata to every normative fixture case.

## Phase 3: Define the Runner Contract [checkpoint: ]

- [ ] Task: Write the language-neutral runner guide that explains how Python, R, and Julia consumers should load fixtures, execute a case, and compare results.
- [ ] Task: Extend the runner guide with future binding CI invocation patterns for TypeScript, Go, and Rust so conformance checks can be reused before package publication.
- [ ] Task: Add a narrow repo-side smoke validator for the fixture manifest and artifact layout so the catalog cannot silently drift.
- [ ] Task: Document the CI strategy for fixture validation so later binding tracks can plug into the same contract cleanly.

## Phase 4: Package Publishing and Binding CI Requirements [checkpoint: ]

- [ ] Task: Add a release matrix that maps each target language to its package manager and publishing channel:
  - [ ] Python: PyPI, TestPyPI, Conda-forge.
  - [ ] R: CRAN when mature, r-universe or GitHub Releases for early distribution.
  - [ ] Julia: Julia General registry.
  - [ ] TypeScript: npm.
  - [ ] Go: tagged Go modules via the Go module proxy plus GitHub Releases.
  - [ ] Rust: crates.io.
  - [ ] .NET: NuGet with .NET 11 (`net11.0`) package validation.
- [ ] Task: Define the minimum CI/CD gates each binding must implement before release: build, lint/format, type/static checks where applicable, unit tests, docs checks, shared conformance fixtures, and package dry-run validation.
- [ ] Task: Define the trusted release trigger model for each binding, including version tags/releases, registry credentials or trusted publishing, generated changelog/release notes, and rollback guidance.

## Execution Notes

- Keep fixtures small and deterministic enough for CI and smaller models to inspect.
- Distinguish normative conformance fixtures from illustrative examples in the manifest.
- Treat package publishing and language-specific CI as part of the binding contract, not as post-implementation cleanup.
