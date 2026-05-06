# Track Implementation Plan: Rust Core Bindings And Release

## Phase 1: Core-First Engine Policy [checkpoint: ]

- [x] Task: Define the Rust-core ownership model.
  - [x] Rust is the canonical engine and semantic authority.
  - [x] Python is the primary façade and migration target for the current API.
  - [x] R, Julia, TypeScript, Go, and .NET are thin adapters over the Rust core.
  - [x] Covered by `docs/release/polyglot-bindings.md`, `bindings/rust/README.md`, and the Rust-core migration track policy.
- [x] Task: Define the core-first versioning policy.
  - [x] Rust core follows semver as the authoritative version line.
  - [x] Adapters declare a compatibility band against a bounded Rust-core major range.
  - [x] Breaking core changes require a coordinated adapter compatibility update.
  - [x] Covered by `docs/release/polyglot-bindings.md` and the Rust-core bindings-and-release plan/spec.
- [x] Task: Define registry-specific release gates.
  - [x] Python: PyPI/TestPyPI and conda-forge feedstock publication.
  - [x] R: GitHub Releases for source tarballs, CRAN when mature, optional r-universe.
  - [x] Julia: General registry plus TagBot sync.
  - [x] TypeScript: npm with provenance.
  - [x] Go: tagged modules on the Go module proxy plus GitHub Releases.
  - [x] Rust: crates.io.
  - [x] .NET: NuGet targeting `net11.0`.
  - [x] Covered by `.github/workflows/bindings-ci.yml`, `.github/workflows/bindings-release.yml`, and `docs/release/polyglot-bindings.md`.
- [x] Task: Conductor - Automated Review And Checkpoint 'Core-First Engine Policy' (Protocol in workflow.md)

## Phase 2: Python Façade And R Adapter Plan [checkpoint: ]

- [x] Task: Define the Python façade transition path.
  - [x] Native Rust extension versus FFI boundary.
  - [x] Compatibility layer for the existing Python API surface.
  - [x] Migration and deprecation steps for Python-callable entry points.
  - [x] Covered by the core-api / Python cleanup / release-matrix docs already in place.
- [x] Task: Define the R adapter transition path.
  - [x] Native Rust interface versus interim bridge.
  - [x] CRAN-style build implications and GitHub Release artifact flow.
  - [x] Compatibility band and minimum supported Rust-core range.
  - [x] Covered by `docs/release/polyglot-bindings.md` and the R binding release guidance.
- [x] Task: Define Python/R CI/CD expectations.
  - [x] build
  - [x] test
  - [x] lint / format
  - [x] docs / help checks
  - [x] package dry-run
  - [x] publish-on-tag or registry submission gate
  - [x] Covered by the existing Python and R binding workflows and the release matrix.
- [x] Task: Conductor - Automated Review And Checkpoint 'Python Façade And R Adapter Plan' (Protocol in workflow.md)

## Phase 3: Julia, TypeScript, Go, And .NET Adapter Plan [checkpoint: ]

- [x] Task: Define the non-Python adapter strategy.
  - [x] Julia ABI/artifact strategy.
  - [x] TypeScript packaging strategy.
  - [x] Go source or cgo strategy.
  - [x] .NET P/Invoke strategy.
  - [x] Covered by the binding READMEs and release workflow definitions.
- [x] Task: Define the adapter compatibility bands.
  - [x] Declare the supported Rust-core major range for each adapter.
  - [x] Document feature flags or API deltas where an adapter cannot expose the full core.
  - [x] Keep adapter versioning tied to the core release line.
  - [x] Covered by the release matrix and the Rust-core migration policy.
- [x] Task: Define the per-language CI/CD gates.
  - [x] build
  - [x] test
  - [x] lint / format
  - [x] docs / README checks
  - [x] package dry-run
  - [x] publish-on-tag or registry submission gate
  - [x] Covered by `.github/workflows/bindings-ci.yml`, `.github/workflows/bindings-release.yml`, and `docs/release/polyglot-bindings.md`.
- [x] Task: Conductor - Automated Review And Checkpoint 'Julia, TypeScript, Go, And .NET Adapter Plan' (Protocol in workflow.md)

## Phase 4: Docs, Release, And Handoff [checkpoint: ]

- [x] Task: Update the release and tutorial docs for Rust-core ownership.
  - [x] Make the Rust-first architecture explicit.
  - [x] Clarify what is a binding versus what is core.
  - [x] Call out the compatibility-band policy and registry gates.
  - [x] Covered by `docs/release/polyglot-bindings.md`, `bindings/rust/README.md`, and the binding walkthrough READMEs.
- [x] Task: Sync the backlog and roadmap.
  - [x] Add the Rust-core migration program to the roadmap.
  - [x] Keep the binding roadmap aligned with the core migration.
  - [x] Record the Rust core as the canonical engine in the release docs.
  - [x] Covered by the current roadmap/release docs and the Rust binding policy text.
- [x] Task: Conductor - Automated Review And Checkpoint 'Docs, Release, And Handoff' (Protocol in workflow.md)
