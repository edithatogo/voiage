# Polyglot Binding Release Matrix

The machine-readable source of truth for the retained v1 surfaces is
[`specs/v1/binding-matrix.json`](../../specs/v1/binding-matrix.json). This
document explains the release and external-gate implications of that matrix;
the drift test is `tests/test_binding_matrix.py`.

Every language binding must have CI before it can be published. Pull requests
must run build, lint/format where applicable, unit tests, conformance checks, and
package dry-run validation. Version tags trigger registry-specific publishing.
The release automation is split between in-repo publish jobs and external
registry steps: PyPI/TestPyPI, npm, crates.io, NuGet, and GitHub release
artifacts are automated from tag or release pushes, while conda-forge,
CRAN/r-universe, and the Julia General registry still depend on their external
registry or feedstock flows. The Rust-core migration has already established
the long-term ownership model: Rust is the authoritative execution core,
Python is the primary façade, and the other language packages are thin
bindings/adapters over the same canonical contract. The R package currently
ships as a reticulate bridge to the Python façade; its in-repo release
path stops at GitHub Release source archives, while CRAN and r-universe remain
external registry targets that require their own approval or indexing steps.

| Language | Package root | Registry | Tag pattern | Required CI gates |
| --- | --- | --- | --- | --- |
| Python | repository root | PyPI, TestPyPI, conda-forge feedstock | `v*` | `tox`, Ruff, ty, pytest coverage, docs; serves as the primary façade over the canonical Rust core |
| R | `r-package/voiageR` | GitHub Releases for source archives now, CRAN when mature, r-universe for early distribution | `r-v*` | `R CMD build`, `R CMD check --as-cran --no-manual`, `tools/build-manual.R`; currently a reticulate bridge to the Python façade, with the narrative vignette and deterministic PDF manual built from the same package tree |
| Julia | `bindings/julia` | Julia General registry, GitHub Releases for tag sync | `julia-v*` | `Pkg.test`, release tarball, TagBot sync |
| TypeScript | `bindings/typescript` | npm | `typescript-v*` | `npm run check`, `npm pack --dry-run`, provenance publish |
| Go | `bindings/go` | Go module proxy via semver tags, GitHub Releases | `bindings/go/v*` | `go test`, `go vet`, release tarball |
| Rust | `bindings/rust` | crates.io | `rust-v*` | `cargo fmt`, `cargo clippy`, `cargo test --locked`, `cargo doc --no-deps`, `cargo package --locked --allow-dirty`, `cargo publish`; canonical execution core and contract owner |
| .NET | `bindings/dotnet` | NuGet | `dotnet-v*` | `dotnet build`, console tests, `dotnet pack`, `dotnet nuget push` targeting `net11.0` |

## Tooling Parity

The Python façade keeps its existing development tooling:

- `scalene` for profiling
- `mutmut` for mutation testing
- `pytest-benchmark` for performance regression checks
- `ruff` and `ty` for linting and static analysis

The non-Python bindings do not mirror those tools one-for-one. Their equivalent
quality gates are language-native build, test, lint, packaging, and conformance
checks enforced in CI:

- TypeScript: `npm run check`, `npm pack --dry-run`
- Go: `go test`, `go vet`
- Rust: `cargo fmt`, `cargo clippy`, `cargo test`, `cargo package`
- Julia: `Pkg.test`
- .NET 11: `dotnet build`, `dotnet test`, `dotnet pack`
- R: `R CMD build`, `R CMD check --as-cran --no-manual`, `Rscript tools/build-manual.R`

The tutorial/documentation tracks are separate from release automation:

- R documentation work covers the package help pages, a narrative vignette, and a deterministic PDF manual built from the package tree with `tools/build-manual.R`.
- The polyglot tutorial surface keeps the Python notebooks, the R vignette, and the other binding walkthroughs aligned to the same canonical use cases.

Tutorial entry points:

- [Python notebook tutorials](../examples/index.rst)
- [R vignette and manual source](../../r-package/voiageR/vignettes/voiageR-getting-started.Rmd)
- [Julia walkthrough](../../bindings/julia/README.md)
- [Go walkthrough](../../bindings/go/README.md)
- [Rust walkthrough](../../bindings/rust/README.md)
- [TypeScript walkthrough](../../bindings/typescript/README.md)
- [.NET walkthrough](../../bindings/dotnet/README.md)

## Versioning and Logging

Each binding is versioned with the registry expectations of its ecosystem:

- Python uses repository tags plus the release metadata in `pyproject.toml`
  for PyPI/TestPyPI publication. The Python façade forwards to the Rust core
  contract rather than owning the execution engine. The separate conda-update
  workflow updates the in-repo conda-forge recipe, but the feedstock PR and
  merge still depend on the external conda-forge process.
- TypeScript uses npm semver tags with trusted provenance publication.
- Go uses semver module tags under `bindings/go/v*` and GitHub release source
  archives.
- Rust uses crates.io-compatible semver tags and `cargo publish`, and is the
  canonical engine for the voiage domain model. Python is the primary façade
  and the other language packages remain thin adapters over the same
  contract.
- Julia uses the General registry for publication and TagBot for release sync.
- .NET uses `net11.0` package releases to NuGet.
- R uses GitHub Release source archives for early distribution; CRAN is the
  primary long-term registry target, while r-universe remains an optional
  external indexing channel when the package policy is ready. The release
  artifact path also includes the non-interactive PDF manual generated from
  the same package tree, so the narrative guide and the reference manual stay
  synchronized with the shipped Rd topics.

Runtime logging is intentionally lightweight. Library calls should stay silent by
default, and CLI output should remain predictable for scripting. The CLI uses
`--quiet` to suppress status chatter and `--format json|csv` for machine-readable
results. Any future verbose logging should remain opt-in and should not break the
stable text, JSON, or CSV output contracts.

The developer-facing rationale for the tooling split, versioning contract, and
logging policy lives in `docs/developer_guide/polyglot_tooling.rst`.

The repo-level version source of truth and the manifest lockstep rule are
spelled out in `docs/developer_guide/versioning_and_release_policy.rst`, and a
CI validator enforces that policy before release automation runs.

Publishing jobs are intentionally credential-gated. Jobs that require registry
tokens fail clearly when the corresponding secret is missing. Jobs that target
registry ecosystems without direct in-repo publishing still produce validated
build artifacts and release archives, and the remaining external registry
steps are called out explicitly in the release matrix above.

For a concise per-language submission checklist, see
[`docs/release/binding-submission-checklist.md`](binding-submission-checklist.md).
For the current concrete live-registry evidence payload used by the registry
verification track, see [`docs/release/registry_audit_snapshot.json`](registry_audit_snapshot.json).
