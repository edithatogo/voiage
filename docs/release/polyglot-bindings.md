# Polyglot Binding Release Matrix

Every language binding must have CI before it can be published. Pull requests
must run build, lint/format where applicable, unit tests, conformance checks, and
package dry-run validation. Version tags trigger registry-specific publishing.

| Language | Package root | Registry | Tag pattern | Required CI gates |
| --- | --- | --- | --- | --- |
| Python | repository root | PyPI, TestPyPI, Conda-forge | `v*` | `tox`, Ruff, ty, pytest coverage, docs |
| R | `r-package/voiageR` | CRAN when mature, r-universe for early distribution | `r-v*` | `R CMD check --as-cran`, package build |
| Julia | `bindings/julia` | Julia General registry | `julia-v*` | `Pkg.test`, package validation |
| TypeScript | `bindings/typescript` | npm | `typescript-v*` | `npm run check`, `npm pack --dry-run` |
| Go | `bindings/go` | Go module proxy via semver tags | `go-v*` | `go test`, `go vet` |
| Rust | `bindings/rust` | crates.io | `rust-v*` | `cargo fmt`, `cargo clippy`, `cargo test`, `cargo package` |
| .NET | `bindings/dotnet` | NuGet | `dotnet-v*` | `dotnet build`, console tests, `dotnet pack` targeting `net11.0` |

## Tooling Parity

The Python reference package keeps its existing development tooling:

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
- R: `R CMD build`, `R CMD check --as-cran`

## Versioning and Logging

Each binding is versioned with the registry expectations of its ecosystem:

- Python uses repository tags plus `semantic_release` metadata in
  `pyproject.toml` for PyPI/TestPyPI/Conda-forge publication.
- TypeScript uses npm semver tags.
- Go uses semver module tags.
- Rust uses crates.io-compatible semver tags.
- Julia uses package registry release tags.
- .NET uses `net11.0` package releases to NuGet.
- R uses package release artifacts for CRAN or r-universe.

Runtime logging is intentionally lightweight. Library calls should stay silent by
default, and CLI output should remain predictable for scripting. The CLI uses
`--quiet` to suppress status chatter and `--format json|csv` for machine-readable
results. Any future verbose logging should remain opt-in and should not break the
stable text, JSON, or CSV output contracts.

Publishing jobs are intentionally credential-gated. Jobs that require registry
tokens check for the corresponding secret before publishing and otherwise still
produce validated build artifacts.
