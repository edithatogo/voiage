# Core API Fixture Runner Contract

This guide defines how downstream bindings should consume the versioned core API fixture catalog.
It is intentionally language-neutral so Python, R, Julia, TypeScript, Go, Rust, and .NET bindings can
share the same canonical cases before a package is published.

## Runner Contract

A binding-side conformance runner should:

1. Load `specs/core-api/fixtures/v1/manifest.json`.
2. Resolve the normative `input_artifact` and `expected_output_artifact` paths from the manifest.
3. Read the deterministic input bundle from `normative/inputs/`.
4. Execute the binding's public VOI entrypoint with the fixture inputs.
5. Compare the observed result to the expected output artifact.
6. Apply the comparison rule from the fixture manifest:
   - `exact` means structural and numeric equality must hold exactly.
   - a later non-exact policy must map to the tolerance rules in `specs/core-api/numerical-equivalence.md`.
7. Preserve provenance and diagnostics in the emitted result payload.

The canonical comparison surface is the committed fixture artifact, not a handwritten expectation embedded in a test.

## Reference Loading Pattern

Bindings should load the manifest first, then treat the input and output artifacts as opaque data files.
The recommended sequence is:

```text
manifest -> input artifact -> binding execution -> expected output artifact -> comparison
```

That order keeps the contract deterministic and makes it straightforward to reuse the same fixture catalog across languages.

## Python, R, and Julia

Python, R, and Julia should consume the same manifest entries and compare the same result fields.

- Python runners can call the smoke validator directly with `uv run python scripts/validate_core_api_fixtures.py`.
- R runners should read the JSON manifest, load the input bundle, execute the binding wrapper, and compare the result with `jsonlite` or an equivalent JSON reader.
- Julia runners should follow the same manifest-first flow and compare the loaded result payload with the committed artifact.

The fixture catalog does not assume a Python-specific object model. It only assumes that the binding can read JSON and compare deterministic output.

## Future Binding CI Patterns

Future binding tracks should add a conformance job before publication and point it at the same catalog.

- TypeScript:
  - run the binding's lint and type checks
  - run the conformance job against the fixture manifest
  - run `npm pack --dry-run`
- Go:
  - run `go test ./...`
  - run `go vet ./...`
  - run a conformance test that reads the shared fixture manifest
  - run the module packaging step that precedes release tagging
- Rust:
  - run `cargo fmt --check`
  - run `cargo clippy --locked`
  - run `cargo test --locked`
  - run `cargo package --locked --allow-dirty`
  - run the conformance job against the shared fixture manifest

The point of the guide is not to prescribe a single implementation language. It is to ensure each binding validates against the same canonical cases before registry publication.

## CI Strategy

The fixture validation strategy for this repository is split into two layers:

1. A smoke check that validates the manifest structure and artifact layout.
2. A full contract check that loads fixture payloads and verifies their contents.

The smoke check should run quickly in every binding track and in repo-side CI.
The full contract check should run when the fixture payloads themselves change.

Recommended repo-side entrypoints:

- `uv run python scripts/validate_core_api_fixtures.py`
- `uv run python scripts/validate_core_api_contract.py`

Later binding tracks should treat the fixture manifest as a shared release gate and run the same catalog before publishing to PyPI, Conda-forge, CRAN, r-universe, Julia General, npm, the Go module ecosystem, crates.io, and NuGet.
