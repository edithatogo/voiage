# Core API Fixture Runner Contract

This guide defines how downstream bindings should consume the versioned core API fixture catalog.
It is intentionally language-neutral so Python, R, Julia, and Rust bindings can
share the same canonical cases before a package is published.

## Runner Contract

A binding-side conformance runner should:

1. Load `specs/core-api/fixtures/v1/manifest.json` for schema fixtures and
   `specs/core-api/fixtures/v1/compatibility-manifest.json` for executable cases.
2. Validate the executable manifest's `version`, non-negative integer `seed`,
   exact `provenance` object, unique case IDs, supported methods, and required
   normal, edge, and invalid coverage.
3. Resolve each case's `input_artifact` beneath `compatibility/inputs/` and
   `expected_artifact` beneath `compatibility/expected/`.
4. Execute the binding's public VOI entrypoint with the fixture inputs.
5. Repeat the execution and require the result or normalized error outcome to
   be deterministic.
6. For successful cases, compare the observed result with the expected
   artifact's `result`. Object keys and non-numeric values compare exactly;
   numeric values use the artifact's optional finite, non-negative
   `absolute_tolerance` and `relative_tolerance`, which both default to zero.
7. For invalid cases, compare the normalized error `category` and `code` with
   the expected artifact's `error` object.
8. Require every observed result or normalized error to be JSON serializable
   without NaN or infinity.

The manifest-level seed is required catalog metadata. The current Python
reference runner validates it but does not initialize a random number
generator or pass it to a method; the committed cases are deterministic
without random sampling. Downstream runners must not claim seeded stochastic
equivalence from this catalog until a case and runner explicitly consume the
seed.

The manifest-level provenance identifies the reference implementation,
execution mode, and catalog. The current runner validates that metadata; it
does not require methods to emit provenance fields in their result payloads.

The canonical comparison surface is the committed fixture artifact, not a handwritten expectation embedded in a test.

## Reference Loading Pattern

Bindings should load the manifest first, then treat the input and output artifacts as opaque data files.
The recommended sequence is:

```text
compatibility manifest -> input artifact -> binding execution -> expected artifact -> comparison
```

That order keeps the contract deterministic and makes it straightforward to reuse the same fixture catalog across languages.

## Python, R, and Julia

Python, R, and Julia should consume the same manifest entries and compare the same result fields.

- Python runners can call the smoke validator directly with `uv run python scripts/validate_core_api_fixtures.py`.
- R runners should read the JSON manifest, load the input bundle, execute the binding wrapper, and compare the result with `jsonlite` or an equivalent JSON reader.
- Julia runners should follow the same manifest-first flow and compare the loaded result payload with the committed artifact.

The fixture catalog does not assume a Python-specific object model. It assumes
that the binding can read JSON, normalize public errors to the catalog's
language-neutral category and code, and compare deterministic output using the
expected artifact's tolerances.

## Retained Binding CI Patterns

Retained binding tracks should run a conformance job before publication and point it at the same catalog.

- Rust:
  - run `cargo fmt --check`
  - run `cargo clippy --locked`
  - run `cargo test --locked`
  - run `cargo package --locked --allow-dirty`
  - run the conformance job against the shared fixture manifest

The point of the guide is not to prescribe a single implementation language. It is to ensure each binding validates against the same canonical cases before registry publication.

## CI Strategy

The fixture validation strategy for this repository is split into two layers:

1. A layout check that validates the normative manifest and artifact paths.
2. An executable compatibility check that runs normal, edge, and invalid golden
   cases against the reference public API.
3. A full schema contract check that loads fixture payloads and verifies their
   contents.

The executable check should run quickly in every binding track and in repo-side CI.
The full contract check should run when the fixture payloads themselves change.

Recommended repo-side entrypoints:

- `uv run python scripts/validate_core_api_fixtures.py`
- `uv run python scripts/validate_core_api_contract.py`

Retained binding tracks should treat the fixture manifest as a shared release gate before publishing to PyPI, Conda-forge, CRAN, r-universe, Julia General, or crates.io.
