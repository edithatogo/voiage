# G5 conformance and pathological-case evidence protocol

## Executed suites

The accepted Rust/Python/polyglot surface is exercised by the following
reproducible suites:

```text
uv run pytest tests/test_binding_matrix.py \
  tests/test_core_api_contract_validator.py \
  tests/test_core_api_fixture_runner_contract.py \
  tests/test_python_rust_bridge.py \
  tests/test_croissant_offline_fixtures.py \
  tests/test_frictionless_offline_fixtures.py --no-cov -q
tox -e ingestion-conformance
cargo test --manifest-path rust/Cargo.toml --workspace --all-targets
```

The focused Python contract/polyglot/ingestion slice passed 171 tests. The
ingestion tox environment and full Rust workspace also pass locally after the
runtime prerequisites were installed.

## Required evidence for future parity claims

Each method/runtime packet must retain the immutable fixture-manifest hash,
canonical output and diagnostic hashes, tolerance result, unsupported-case
diagnostic, ABI/layout result, toolchain and platform. A missing installed
runtime remains `blocked`; it cannot be represented as a passing parity case.

## Pathological and negative cases

The suites cover malformed schemas, invalid dimensions, unsupported methods,
missing native libraries, rank deficiency, non-finite values, ABI misuse and
offline Croissant/Frictionless fixtures. Warnings about explicitly
non-stable compatibility estimators are retained and do not promote those
estimators to stable scientific status.
