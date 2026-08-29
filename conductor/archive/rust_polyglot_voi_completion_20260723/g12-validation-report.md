# G12 focused validation and harness report

## Results

| Gate | Command | Result |
|---|---|---|
| Python conformance | `uv run pytest tests/test_perspective_conformance_v1.py tests/test_perspective_promotion_evidence.py -q` | Passed: 5 tests. |
| Source-policy regression | `uv run pytest tests/test_source_policy_cache.py -q` | Passed: 38 tests. |
| Conductor cross-references | `python scripts/validate_conductor_github_cross_references.py` | Passed. |
| Metadata parse | `python -m json.tool conductor/tracks/rust_polyglot_voi_completion_20260723/metadata.json` | Passed. |
| Rust workspace | `cargo test --manifest-path rust/Cargo.toml --workspace --all-targets` | Core, ABI, numerics, properties and benchmarks passed; the PyO3 `_core` target is blocked by missing `libpython3.13.dylib`. |
| Diff hygiene | `git diff --check` | Passed. |

## Disposition

The repository-owned focused validation and Conductor harness checks pass. The
full Rust workspace remains blocked only at the environment-dependent PyO3
loader target, consistent with G8 and G10. This is not converted into a
parity or release claim. A matching Python runtime/loader runner must rerun
the workspace and the installed R/Julia shared-fixture packet before stable
promotion or track closeout.
