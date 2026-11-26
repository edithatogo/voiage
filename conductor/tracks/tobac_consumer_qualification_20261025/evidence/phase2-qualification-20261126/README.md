# Phase 2 Producer Support & Backend Qualification, 26 November 2026

Phase 2 (`P2.1`–`P2.4`) qualified the installed CPython 3.15.2 environment,
Rust `voiage-domain` / `voiage-numerics` crates, Arrow/Polars/Parquet
interchange paths, and optional `JaxBackend` (`jax_enable_x64=True`) parity on
branch `codex/tobac-r11-contracts-20261126` (`d0cd33b45d95a1fb383b37b3769885782d90fa08`).
No runtime code in `voiage/` or `rust/` was modified.

## P2.1 Dependency & Native Qualification

- **Python runtime**: `3.15.2` on `macOS-27.2.1-arm64-arm-64bit-Mach-O`
- **Installed `voiage`**: `3.1.0`, native extension `.venv/lib/python3.15/site-packages/voiage/_core.abi3-darwin.so` (SHA-256 `7b6b340c9b9f97be619909991ccadbdbe21b3e8a814850af8f69b461f5c35c36`)
- **Installed packages**: `numpy==2.5.4`, `scipy==1.18.1`, `polars==2.1.0`, `pyarrow==26.0.1`, `pydantic==2.14.3`, `jsonschema==4.26.0`, `scikit-learn==1.9.1`, `jax==0.12.0.1`, `jaxlib==0.12.0.1`, `pytest==9.2.0`, `hypothesis==6.170.15`
- **Rust toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09) (Homebrew)`, `cargo 1.100.0 (3acc79c0c 2026-11-05) (Homebrew)`
- **Pandas-free source check** (`.venv/bin/python scripts/check_pandas_free.py --source voiage`): passed (`violations: []`).
- **Locked Rust unit tests** (`cargo test --manifest-path rust/Cargo.toml --locked -p voiage-domain -p voiage-numerics --lib`): passed (`voiage-domain`: 0 tests, `voiage-numerics`: 29 passed, 0 failed).

### Genuine Blockers Recorded

1. `[tool.voiage.unavailable-extras.deep_learning]`: PyTorch has no CPython 3.15 wheels (`installation_available = false`; fallback is explicit `scikit-learn` metamodel).
2. `information_source_portfolio` schema rejects `probability == 0.0` (strict `> 0`), whereas `paired_draw_bundle` accepts nonnegative draw weights (`>= 0`).
3. `scripts/dependency_frontier.py` times out on SSL handshake to PyPI in this environment (no live PyPI fetches performed).
4. `tobac` default adapter targets published `voiage 3.1.0`, while experimental credal and censored APIs sit on post-3.1.0 `main`.

## P2.2 Bounded Gap Disposition

Phase 1 R11.1–R11.5 ([`tests/test_tobac_r11_contracts.py`](../../../../tests/test_tobac_r11_contracts.py))
and existing `tobac` suites verified `AnalysisSpec`, `ParameterSpec`,
`NumericalPolicy`, `RunContext`, `NormalizedInputBundle`, `run_evpi`,
`information_source_portfolio_value`, and `paired_draw_bundle` with zero runtime
defects. Per [spec.md](../../spec.md), no new runtime code or wrapper API was
added.

## P2.3 Interchange, Parity & Measured Timings

Added [`tests/test_tobac_phase2_interchange_and_backends.py`](../../../../tests/test_tobac_phase2_interchange_and_backends.py)
(4 tests, SHA-256 `d877c5033773e10d08012290cc4d45bf5ac98ec0b01de8085f27c25d861bb445`)
covering:

- Polars `DataFrame` -> `paired_draw_bundle` -> Arrow IPC & Parquet round-trip
  preserving `draw_id`, `strategy_id`, `perspective`, `subgroup_id`, weights,
  `content_digest`, zero-copy Arrow column buffer addresses, and
  `dataframe_interchange.copy.zero_copy` (`allow_copy=False` inside
  `paired_draw_bundle` and direct `from_dataframe(pa.Table, allow_copy=False)`).
- `ValueArray.from_numpy` vs `ValueArray.from_polars` parity on `evpi` (`4,096 x 4`).
- `evpi` (Rust core), `run_evpi` (typed contract envelope), and `JaxBackend`
  (`jax_enable_x64=True`) parity within `atol=1e-12` on both the `4,096 x 4`
  regulatory payoff grid and the exact `Fraction(1, 2**34)` binary-fraction
  near-tie oracle.

### Median-of-5 Timings & Max Absolute Differences (`.venv/bin/python -I`)

| Workload | `evpi` Rust core (ms) | `run_evpi` NumPy contract (ms) | `JaxBackend` x64 (ms) | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `4,096 x 4` payoffs | 0.6587 | 1.2901 | 0.7999 | `1.44e-15` |
| `20,480 x 4` payoffs | 3.5917 | 1.8260 | 0.9531 | `4.00e-15` |

- `4,096 x 4` long-format (`16,384 x 6`) `paired_draw_bundle` + Arrow IPC +
  Parquet round-trip: median `54.0338` ms (`all_ms`: `[63.494, 54.0338, 54.0886, 52.9386, 48.5523]`).
  Arrow IPC preserved `content_digest` (`de872292238a9714934791de18b0ddf7c09f60578cfebb3765f7fd73e0ea3d3b`)
  and `table.equals`; Parquet preserved `table.equals`, `canonical_json`, and
  `schema_fingerprint` (noting `CompatLevel.newest()` `string_view` variadic
  buffers repack at 16,384 rows, whereas `CompatLevel.oldest()` `large_string`
  and 256-row `string_view` preserve `content_digest` identically).

## P2.4 Verification

- `ruff check` and `ruff format --check` on `tests/test_tobac_phase2_interchange_and_backends.py`: passed.
- `.venv/bin/python -I scripts/run_installed_pytest.py -o addopts='' -v tests/test_tobac_phase2_interchange_and_backends.py tests/test_tobac_r11_contracts.py`: `10 passed in 4.17s` (4 Phase 2 + 6 R11).

Full machine-readable receipt: [receipt.json](./receipt.json).
