# Phase 3 (`P3.2`) Installed Bounded-Extension & R11.1–R11.5 Qualification, 26 November 2026

Both `tobac-regulatory-research==0.2.0` (built from signed `tobac` `main` commit
`fe06ef5958ba0f218317a738fe8de95973300a12`, tree `40892de77d11b0c62d6769709ccbfec3a6c16459`)
and `voiage==3.1.0` (built from branch `codex/tobac-r11-contracts-20261126`, commit
`d0cd33b45d95a1fb383b37b3769885782d90fa08`, tree `902db1faf9d878e109cbbd096c72db87b7a24287`)
were compiled as non-editable wheels on CPython `3.15.2` (`macOS-27.2.1-arm64-arm-64bit-Mach-O`),
installed into an isolated scratch virtual environment on `/Volumes/PortableSSD`, and
executed from an external working directory with `python -I`.

No runtime code in `voiage/` or `rust/` was modified.

## Build, Installation & Module Resolution

- **Wheels built & verified**:
  - `tobac_regulatory_research-0.2.0-py3-none-any.whl` (SHA-256 `849eea097d3d236a9f89a803fd3105389734a23020a7ea3300c867c60260cf4f`) — includes both `tobac` and `tobac_supply` from `src/`
  - `voiage-3.1.0-cp315-abi3-macosx_11_0_arm64.whl` (SHA-256 `3262f08d477b0b2528c747bbec30a5019f3cc17d59c859b93d5d2365096b5d48`)
- **Installed module digests (inside isolated venv `site-packages`)**:
  - `voiage/__init__.py`: `552edd0f753af8f50a873aeda971201f6bf0175a252b7a27ad48489665c3873f`
  - `voiage/_core.abi3-darwin.so`: `d786fd63dcf99e9db9ae6603734360a0b04f2c04b369545ead322af3741be79c`
  - `tobac/__init__.py`: `438a1124a46dfc7a91b5424c8127a8ead3e01c0f34e45740a1238429aa54f94f`
  - `tobac/retail_case.py`: `248ec4ed391440ce8277f4e31dc30ace0586cc30ab1ac8ced524b1dae8558ff5`
  - `tobac_supply/__init__.py`: `16467a877d1de4de35636a768f89d2e62d6b8407efb0d95db031930c17052f6f`
- **`uv pip check`**: `Checked 36 packages in 1ms` — `All installed packages are compatible`.
- **`python -I` resolution check**: `voiage`, `voiage._core`, `tobac`, and `tobac_supply` all resolved inside the isolated venv `site-packages`.

## Installed `tobac` Engineering Suites

Executed from an external working directory using an empty `pytest.ini` (`--noconftest --import-mode=importlib`):

- `tests/engineering/test_voiage_installed.py`: **3 passed**, 0 failed, 0 skipped
- `tests/engineering/test_ecosystem_contracts.py`: **12 passed**, 0 failed, 0 skipped
- **Combined**: **15 passed** in `62.66s` (exit code `0`)

## Installed Bounded-Extension & R11.1–R11.5 Verification

Executed via `python -I` exercising both non-editable wheels together:

### R11.1 — Parameter Round-Trip & Fail-Closed Rejections

- `tobac.parameters.save_parameters` / `load_parameters` round-trip on `configs/regulatory.json` (SHA-256 `8fff72e501726af284d5b07bf61a40e803bb6fc41da767f49f77ecc7ec4eaaa7`) preserved equality and axis units with canonical `tobac_parameter_sha256` = `f49b7b6926940098254ab3150f1897af4c88bbb9e62b60920359875c2054a23c`.
- Unknown envelope fields raised `ValueError("Unexpected parameter envelope fields")`; unknown nested constant fields raised `ValueError("Unexpected constants parameter fields")`.
- `voiage.contracts.analysis.AnalysisSpec.canonical_json()` round-tripped identically (`contract_digest` = `c0ca6fc9e389fb0b226b8dff1066afe210cd6dc5716fe10058dcced78253ab06`) and rejected unknown fields with Pydantic `ValidationError` (`extra_forbidden`).

### R11.2 — Order Independence & Input Immutability

| Scenario | Forward EVPI | Reverse EVPI | Reversed-State EVPI | Bit-Identical |
| --- | ---: | ---: | ---: | --- |
| Regulatory `illicit=0.2` (`installed_evpi`, `expected_version="3.1.0"`) | `0.08134530468749951` | `0.08134530468749951` | — | Yes |
| Regulatory `illicit=0.8` (`installed_evpi`, `expected_version="3.1.0"`) | `0.0609396328124996` | `0.0609396328124996` | — | Yes |
| Frozen M2 Retail `cw=1.0` (`exact_evpi(..., backend="voiage")`) | `0.06181538352543689` | `0.06181538352543689` | `0.06181538352543689` | Yes |

All `values` and `prior_weights` SHA-256 digests and `regulatory.json` configuration remained strictly unchanged after execution.

### R11.3 — Exact Oracle Parity Under Disk/Network I/O Guards

With `builtins.open`, `pathlib.Path.open`, and `socket.create_connection` replaced by raising assertions during pure numerical evaluation (and `voiage.contracts.kernel._package_version` stubbed for `run_evpi` metadata lookup):

| Scenario | `tobac` Exact Oracle | Installed `voiage` | Abs Diff (`atol=1e-10`) |
| --- | ---: | ---: | ---: |
| Regulatory `illicit=0.2` | `0.08134530468749991` | `0.08134530468749951` | `4.0246e-16` |
| Regulatory `illicit=0.8` | `0.0609396328124999` | `0.0609396328124996` | `2.9837e-16` |
| Frozen M2 Retail `cw=1.0` (`exact_evpi` / `voiage`) | `0.06181538352544015` | `0.06181538352543689` | `3.2613e-15` (`3.3307e-15` vs frozen CSV) |
| Frozen M2 Retail `cw=1.0` (`EvpiKernel` / `run_evpi`) | `0.06181538352544015` | `0.061815383525438326` | `1.8249e-15` |

### R11.4 — Bounded Extension Preserves All Reference Outputs & Digests

Adding an extended scenario (`crime_weight=1.5` in `tobac.retail_case` across 20,480 states, paired with an extended `AnalysisSpec` + `NormalizedInputBundle` via `run_evpi`):

- Extended scenario `crime_weight=1.5`:
  - `tobac` exact oracle: `0.06817987982961292`
  - `tobac` `exact_evpi(..., backend="voiage")`: `0.06817987982961574` (`abs_diff = 2.8172e-15`)
  - `voiage.contracts.kernel.run_evpi`: `0.0681798798296164` (`abs_diff = 3.4833e-15`)
  - Extended `NormalizedInputBundle.content_digest`: `af0f8248f60d06cec8edf428c6cab54fa8044dd3187abeb67ff4329a520a7cfe`
  - Extended `AnalysisSpec.contract_digest()`: `c93dada4725259abcd240c33187883303c11835bba4db6afef87132c14ab53d6`
- Re-evaluating all reference scenarios after the extension confirmed strictly identical EVPIs and digests (`spec_contract_digest` = `c0ca6fc9e389fb0b226b8dff1066afe210cd6dc5716fe10058dcced78253ab06`, `bundle_content_digest` = `4f36d1d66f3fe70c069497fd7345b4a0097a57bbd4d966dafbe1ab8867c32882`, `run_evpi` = `0.061815383525438326`, `tobac` M2 = `0.06181538352543689`, `reg_02` = `0.08134530468749951`, `reg_08` = `0.0609396328124996`).

### R11.5 — Independent Provenance Layers

Controlled mutations from the `base` variant verified that each provenance layer varies independently:

1. **Input/parameter layer (`mutated_input_cw_1_5`)**: `normalized_input_digest` (`4f36d1d6...` -> `af0f8248...`) and `calculation_input_digest` (`a87d89d9...` -> `6cc0d421...`) and EVPI (`0.061815383525438326` -> `0.0681798798296164`) updated while `method_contract_version` (`1.0.0`), `selected_backend` (`numpy`), `device` (`cpu`), and `seed` (`7`) remained invariant.
2. **Model/schema layer (`mutated_model_schema_v1_1_0`)**: `method_contract_version` (`1.0.0` -> `1.1.0`) and `spec_digest` (`875e815e...` -> `6787a953...`) updated while `normalized_input_digest`, `calculation_input_digest`, `selected_backend`, `seed`, and EVPI (`0.061815383525438326`) remained invariant.
3. **Execution settings layer (`mutated_execution_settings`)**: `NumericalPolicy` (`seed=7, atol=1e-10` -> `seed=19, atol=1e-12`) and `spec_digest` (`875e815e...` -> `e4bc051f...`) updated while `normalized_input_digest`, `calculation_input_digest`, `method_contract_version`, and EVPI (`0.061815383525438326`) remained invariant.

Full machine-readable receipt: [receipt.json](./receipt.json).
