# CE05 Installed-Package Parity Receipt (`edithatogo/tobac` vs `voiage`)

- **Track:** `tobac_credal_endogenous_voi_20261110` (`CE05`)
- **Recorded (UTC):** `2026-11-26T02:46:03Z`
- **Status:** Passed (installed non-editable wheels on CPython `3.15.2`, `darwin-arm64`, isolated interpreter `-I`)
- **Machine-readable receipt:** [receipt.json](./receipt.json)

## Artifacts & SHA-256 Digests

| Artifact | Source Commit / Version | SHA-256 |
| --- | --- | --- |
| `voiage-3.1.0-cp315-abi3-macosx_11_0_arm64.whl` | `d0cd33b45d95a1fb383b37b3769885782d90fa08` (`codex/tobac-r11-contracts-20261126`) | `8890d790d60386014215bd7f55a35b020e88277c2f108fbcc663501727d7d5ab` |
| `tobac_regulatory_research-0.2.0-py3-none-any.whl` | `fe06ef5958ba0f218317a738fe8de95973300a12` (`tobac` `main`) | `28b501a07bba28a8ad47cea0c50c0a2ee334744f1c802772951e31d6484312e1` |
| `voiage/_core.abi3-darwin.so` (installed) | `3.1.0` | `348b68320b0ac88f6261a00ea7a3546d0e346fd3dd509a4acff1c80b99385acd` |
| `tobac/credal.py` (installed) | `0.2.0` | `bcf6b651bce5d1f2127f23c2e321e3b3cc469080005f301d2bb71afc5b946380` |
| `tobac/partial_identification.py` (installed) | `0.2.0` | `34f528557b34f050036e52a4a5e66b189de75c14e24087b6d9bc85e60aee834c` |
| `tobac/information.py` (installed) | `0.2.0` | `355e3fbef1cacdb783d96af5561bb150ef9958873abfb6a658b2a5a69bd7f5e8` |

`uv pip check` reported all 36 installed packages compatible, and `python -I` resolved `voiage`, `voiage._core`, `tobac`, and `tobac_supply` strictly inside the scratch site-packages directory.

## Verification Summary

### 1. `tobac` Property Suite (`tests/engineering/test_properties.py`)

- **Command:** `python -I -m pytest -c pytest.ini --noconftest --import-mode=importlib -v <tobac-src>/tests/engineering/test_properties.py`
- **Outcome:** Exit code `0` — `8 passed, 0 failed, 0 skipped` in `22.06s`.

### 2. Credal LP Dominance & E-Admissibility Parity

Exercised `tobac.partial_identification.restriction_bounds` and `accounting_bounds` to construct simplex parameter bounds and compared `tobac.credal.assess_decision` against `voiage.experimental.credal_decisions` across both `pyarrow.Table` and `polars.DataFrame` inputs (`max_bases=8192`).

- **`strict_dominance`:** `tobac` (`identified`, common optimal `["dominant"]`) and `voiage` (`feasible`, E-admissible `["dominant"]`, strictly dominated `["suboptimal_a", "suboptimal_b"]`) match exactly. Dual bounds match as exact `Fraction`s (`9955325492082149/4503599627370496` for `suboptimal_a`, `18962524746823141/4503599627370496` for `suboptimal_b`).
- **`e_admissibility_and_mixed_dominance`:** `tobac` (`not_identified`, common optimal `[]`) and `voiage` (`feasible`, E-admissible `["action_d", "action_p", "action_q"]` with verified exact rational primal points; `"mixed_dominated"` strictly dominated by mixture `{"action_d": "4/9", "action_p": "1/9", "action_q": "4/9"}` with Farkas margin `"5/9"`).
- **`multi_restriction_simplex`:** Simplex equality (`sum p <= 1` and `-sum p <= -1`) plus 6 `LinearRestriction` rows narrows the credal set so `action_d` is uniquely common-optimal and E-admissible, with exact `Fraction` dual margins (`1114048328875859/562949953421312` vs `action_p`, `9457559217478041/2251799813685248` vs `action_q`).
- **`infeasible_restrictions`:** `tobac` returns `incompatible` with verified rational infeasibility certificate (`multipliers=["1/2", "0", "0", "0", "1/2"]`, `-1/2 > -653318235417891/1125899906842624`) and `voiage` returns `infeasible`.

### 3. Tobit-Censored Sequential EVSI Parity

Compared `tobac.information.evaluate`, `tobac.information.scalar_oracle`, and `tobac.information.joint_likelihood` against `voiage.experimental.paired_draw_bundle` + `voiage.experimental.censored_sequential_evsi` using exact Gaussian CDF (`scipy.special.ndtr`) bin probabilities across nonuniform priors, reversed draw order, multi-perspective/multi-subgroup rows, and Polars/Arrow IPC round-tripping:

- **Max absolute error across all scenarios:** `4.440892098500626e-16` (tolerance `1e-12`).
- **Original Tobit channel (`limit=1.5`, `cutpoints=(2.5, 4.0)`):** `voiage` `4.073321328132176` vs `tobac.evaluate` / `scalar_oracle` `4.073321328132176` (`abs_err = 0.0`), invariant across forward `(10, 20, 30, 40)` and reversed `(40, 10, 30, 20)` draw order.
- **Blackwell garbling monotonicity:**
  - Degraded detection (`0.5x`): `1.6866606640660877 <= 4.073321328132176` (`abs_err = 4.44e-16`).
  - Coarsened bins (`cutpoints=()`): `1.5888357862274438 <= 4.073321328132176` (`abs_err = 2.22e-16`).
- **Zero-information degeneracy:** Both uninformative Tobit parameters and empty stage schedules return exact `Fraction(0)` (`tobac` `0.0`).
- **Two-stage joint & alternate scope:** Two-stage schedule matches `joint_likelihood` (`3.76615168477376`, `abs_err_oracle = 4.44e-16`), and `societal`/`catchment_a` scope matches (`2.5640025920489204`, `abs_err = 4.44e-16`) while preserving `draw_id`, `strategy_id`, `perspective`, and `subgroup_id` across Polars/Arrow IPC round-trips.

## Reproduction Steps

1. Export `edithatogo/tobac` `main` at commit `fe06ef5958ba0f218317a738fe8de95973300a12` into a scratch directory.
2. Build non-editable wheels for both `tobac` and `voiage` (`HEAD` `d0cd33b45d95a1fb383b37b3769885782d90fa08`) with Python `3.15.2` (`uv build --wheel`).
3. Create a fresh `uv venv`, install both wheels plus `pytest` and `hypothesis`, and verify `uv pip check` and `python -I` module resolution inside the venv.
4. Run `pytest` on `tests/engineering/test_properties.py` with `--noconftest --import-mode=importlib` and run the credal + censored EVSI parity assertions in an isolated working directory.
