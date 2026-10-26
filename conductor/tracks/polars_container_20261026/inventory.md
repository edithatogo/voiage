# Phase 1 inventory: runtime xarray references (AC1)

Baseline commit: `28fbb1453aba876c202908a38664890b75c283d4`
Inventory command: `git grep -n 'xarray' -- voiage/` and `git grep -n -E 'xr\.[A-Za-z_]+' -- voiage/`
Inventory digest: sha256 `2afb5a56f357ba0b17b3845b061ff5c3b5454cd3c3a00248bc9b337c3fd0a673`
Recorded: 2026-10-26

Every runtime reference is classified as **container**, **validation**,
**conversion**, or **dead**; dead references are deleted, never ported. Each
entry names its owning task. References in comments/docstrings that describe the
container are classified container (they are rewritten with the container);
references that are stale comments about dead paths are classified dead.

## Classification legend

- **container** — the xarray Dataset is the canonical container surface (constructed, held, sliced, compared). Owned by T2.2 (schema) or T4.2 (call sites) / T7.2 (cli, io).
- **validation** — structural guarantees implemented with xarray (dims/coords/data_vars checks). Owned by T3.2.
- **conversion** — an interop/type-check path that converts xarray to NumPy or accepts xarray as input. Owned by T4.2 (or removed where the container no longer produces xarray).
- **dead** — comment/code-path mention of xarray with no live effect; deleted, not ported.

## voiage/schema.py (container module — 36 scripted occurrences)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L8 docstring "use xarray Datasets as the canonical in-memory representation" | container | T2.2 | rewritten to Polars frame documentation |
| L19 `import xarray as xr` (module level) | container | T2.2 | removed; replaced by `import polars as pl` |
| L52-53 ValueArray docstring `dataset : xarray.Dataset` | container | T2.2 | rewritten |
| L68 `dataset: xr.Dataset` field | container | T2.2 | replaced by `frame: pl.DataFrame` |
| L72-73 isinstance check + message | validation | T3.2 | replaced by Polars type/stopgap checks |
| L75-121 `__post_init__` rename/coord/dim checks | validation | T3.2 | reimplemented over `pl.DataFrame` |
| L123-130 `from_dataset`/`to_dataset` | conversion | T2.2 | removed — v3 contract marks `from_dataset`/`to_dataset` removed |
| L132-135 `values` -> xr.DataArray | container | T2.2 | returns `np.ndarray` per AC2/v3 |
| L137-147 `numpy_values`/`jax_values` | conversion | T2.2 | unchanged behaviour via `self.values` |
| L149-162 `n_samples`/`n_strategies`/`strategy_names` | container | T2.2 | from frame columns |
| L164-166 `copy` via `from_dataset` | container | T2.2 | from frame |
| L168-173 `get_strategy_index` | container | T2.2 | unchanged (uses names) |
| L175-181 `slice_by_strategies` via `isel` | container | T2.2 | column selection on frame |
| L183-187 `__eq__` via `dataset.identical` | container | T2.2 | frame equality |
| L189-231 `from_numpy` (xr.Dataset construction) | container | T2.2 | wide frame construction |
| L233-291 `from_numpy_perspectives` (3D coords) | container | T2.2 | `pl.Array(pl.Float64, n_perspectives)` columns |
| L293-298 `perspective_names` from coords | container | T2.2 | from `pl.Array` metadata |
| L300-346 `from_jax` | container | T2.2 | via `from_numpy` |
| L349-371 ParameterSet docstring + `dataset: xr.Dataset` | container | T2.2 | rewritten to frame |
| L372-384 `__post_init__` xarray checks | validation | T3.2 | frame checks |
| L386-393 `from_dataset`/`to_dataset` | conversion | T2.2 | removed (v3) |
| L395-409 `parameters` property | conversion | T2.2 | from frame columns |
| L410-433 `n_samples`/`parameter_names`/`copy`/`subset_by_parameters` | container | T2.2 | frame surface |
| L438-442 `__eq__` | container | T2.2 | frame equality |
| L444-511 `from_numpy_or_dict` | container | T2.2 | frame construction |
| L513-571 `from_jax` | container | T2.2 | via numpy |

## voiage/metamodels.py (3 occurrences)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L8 `import xarray as xr` (module level) | conversion | T4.2 | removed; type annotation uses a Duck-type/`pl`-agnostic protocol |
| L141 `values: np.ndarray | xr.DataArray | _SparseMatrixProtocol` | conversion | T4.2 | `xr.DataArray` removed from the union; `_as_numpy` already duck-typed via `hasattr(values,"values")` |
| L361-362 `ParameterSet(dataset=xr.Dataset(train_params))` | container | T4.2 | `ParameterSet.from_numpy_or_dict(train_params)` |
| L941-945 `_array_to_parameterset` xr.Dataset | container | T4.2 | `ParameterSet.from_numpy_or_dict(param_dict)` |
| L806/817 comments "xarray DataArray" | dead | T4.2 | comments stripped (code path already duck-typed) |

## voiage/methods/adaptive.py (3 occurrences, all container)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L145-155 `_simulate_adaptive_trial_net_benefits` xr.Dataset -> NetBenefitArray | container | T4.2 | `NetBenefitArray.from_numpy(..., ["Control Only", "With Treatment"])` |
| L383-393 post-interim xr.Dataset | container | T4.2 | `from_numpy` |
| L564-574 `_simple_adaptive_modeler` xr.Dataset | container | T4.2 | `from_numpy` |

## voiage/methods/basic.py (1 occurrence + 1 dead)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L57 comment "Handle xarray or other types if necessary" | dead | T4.2 | comment deleted (path raises for non-dict) |
| L287-297 filtered ParameterSet xr.Dataset | container | T4.2 | `ParameterSet.from_numpy_or_dict(filtered_parameters)` |

## voiage/methods/calibration.py (2 occurrences, both container)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L203 import xarray | container | T4.2 | removed |
| L278-291 calibrated net-benefit xr.Dataset -> ValueArray (L278) | container | T4.2 | `ValueArray.from_numpy(np.column_stack([nb_standard, nb_treatment]), ["Standard Care", "New Treatment"])` |
| L315 import xarray, L317-325 `simple_cal_modeler` xr.Dataset (L317) | container | T4.2 | `from_numpy` |

## voiage/methods/network_nma.py (2 occurrences, both container)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L435-441 updated ParameterSet xr.Dataset | container | T4.2 | `PSASample.from_numpy_or_dict(updated_parameters)` |
| L518-531 ValueArray xr.Dataset | container | T4.2 | `NetBenefitArray.from_numpy(..., ["Standard Care", "Treatment B", "Treatment C"])` |

## voiage/methods/observational.py (2 occurrences, both container)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L360-375 `_value_array_from_net_benefits` xr.Dataset | container | T4.2 | `NetBenefitArray.from_numpy(net_benefits, strategy_names)` |
| L398-408 `simple_obs_modeler` xr.Dataset | container | T4.2 | `from_numpy` |

## voiage/methods/sample_information.py (1 occurrence, container)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L969-975 sampled ParameterSet xr.Dataset | container | T4.2 | `ParameterSet.from_numpy_or_dict(sampled_parameters)` |

## voiage/parallel/monte_carlo.py (2 occurrences, both container)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L146 import xarray | container | T4.2 | removed (no xarray remains) |
| L192-196 `_bayesian_update` xr.Dataset | container | T4.2 | `ParameterSet.from_numpy_or_dict(posterior_samples)` |

## voiage/core/io.py (1 import + 2 constructions)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L9 `import xarray as xr` | container | T7.2 | removed |
| L134-142 `read_value_array_csv` xr.Dataset | container | T7.2 | `ValueArray.from_numpy(values, final_option_names)` |
| L188-195 `read_parameter_set_csv` xr.Dataset | container | T7.2 | `ParameterSet.from_numpy_or_dict({name: values[:, i] ...})` |

## voiage/cli.py (1 import + 1 construction)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L964 import xarray, L966-974 `_read_plot_surface` xr.Dataset (3D n_wtp) | container | T7.2 | `ValueArray.from_numpy_perspectives(...)` (3D wtp surface) or explicit frame construction; keep the 3D numpy semantic |

## voiage/analysis.py + voiage/methods/basic.py dead comments

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| voiage/analysis.py L1454 comment "Handle xarray or other types if necessary" | dead | T4.2 | comment deleted (path raises) |

## voiage/governance.py (dependency declaration, not a reference)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L287 `"xarray"` in DEPENDENCY_POLICY | — (declared dependency) | `pandas_free_io_20261026` | retained; this track does not remove declared dependencies (AC8) |
| L286 `"pandas"` in DEPENDENCY_POLICY | — (declared dependency) | `pandas_free_io_20261026` | retained |

## voiage/core/memory_optimization.py (container consumer, not an xarray import)

| Ref | Class | Owner | Disposition |
|-----|-------|-------|-------------|
| L169-170 `obj.dataset.nbytes` estimate for ValueArray/ParameterSet | container | T4.2 | ported to `obj.frame` size estimate (e.g. `frame.estimated_size()` via `to_arrow().nbytes` or column nbytes) |

## voiage/widgets/voi_widgets.py

No xarray reference. Uses `ValueArray.from_numpy`/`ParameterSet.from_numpy_or_dict` — unaffected. (Spec listed it as "type checks only"; verified zero xarray references at baseline.)

## Cross-checks for AC1

- Every `import xarray` / `from xarray` under `voiage/` appears above: schema L19, metamodels L8, methods/adaptive L145/383/564, methods/basic L287, methods/calibration L203/315, methods/network_nma L435/518, methods/observational L365/398, methods/sample_information L969, parallel/monte_carlo L146, core/io L9, cli L964. ✓
- Every `xr.` call under `voiage/` appears above (construction, `.dims`, `.coords`, `.values`, `.isel`, `.assign_coords`, `.identical`, `.sizes`, `.data_vars`). ✓
- No unclassified reference remains. Dead references (comments) are deleted, not ported. ✓