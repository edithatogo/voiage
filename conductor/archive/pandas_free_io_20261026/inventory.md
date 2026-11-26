# Pandas I/O baseline and next implementation boundary

T1.1 is complete. The maintainer authorised the recommended follow-up on
27 October 2026. `phase-1-baseline.json` freezes the source commit, both module
digests, 42 function signatures/return annotations, the exact `git grep`
command, its digest and 36 source hits. This is inventory evidence; T1.2 and
all runtime/package acceptance tasks remain pending.

| Surface | Existing use | Required follow-up |
|---|---|---|
| `DataFrameConnector.import_health_data` | pandas CSV, Excel and Parquet readers | Polars readers with explicit inference/null policy; preserve returned dictionary keys, shape tuple, ordering and warning behavior |
| `DataFrameConnector.export_health_data` | DataFrame construction, CSV/JSON export and multi-sheet Excel writer | Preserve treatment/state/result tables and file naming; qualify new Excel engines separately |
| `DataFrameConnector._process_dataframe` | records, columns, shape and missing-value counts | Equivalent public payload; record null versus NaN semantics before substitution |
| `_load_tabular_artifact` and `load_heoml_run_bundle` | CSV/Parquet producer ingestion and conversion | Preserve strategy/parameter ordering, units and manifest validation |
| Widget net-benefit and parameter uploads | two function-local pandas imports and numeric conversion | Polars CSV parsing; preserve labels, error messages and state transitions |
| Generated Jupyter analysis | pandas import, DataFrame construction and row iteration inside strings | Convert generated source too; compile and exercise the emitted notebook cells |
| Producer tests | externally supplied pandas frames | Retain caller compatibility through Arrow/interchange; distinguish test-only producer dependencies from package requirements |

The direct-import check currently exits 0 (matches found). Its required green
state is exit 1 (no matches). This expected red state is recorded in the JSON
baseline; it is not a passing no-pandas gate.

The next task is T1.2, beginning with failing behavioral tests for the existing
CSV/Parquet and widget upload payloads. T1.3 requires those tests to pass after
the port. Excel engines, generated snippets, dependency removal and the
repository-wide allow-list remain explicitly owned by later phases.

Container qualification is still pending. The Python 3.15.1 preflight found
that the existing JAX constraint has no matching `jaxlib` wheel and that the
declared SciPy and PyArrow ranges exclude the versions used in the isolated
qualification environment. Resolve the runtime/dependency policy before
claiming a clean supported install; do not downgrade Python or promote an
unqualified dependency set to bypass that gate.
