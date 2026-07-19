# Dependency and experimental-feature policy

The Python 3.14 line tracks the latest compatible stable direct dependencies and commits `uv.lock` for reproducibility. Run `uv lock --upgrade` followed by `python scripts/dependency_frontier.py . --strict` before a release or dependency PR.

Apache Arrow/Parquet is the canonical public tabular interchange. JSON Lines is an explicit human-debug format, not an automatic fallback. Polars is available for lazy, Arrow-native processing; pandas remains supported for the existing analysis API during migration.

Experimental dependencies live in the `experimental` extra. DuckDB, orjson, msgspec, JAX, NumPyro, and Scalene must remain behind stable interfaces and require numerical-equivalence or serialization-round-trip tests before promotion. Polars GPU is intentionally excluded from the Windows-native environment because its open-beta backend requires Linux or WSL2; CPU LazyFrame execution remains supported.

Python free-threading is a beta classifier, not a performance claim. It requires a dedicated CI job and deterministic equivalence evidence before becoming a release gate.
