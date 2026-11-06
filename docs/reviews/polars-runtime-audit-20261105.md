# Polars runtime audit — 5 November 2026

On the source candidate based at `bf74b7fd5`, the repository's pandas-free
source checker passed and a search found no direct pandas or xarray import in
`voiage/`. The 25 tests in `tests/test_pandas_free_enforcement.py` passed under
CPython 3.15.1. The check is scoped to the repository source; it does not prove
that all possible optional third-party dependencies are pandas-free.

The `producer-tests` extra retains pandas to test interoperability. The runtime
also reads a Parquet metadata key named `pandas` for compatibility; this does not
import pandas. The v3 tabular, Polars container, pandas-free I/O and translation
reference tracks retain their existing accepted evidence. Installed HPC package
qualification is deferred and must not be inferred from these checks.

The GitHub homepage has documentation, PyPI, CI, coverage, CodeQL, OpenSSF and
license links. On 5 November, GitHub and PyPI both reported v3.1.0 as published;
`bf74b7fd5` updates the source README, CITATION.cff and CodeMeta accordingly.
Protected delivery and current hosted evidence remain required for that change.

GitHub REST readback on 5 November reports the public repository homepage as
`https://edithatogo.github.io/voiage/`, its detected license as Apache-2.0,
and issues enabled. The community profile identifies the README, contribution
guide, code of conduct, pull-request template and license; its reported health
percentage is 100. This is repository metadata readback, not proof that every
linked destination or release artifact is current.
