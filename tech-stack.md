# voiage - Technology Stack

## Runtime Languages

The supported Python runtime range is **Python 3.10-3.14**.

- **Python**: 3.10-3.14, with Python package metadata in `pyproject.toml`.
- **Rust**: native core foundations, deterministic kernels, benchmarks, and
  binding/runtime expansion work.
- **Polyglot Bindings**: R, Julia, TypeScript, Go, Rust, and .NET 11 are tracked
  as external binding targets against shared fixtures.

## Python Package Dependencies

- **Arrays and DataFrames**: NumPy, pandas, and xarray.
- **Scientific Computing**: SciPy.
- **Probabilistic and Accelerator Stack**: JAX and NumPyro.
- **Statistics and Metamodeling**: scikit-learn and statsmodels.
- **Plotting**: matplotlib and seaborn.
- **CLI**: Typer.
- **Runtime Safety and Configuration**: defusedxml plus Pydantic v2 validated
  logging configuration.
- **System Metrics**: psutil.

Use `pyproject.toml` and `uv.lock` as the authoritative dependency and version
sources. This file records the architectural intent, not a duplicate lockfile.

## Testing and Quality

- **Test Runner**: pytest through tox and targeted `uv run pytest` commands.
- **Coverage**: pytest-cov with a 90% fail-under gate.
- **Lint and Formatting**: Ruff.
- **Type Checking**: ty across the maintained package plus BasedPyright as an
  independent checker for new and progressively ratcheted contracts.
- **Security Scan**: Ruff security rules and Bandit in the lint gate.
- **Session Orchestration**: tox for CI parity and nox for uv-backed local
  sessions.
- **Environment Frontends**: `uv.lock` is canonical; Pixi provides portable
  task entrypoints that delegate to frozen uv resolution.
- **Profiling**: Scalene on deterministic workloads, with artifacts retained by
  scheduled/manual CI.
- **Dependency Updates**: Renovate.

## Documentation

- **Starlight/Astro**: authoritative documentation site under `docs/astro-site`,
  used by local validation, tox, CI, and GitHub Pages.
- **Vale**: prose linting for Markdown and reStructuredText.
- **Notebooks and Vignettes**: examples, R vignette/manual assets, and binding
  walkthrough READMEs remain part of the tutorial surface.

## HPC and Acceleration

- **CPU Parallelism**: local, process, thread, Dask, Ray, and scheduler-adapter
  contracts.
- **Integrated and Discrete GPU**: JAX paths plus Apple Metal and discrete GPU
  evidence gates.
- **TPU**: Colab/runtime evidence paths with parity and visibility checks.
- **FPGA and ASIC**: pre-silicon OSS flow evidence first; physical board,
  shuttle, and fabricated-silicon runtime evidence are future external gates.
- **HPC Distribution Targets**: Spack, EasyBuild, HPSF, and E4S readiness
  tracks, with external curation gates kept explicit.

## Release Targets

- **Python**: PyPI, TestPyPI, and conda-forge feedstock handoff.
- **R**: GitHub Releases for source archives, with CRAN/r-universe as external
  maturity or indexing targets.
- **Julia**: Julia General registry.
- **TypeScript**: npm with provenance.
- **Go**: tagged modules through the Go module proxy.
- **Rust**: crates.io.
- **.NET**: NuGet targeting .NET 11.
