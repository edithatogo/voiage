# voiage - Technology Stack

## Core Language
- **Python**: >=3.14; a non-blocking Python 3.14t lane observes free-threaded compatibility
- **Rust**: Stable toolchain for the execution core and native benchmarks

## Core Libraries
- **NumPy**: >=2.5.1,<3 - Fundamental array computing
- **SciPy**: >=1.18,<2 - Scientific computing algorithms
- **pandas**: >=3.0.3,<4 - Data manipulation and analysis
- **xarray**: >=2026.7,<2027 - Labeled multi-dimensional arrays
- **PyArrow**: >=25,<26 - schema-bearing interchange, Parquet, and IPC
- **Polars**: >=1.42.1,<2 - Arrow-native dataframe interoperability

## High-Performance Computing
- **JAX**: >=0.4,<0.5 - High-performance numerical computing with GPU/TPU support
- **NumPyro**: >=0.13,<0.20 - Probabilistic programming with JAX
- **Spack**: HPC source package manager for reproducible scientific stacks
- **EasyBuild**: Automated scientific software build/install framework for HPC
- **HPSF / E4S**: HPC ecosystem distribution and curated-stack targets

## Machine Learning & Statistics
- **scikit-learn**: >=1.0,<2.0 - Machine learning algorithms (regression-based EVSI)
- **statsmodels**: >=0.13,<1.0 - Statistical models and tests

## CLI
- **Typer**: >=0.9,<1.0 - Modern CLI framework with auto-completion

## Visualization
- **matplotlib**: >=3.4,<4.0 - 2D plotting library
- **seaborn**: >=0.11,<1.0 - Statistical data visualization

## Testing
- **pytest**: >=7.0,<9.0 - Testing framework
- **pytest-cov**: >=3.0,<6.0 - Coverage reporting
- **hypothesis**: >=6.0,<7.0 - Property-based testing

## Code Quality
- **Ruff**: >=0.1.9,<1.0 - Fast Python linter, formatter, and security-rule checker
- **ty**: >=0.0.1,<1.0 - Static type checking
- **tox**: >=4.0,<5.0 - Test environment automation
- **nox**: >=2024.0 - Python-coded session orchestration backed by uv
- **pre-commit**: >=3.0,<4.0 - Git pre-commit hooks

## Documentation (Current)
- **Starlight**: >=0.32.0 — Primary documentation framework (Astro-based, MDX)
- **@astrojs/starlight**: Docs framework with built-in search (Pagefind),
  i18n, and MDX support — **replaces Sphinx**
- **starlight-versions**: >=0.4.0 — Versioned documentation navigation
- **starlight-links-validator**: >=0.14.0 — Broken-link validation in CI
- **starlight-llms-txt**: >=0.5.0 — LLM-friendly text export
- **starlight-polyglot**: Auto-generated multi-language API reference from
  Python docstrings and TypeScript type definitions
- **Vale**: prose linting for Markdown
- **pnpm**: Package manager for the Starlight/Astro site

- **cbindgen**: optional header generation for a narrow Rust C ABI edge
- **WASM / N-API**: conditional TypeScript interop options if a native JS edge
  becomes necessary

## Optional Dependencies
- **PyTorch**: >=1.9,<3.0 - Deep learning (optional, for deep_learning extra)

## Binding and Release Targets
- **R**: CRAN-style package checks and GitHub Releases for source archives
- **Julia**: General registry with TagBot synchronization
- **TypeScript**: npm with provenance
- **Go**: tagged modules via the Go module proxy
- **Rust**: crates.io
- **.NET**: NuGet targeting `net11.0`

## Follow-Through Evidence Tooling
- **GitHub Actions**: repeatable release, registry-audit, benchmark, pre-silicon,
  and evidence-artifact workflows
- **GitHub CLI (`gh`)**: workflow monitoring, PR tracking, issue/release
  evidence, and artifact retrieval
- **Colab CLI (`colab`)**: best-effort GPU/TPU notebook execution and evidence
  JSON capture when free runtimes are available
- **Google Cloud CLI (`gcloud`)**: optional TPU/GPU or Cloud Shell workflows only
  when project, quota, billing, and authentication are available
- **Browser / Chrome automation**: external registry, curation, or hardware
  portal workflows only; pause before login-bound irreversible submissions,
  account actions, or paid resource use
