# voiage - Technology Stack

## Core Language
- **Python**: >=3.12 through 3.14; a non-blocking Python 3.14t lane observes free-threaded compatibility
- **Rust**: Stable toolchain for the execution core and native benchmarks

## Core Libraries
- **NumPy**: >=2.2.6,<3 - Fundamental array computing
- **SciPy**: >=1.16.3,<1.17 - Scientific computing algorithms
- **pandas**: >=1.3,<3 - Data manipulation and analysis
- **xarray**: >=0.19,<2025 - Labeled multi-dimensional arrays
- **PyArrow**: >=25,<26 - schema-bearing interchange, Parquet, and IPC
- **Polars**: >=1.42.1,<2 - Arrow-native dataframe interoperability
- **Pydantic**: >=2.13.4,<3 - Strict domain, configuration, and logging contracts

## Optional High-Performance Computing
- **JAX**: >=0.7.1,<0.8 - Optional accelerator and research integrations only;
  never a stable-core execution authority
- **Spack**: HPC source package manager for reproducible scientific stacks
- **EasyBuild**: Automated scientific software build/install framework for HPC
- **HPSF / E4S**: HPC ecosystem distribution and curated-stack targets

## Machine Learning & Statistics
- **scikit-learn**: >=1.0,<2.0 - Machine learning algorithms (regression-based EVSI)

## CLI
- **Typer**: >=0.9,<1.0 - Modern CLI framework with auto-completion

## Visualization
- **matplotlib**: >=3.11.1,<4 - 2D plotting library
- **seaborn**: >=0.13.2,<1 - Statistical data visualization

## Testing
- **pytest**: >=9.1.1,<10 - Testing framework
- **pytest-cov**: >=7.1,<8 - Coverage reporting
- **hypothesis**: >=6.157,<7 - Property-based testing

## Code Quality
- **Ruff**: >=0.15.22,<1 - Fast Python linter, formatter, and security-rule checker
- **ty**: >=0.0.61,<1.0 - Static type checking
- **BasedPyright**: >=1.39.9,<2 - Strict second-opinion static analysis
- **tox**: >=4.57,<5 - Test environment automation
- **nox**: >=2026.7.11,<2027 - Python-coded session orchestration backed by uv
- **pre-commit**: >=4.6,<5 - Git pre-commit hooks
- **Mutmut**: >=3.6,<4 - Source-bound mutation assurance
- **Scalene**: >=2.3,<3 - CPU and memory profiling evidence

## Documentation (Current)
- **Astro**: 7.1.3 — Sole documentation build platform
- **@astrojs/starlight**: 0.41.4 — Docs framework with built-in search,
  navigation and MDX support; replaces Sphinx
- **astro-polyglot**: source-pinned 0.1.0 at commit `dc3f9c0` — generates
  Starlight-native API documentation from language-native source extraction;
  npm publication remains an external gate
- **starlight-links-validator**: 0.25.2 — build-time internal-link validation
- **starlight-llms-txt**: 0.11.0 — LLM-friendly text export
- **Griffe**: >=1.14,<2 — Python API extraction for the initial polyglot lane
- **Vale**: prose linting for Markdown
- **pnpm**: 10.32.1 package manager for the Starlight/Astro site

- **cbindgen**: optional header generation for a narrow Rust C ABI edge

## Optional Dependencies
- **PyTorch**: >=2.13,<3 - Deep learning (optional, for the `deep_learning` extra)

## Binding and Release Targets
- **R**: CRAN-style package checks and GitHub Releases for source archives
- **Julia**: General registry with TagBot synchronization
- **Rust**: publishable core crates on crates.io plus GitHub Releases for the
  private FFI, PyO3, and test-support workspace artifacts

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
**Rust**: crates.io core crates plus GitHub Releases
