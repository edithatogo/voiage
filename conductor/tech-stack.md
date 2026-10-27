# voiage - Technology Stack

## Core Language
- **Python**: 3.15.1; a non-blocking Python 3.15.1t lane observes free-threaded compatibility
- **Rust**: Stable toolchain for the execution core and native benchmarks

## Core Libraries
- **NumPy**: >=2.2.6,<3 - Fundamental array computing
- **SciPy**: >=1.18.1,<1.19 - Scientific computing algorithms
- **pandas**: >=3.1,<4 - Data manipulation and analysis
- **xarray**: >=2026.9,<2027 - Labeled multi-dimensional arrays; retained as a declared dependency until `pandas_free_io_20261026`, no longer used by the `ValueArray`/`ParameterSet` containers (v3 contract)
- **PyArrow**: >=26,<27 - schema-bearing interchange, Parquet, and IPC
- **Polars**: >=2.1,<3 - Arrow-native dataframe interoperability; the in-memory tabular model for `ValueArray`/`ParameterSet` (v3 contract)
- **Pydantic**: >=2.13.4,<3 - Strict domain, configuration, and logging contracts

## Optional High-Performance Computing
- **JAX**: >=0.12.0.1,<0.13 - Optional accelerator and research integrations only;
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

### Opt-in local acceleration preparation (31 August 2026)

Issue #1028 evaluates a named `test-acceleration` extra with pytest-testmon 2.2,
pytest-gremlins 1.9, coverage 7.16 and VCR.py 8.3. Selection stays serial and
invalidates its database on dependency, interpreter, test inventory and relevant
resource changes. Measurements must document plugin incompatibilities and
compare cold, warm and full runs. This does not replace full tox, CI coverage,
network-isolated tests or the existing mutation authority. Tool versions remain
subject to the evaluated extra and committed lock, not a promotion claim.

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
- **astro-polyglot**: source-pinned 0.1.0 at commit `054d11e` — generates
  Starlight-native API documentation from language-native source extraction;
  npm publication remains an external gate
- **starlight-links-validator**: 0.25.2 — build-time internal-link validation
- **starlight-llms-txt**: 0.11.0 — LLM-friendly text export
- **Griffe**: >=1.14,<2 — Python API extraction for the initial polyglot lane
- **Vale**: prose linting for Markdown
- **pnpm**: 10.34.4 package manager for the Starlight/Astro site

- **cbindgen**: optional header generation for a narrow Rust C ABI edge

## Optional Dependencies
- **PyTorch**: Declared unavailable on Python 3.15 by repository policy; APIs and the `deep_learning` extra name remain reserved pending qualified wheels

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

## Native Excel migration design (27 October 2026)

Phase 3 selects the MIT-licensed `python-calamine` 0.9 native Rust binding
for raw worksheet cells, then constructs Polars columns with explicit
whole-column inference and preserves heterogeneous cells where necessary.
The Polars/fastexcel 0.21 comparison changes mixed numeric cells, leading blank
areas, header types and time-only values in the frozen legacy fixtures, so it
is not selected. This named alternative follows AC3 of the pandas-free I/O track.

Writing uses Polars `write_excel` with BSD-2-Clause XlsxWriter 3.3 and a shared
workbook. The `excel` extra declares the selected reader and writer; openpyxl
remains a CI-only producer/oracle, never a runtime fallback. Engine wheels and
workbook behavior must qualify on Python 3.15.1 before acceptance. Pandas/xarray
package dependency removal remains Phase 4 work.

### 2026-10-27 native Excel whitespace compatibility

Review reproduced whitespace loss from both CSV inference and Calamine's XML
space policy. Preserve the legacy reader with a bounded streaming SAX pass
that sets `xml:space="preserve"` on text elements in an owned temporary ZIP,
then read that copy through Calamine. The source archive is unchanged; XML
entities/DTDs remain forbidden through the existing defusedxml dependency,
which is now also declared by the Excel extra. No legacy reader fallback is
introduced. Qualify whitespace-only and padded cells with the old producer
oracle and repeat resource/coverage gates.

### 2026-10-27 pandas-free packaging reconciliation

One runtime policy binds PyPI core metadata, the Conda dependency block, tox
Polars/Arrow floor/latest lanes and dependency-frontier/preview configuration.
Keep Polars >=2.1.0,<3 (qualified 3.15.1 floor, major-version ceiling) and Arrow
>=26.0.0,<27 for ingestion. Scientific core bounds remain unchanged. Remove
pandas/xarray from runtime declarations and direct tox dependencies. Actual
external-producer/oracle tests retain pandas in an unpublished PEP 735 group
`producer-tests`; uv default groups are empty and test lanes opt in explicitly.
The clean no-extras wheel closure must contain neither legacy package. This is
test tooling, not a compatibility shim, optional production engine or full
repository zero-reference acceptance. Retain Conda's immutable published source
pin; the new dependency block is a next-release handoff, requiring a matching
published migrated sdist and native Conda proof before release claims.
