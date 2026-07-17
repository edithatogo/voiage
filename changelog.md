# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- Removed stale `conductor/tracks/dataset-registry-and-example-corpus_20260625/` and
  `conductor/tracks/voi-frontier-architecture-dependency-governance_20260625/`
  directories after archiving to `conductor/archive/`.
- Added `conductor/setup_state.json` to `.gitignore` as Conductor tool runtime state.

### Added
- Added discrete GPU/CUDA benchmark packet validation for CPU comparison,
  transfer and compile overhead, result parity, CPU fallback, and explicit
  Colab/cloud runner blocks; T4 visibility alone does not prove speedup.
- Added Apple Metal/MPS benchmark packet validation with CPU reference, result
  parity, CPU fallback, and explicit unavailable-hardware status; no Apple
  speedup claim is made without production evidence.
- Added CPU cluster benchmark evidence packets and validation for single-process
  references, local scheduler smoke checks, and explicit multi-node capacity
  gates; local parallel execution does not imply cluster speedup.
- Added a production-speedup evidence schema, deterministic packet validator and
  CPU/backend handoff states; accelerator promotion and HPC-native claims remain
  gated on reviewed production-scale measurements.
- Added fixture-backed adaptive learning and bandit VOI with UCB, Thompson, and epsilon-greedy policies, sequential allocation diagnostics, CLI support, and explicit stable-promotion gates.
- Added fixture-backed ambiguity and distribution-shift VOI with robust
  radius scoring, source-target drift sensitivity, scenario regret, CLI support,
  and explicit parity/open-data stable-promotion gates.
- Added a fixture-backed equity-information VOI runtime, CLI, deterministic
  contract, subgroup allocation diagnostics, and explicit parity/open-data
  promotion gates. The method remains fixture-backed pending external evidence.
- Archived the implementation-strategy comparison VOI track after completing
  its fixture-backed runtime, CLI, governance registration, coverage
  hardening, and hosted CI slice; strategy-specific evidence, parity, and
  mature approval remain gated.
- Added a fixture-backed implementation-strategy comparison VOI runtime and
  CLI for uptake, adherence, coverage, delay, scale-up cost, adoption
  uncertainty, and population-impact diagnostics. Strategy-specific evidence,
  parity, and stable promotion remain explicit gates.
- Archived the monitoring and surveillance VOI track after completing its
  fixture-backed runtime, CLI, governance registration, and hosted CI slice;
  open-data attribution, parity, and mature approval remain gated.
- Added a fixture-backed monitoring and surveillance VOI runtime and CLI for
  periodic signal detection, stopping rules, decision revision, delay, and
  false-signal diagnostics. Open-data attribution, parity, and stable
  promotion remain explicit gates.
- Archived the expert-elicitation and evidence-synthesis VOI track after
  completing its fixture-backed runtime, CLI, evidence manifest, and hosted CI
  slice; attribution, parity, and mature approval remain gated.
- Added a fixture-backed expert-elicitation and evidence-synthesis VOI runtime
  with calibration/disagreement diagnostics, CLI support, and explicit
  attribution/parity gates.
- Archived the computational VOI and model-refinement track after completing
  its fixture-backed runtime, CLI, evidence manifest, and hosted CI slice;
  profiling, parity, and mature approval remain gated.
- Added a fixture-backed computational VOI and model-refinement runtime with
  compute-budget, approximation-error, refinement-weight, and CLI diagnostics;
  profiling, parity, and mature approval remain gated.
- Archived the data-quality, measurement-error, privacy, and linkage VOI
  track after completing its fixture-backed runtime, CLI, evidence manifest,
  and hosted CI slice; open-data attribution, parity, and mature approval
  remain gated.
- Added a fixture-backed data-quality, measurement-error, privacy, and linkage
  VOI runtime with CLI support and explicit parity/open-data gates.
- Archived the causal transportability VOI track after completing its
  fixture-backed Python runtime, CLI, evidence manifest, and hosted CI slice;
  open-data attribution, parity, and mature approval remain gated.
- Added a fixture-backed causal-identification, transportability, and
  external-validity VOI runtime with CLI support and explicit open-data and
  cross-language parity gates.
- Archived the dynamic real-options VOI track after implementing its staged
  runtime and CLI and passing hosted CI; retained longitudinal-data, parity,
  and mature approval as explicit gates.
- Added a fixture-backed dynamic real-options VOI runtime and CLI supporting
  staged evidence, delay, discounting, irreversibility, policy lock-in,
  option/waiting value, robust/Pareto strategy diagnostics, and explicit
  maturity evidence. Longitudinal open-data and cross-language parity remain
  gated.
- Archived the perspective-uncertainty VOI track after completing its
  seven-perspective evidence slice and hosted CI validation; retained
  real-data attribution, parity, and mature approval as explicit gates.
- Added a seven-perspective Value of Perspective catalog covering payer,
  societal, patient, provider, regulator, equity-weighted, and custom views,
  with hash-pinned maturity evidence and an explicit real-data attribution
  gate. The method remains fixture-backed pending parity and review.
- Archived the distributional and implementation VOI promotion track after
  adding open-data evidence with provenance and passing hosted CI; retained
  cross-language parity and stable approval as explicit external gates.
- Added provenance-preserving open-data snapshots for the experimental
  distributional and implementation-adjusted VOI contracts, including source
  URLs, licenses, reproducible selection rules, hashes, and explicit proxy
  limitations. Cross-language parity and stable promotion remain gated.
- Added experimental, NumPy-only interfaces for expected information gain,
  cost-aware Bayesian experimental design, active-learning batch selection,
  and amortized EVSI summaries. Heavy Bayesian and simulation-based-inference
  backends remain optional and evidence-gated.
- Added a machine-readable frontier stable-promotion evidence matrix and
  validator. Frontier families remain fixture-backed or experimental until
  cross-language parity, documentation, migration guidance, and explicit
  promotion evidence are available.
- Completed the shared code-scanning rollout by enforcing the pinned
  organization gate after both CodeQL and OpenSSF Scorecard SARIF uploads.
- Archived all completed Conductor track records under `conductor/archive/` and
  updated the regression tests and integration documentation to read those
  historical records from their canonical archived locations.
- Updated the Python and Starlight documentation dependency lockfiles to
  patched releases for the current Dependabot advisories, including the
  critical Jupyter Server advisory, and added a pnpm esbuild override to keep
  all resolved documentation builds on the patched release.
- Updated the Starlight configuration for its current social-link and sidebar
  APIs, preserving a clean Astro check and static documentation build after
  the security upgrade.
- Added a fail-closed repository harness for workflow permissions, immutable
  GitHub Action pins, required governance files, and security workflow
  presence; wired it into tox and contributor documentation.
- Added dependency review, OpenSSF Scorecard, Zizmor workflow auditing,
  Dependabot configuration, and release artifact provenance attestations.
- Hardened existing GitHub workflows with least-privilege permissions, pinned
  action commits, and a repaired blocking CodeQL alert gate.
- Added a pre-silicon FPGA/ASIC evidence harness with a deterministic
  fixed-point EVPI-style RTL kernel, CPU fixture bundle, manifest generator,
  committed probe manifests, and a GitHub Actions smoke workflow for free-runner
  evidence collection.
- Added a Colab accelerator validation notebook that can be uploaded with
  `colab-cli` to check JAX CPU/GPU/TPU device visibility, EVPI parity, and the
  explicit FPGA/ASIC placeholder-adapter contract.
- Added the captured Colab T4 GPU and v5e TPU evidence packets under the HPC
  acceleration abstraction handoff so hardware-backed validation results are
  available in the repository.
- Updated the HPC roadmap, accelerator contract, distribution contract, and
  Conductor feasibility records so they cite the actual Colab CPU/GPU/TPU
  evidence without overstating production accelerator readiness.
- Cleaned the HPC tracking records so FPGA is included in the shared
  accelerator abstraction scope, Colab packets are described as visibility and
  parity evidence rather than throughput packets, and completed spec-track
  metadata no longer conflicts with the track registry.
- Marked the historical TPU and ASIC feasibility records as superseded so the
  archived roadmap and track pages clearly point to the current hardware-
  dependent follow-up state.
- Clarified the Conductor registry so TPU feasibility is backed by compact
  Colab runtime evidence, while ASIC feasibility remains complete and the
  separate ASIC implementation lane stays open only for placeholder/runtime
  follow-up.
- Repointed the HPC accelerator docs from the old feasibility links to the
  current TPU and ASIC implementation track locations.
- Clarified that GitHub CI covers CPU-first and distributed CPU paths while
  FPGA and ASIC remain deferred hardware-backed follow-up work.
- Added explicit later-action notes to the FPGA and ASIC working notes so the
  remaining hardware-dependent work can be resumed cleanly when devices are
  available.
- Clarified the HPC distribution contract so FPGA and ASIC are documented as
  explicit placeholder schedulers rather than hidden runtime claims.
- Added `scheduler_is_placeholder` to the distributed config template so the
  generated config matches the CLI payload contract.
- Added a `scheduler_is_placeholder` flag to the distributed CLI payload so
  FPGA and ASIC placeholder backends are visible in machine-readable output.
- Clarified that the FPGA and ASIC scheduler names are placeholder adapters in
  the public performance guide and developer accelerator guide.
- Added a placeholder-adapter detection helper so unsupported FPGA/ASIC
  scheduler names can be identified programmatically.
- Updated the HPC-native roadmap wording so FPGA and ASIC are clearly marked
  as explicit adapter placeholders with real runtimes still pending.
- Added an execution-adapter discovery helper so the supported scheduler names
  are enumerated consistently across the CLI and parallel module exports.
- Aligned the FPGA and ASIC Conductor plans and registry notes with the
  explicit placeholder adapters and their documentation.
- Added CLI help and config-template coverage for the explicit FPGA/ASIC
  scheduler placeholders so the new adapter names stay visible to users.
- Aligned the open FPGA and ASIC Conductor tracks with the new explicit
  adapter placeholders so their remaining runtime and evidence tasks stay
  clearly open.
- Added explicit `fpga` and `asic` execution-adapter placeholders with
  deterministic `NotImplementedError` failures so the remaining accelerator
  lanes have a stable code surface without claiming unsupported runtimes.
- Clarified the remaining FPGA and ASIC track working notes so the open
  implementation lanes explicitly record their runtime/toolchain blockers.
- Archived the umbrella HPC capability implementation program and the CPU
  cluster parallelism implementation track after completing the CPU cluster,
  Metal, GPU, TPU, FPGA, and ASIC lane setup work.
- Added a new Conductor track `binding-registry-live-verification_20260511` with
  a machine-readable registry-audit snapshot so live Python, R, Julia,
  TypeScript, Go, Rust, and .NET release confirmations are tracked as explicit
  evidence.
- Added the registry evidence refresh utility
  (`scripts/refresh_binding_registry_audit.py`) with offline refresh support and
  schema fields for `checked_at` and `evidence_confidence`.
- Added a dedicated
  `docs/developer_guide/hpc_acceleration_abstraction_contract.rst` reference
  that defines the shared GPU/TPU/FPGA/ASIC abstraction decision and benchmark
  contract.
- Added the missing `tests/test_r_release_workflow.py` contract test so all
  non-Python binding releases have explicit workflow assertions.
- Added a new Conductor track `hpc-acceleration-abstraction-contract_20260511`
  to lock a shared GPU/TPU/FPGA/ASIC acceleration policy before expansion
  beyond Apple Metal.
- Updated the Apple Metal Phase-3 handoff guide to the current unified review packet
  schema produced by `compile_phase_3_handoff_packet`, including a single
  top-level `benchmarks` packet for scalar and memory and explicit `apple_metal`
  status handling.
- Updated the accelerator track registry with explicit in-progress status for
  discrete GPU, TPU, and ASIC feasibility tracks and added a Phase-3 evidence
  manifest to the Apple integrated GPU optimization track that points to existing
  CPU-reference proof artifacts.
- Added a memory/throughput benchmark helper for the Apple Metal prototype so the committed Apple workloads can be compared against the same cold/warm sample shape used by the Rust baselines.
- Added an optional Apple Metal backend prototype that routes EVPI and ENBS-style reductions through a PyTorch MPS path on Apple Silicon when the runtime supports it.
- Added a dedicated Apple Metal backend prototype track so the Apple integrated GPU benchmark step has a real device-backed implementation to measure against the committed CPU baselines.
- Added a live registry audit note and test that records the current published state for the language bindings, including the packages that still return 404 or no released versions.
- Added HPC-native roadmap baseline anchors, Apple Metal adapter strategy guidance, and Apple deployment requirements in the public guide so the first accelerator track has a concrete CPU comparison set and contract-preserving execution guidance.
- Published the HPC-native enablement roadmap with Apple-first baseline references in the developer guide and recorded the umbrella roadmap track completion.
- Added a .NET release workflow contract test and clarified the binding submission checklist so the thin adapter role and NuGet publish path are explicit.
- Added a Rust release workflow contract test and clarified the binding submission checklist so the canonical core role and crates.io publish path are explicit.
- Added a Go release workflow contract test and clarified the binding submission checklist so the thin adapter role and module-proxy indexing path are explicit.
- Added a TypeScript release workflow contract test and clarified the binding submission checklist so the thin adapter role and automated npm provenance path are explicit.
- Added a Julia release workflow contract test and clarified the binding submission checklist so the thin adapter role and the external Julia General registry boundary are explicit.
- Added an R release workflow contract test and clarified the binding submission checklist so the thin reticulate bridge role and the external CRAN/r-universe boundary are explicit.
- Added the SOTA strategy orchestration guide and archived track to codify the dependency graph, shared gates, and parallel lanes for the packaging, HPC, Rust-core, and docs strategy work.
- Marked the packaging, HPC, Rust ABI, and polyglot docs strategy tracks complete and summarized the current-state / future-state guidance in the roadmap.
- Recorded the Python cleanup-against-spec completion in the track docs, including the stable v1 compliance result, the EVPPI raw-dict compatibility alias migration note, and the deliberate follow-up boundary around xarray/JAX/Arrow/pandas cleanup.
- Added a binding submission checklist that separates in-repo publishing automation from external registry-side submission steps for Python, R, Julia, TypeScript, Go, Rust, and .NET.
- Added an explicit HPC distribution contract page and a new Conductor release program with child tracks for Python, R, Julia, TypeScript, Go, Rust, and .NET submission readiness.
- Added an HPC-native enablement roadmap with Apple integrated GPU, discrete GPU, TPU, and ASIC child tracks to stage accelerator work before claiming HPC-native status.
- Reordered the roadmap so registry deployment completion is the prerequisite for the HPC-native enablement stage.
- Added a Python release workflow contract test and clarified the checklist so the Python façade is explicitly called the stable release surface.

- Added the community support surface with `SUPPORT.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, a support-question issue template, and contributor-facing
  links from the main docs.
- Added a repository-local version synchronization policy and validator so
  `pyproject.toml` is the canonical release version source and the binding
  manifests, tox, and CI stay in lockstep.
- Reflected the implemented preference / individualized-care runtime, CLI
  surface, and fixture-backed conformance contract in the roadmap and backlog
  so the frontier status now distinguishes the live interface from the
  remaining cross-language follow-through.
- Added CLI entrypoints for the model-validation and threshold / robust
  frontier methods, plus the matching advanced-VOI and migration-guide
  examples.
- Added two new Conductor implementation tracks for the remaining frontier
  work: model-validation VOI and threshold / robust VOI.
- Marked the model-validation and threshold runtime plus fixture-backed
  conformance slices complete in the roadmap and backlog, leaving CLI
  integration as the remaining open step.

- Reconciled the Rust numerics and profiling track bookkeeping so the open work now reads as EVPPI, EVSI/frontier-adjacent kernels, and the remaining profiling measurement and guidance phases rather than stale placeholders.
- Added a Rust-core migration program to the Conductor backlog with separate tracks for the migration foundation, domain model, numerics engine, profiling/backends, and bindings/release adaptation.
- Added a new Conductor track for publication-quality R package documentation, including a PDF reference manual and a narrative vignette path.
- Added a new Conductor track for a polyglot tutorial surface so Python notebooks, R long-form docs, and binding-specific walkthroughs stay aligned.
- Added focused CLI branch coverage tests for the helper utilities, calibration/observational paths, sequential and NMA branches, adaptive and portfolio branches, and the plot command family.
- Tightened the CLI integration slice with a registry-level command-surface assertion and a backend-consistency integration test that checks the EVPI workflow against the NumPy/JAX calculation path.
- Added an engineering VOI tutorial notebook and refreshed the examples index so the tutorial set now covers financial, environmental, engineering, and JAX performance scenarios alongside the validation notebooks.
- Added the remaining validation and tutorial notebooks for network meta-analysis, structural VOI, engineering VOI, financial VOI, environmental VOI, and JAX performance, and verified the new notebooks execute.
- Expanded the Sphinx API reference coverage to include the remaining public `voiage.methods` submodules and configured autodoc mocks for optional dependencies.
- Wired the benchmark regression checks into CI and closed the pytest ignore-pattern bookkeeping in the CLI integration track.
- Added compact NMA and structural VOI validation notebooks under `examples/`,
  with runnable notebook executions and focused tests to keep the method
  validation slices aligned with the published APIs.
- Added runnable EVPI, EVPPI, EVSI, and getting-started notebooks, lightweight CLI benchmark/regression scaffolding for the core VOI methods, and a small CLI sample-input bundle.
- Added the core API extension-evolution contract and the missing user-facing method/reference pages for methods, plotting, CLI, data structures, and backends.
- Added the developer-guide architecture and how-to-contribute pages and wired them into the onboarding navigation.
- Added deterministic lifecourse EVSI and ENBS expectation payloads so the shared result-envelope fixture now covers all supported core result families.
- Added CLI property-based and fuzzing tests for the remaining invariant and malformed-input cases.
- Defined the core API method-metadata contract for capability, stability,
  and maturity labels, with an explicit approximation-status rule and
  matching schema/example coverage.
- Added a developer guide page for implementing new stable core VOI methods and linked it from the developer-guide navigation.
- Documented the core API fixture runner contract and CI strategy, including
  the smoke validator for fixture catalog layout and future binding invocation
  patterns for TypeScript, Go, and Rust.
- Defined the core API diagnostics contract for stable warning payloads,
  degraded-path reporting, and approximation caveats, with a matching
  schema/example pair and validator coverage.
- Added the core API conformance-fixture input bundle and a reusable fixture-case loader so the Phase 1 manifest now validates deterministic input/output pairs for the canonical v1 methods.
- Expanded the CLI integration suite to cover the experimental Value of Perspective command pair and aligned the help assertions with the exposed command surface.
- Clarified the polyglot release matrix and external binding track so the
  docs now distinguish automated in-repo publishing from registry-side
  dependencies such as conda-forge feedstocks, CRAN/r-universe, and the Julia
  General registry.
- Wired the polyglot release workflows so npm, crates.io, and NuGet publish on
  tag pushes, Go/Julia/R emit GitHub release archives, the Julia registry sync
  workflow is now scheduled, and the conda-forge feedstock updater now writes
  the correct release metadata.
- Expanded the supported Python matrix to 3.13 and 3.14, split the core
  JAX/NumPy/SciPy pins by Python version, and updated the local/CI runners and
  user-facing support docs to match.
- Added an optional uv-backed `nox` runner that mirrors the main tox sessions
  for local development.
- Promoted Value of Perspective to a fixture-backed frontier contract and
  aligned the runtime, schemas, fixtures, and tests with the committed
  screening-program conformance payloads.
- Expanded the frontier fixture registry and validator to include the new
  fixture-backed adjacent frontier families alongside the earlier experimental
  contract sets.
- Added deterministic fixture manifests and normative payloads for the
  adjacent frontier causal, data-quality, computational, and expert-synthesis
  contract families, and registered them in the frontier fixture registry.
- Added a shared adjacent-frontier maturity contract with maturity labels,
  fixture-backed criteria, next-step requirements, and CHEERS-VOI reporting
  conventions.
- Added an expert-synthesis frontier contract scaffold with schema and
  example payloads for elicitation costs and synthesis penalties.
- Added a computational frontier contract scaffold with schema and example
  payloads for compute budgets, approximation error, and refinement value.
- Added a data-quality frontier contract scaffold with schema and example
  payloads for acquisition costs, privacy constraints, measurement error, and
  linkage weights.
- Added a causal-transportability frontier contract scaffold with schema and
  example payloads for source-to-target population shifts, transport weights,
  and validity penalties.
- Marked the CHEERS-VOI reporting task complete after extending the shared reporting envelope to the remaining frontier families and fixture-backed contract payloads.
- Marked the dynamic real-options contract schemas, examples, fixtures, and track plan item as complete so the frontier backlog matches the committed artifacts.
- Added deterministic fixtures and a registry entry for the dynamic
  real-options frontier contract so it now participates in the shared
  frontier-fixture discovery layer.
- Added the dynamic real-options frontier contract scaffold, including
  versioned schemas, example payloads, and contract tests for staged-evidence
  and policy-lock-in decision timing.
- Marked the first-external-bindings release matrix and HEOR naming brainstorm
  items as track-backed work in the todo list now that their Conductor tracks
  exist.
- Registered the live ecosystem-module-incubation track in the Conductor
  registry so the active track list matches the actual track directories on
  disk.
- Added dedicated frontier tracks for dynamic real-options VOI and the
  remaining adjacent VOI extension families, so the roadmap backlog now maps
  to concrete track entries instead of only umbrella prose.
- Split the dynamic real-options and adjacent frontier tracks into
  family-level phases so each VOI variant now has its own contract, schema, and
  fixture path.
- Added a first-external-bindings Conductor track that defines the release
  matrix, CI/CD gates, and trusted publishing contract for Python, R, Julia,
  TypeScript, Go, Rust, and .NET bindings.
- Expanded the SOTA frontier track into explicit phases for dynamic real-
  options VOI and the remaining adjacent VOI extension families, and linked
  the roadmap phases back to the numerics, docs, CLI, and frontier track
  stubs that will implement them.
- Aligned the GitHub Actions Ruff gate with the repo's tox lint scope so the
  workflow checks `voiage` and `tests` instead of legacy root-level scratch
  scripts that are outside the supported lint surface, and corrected the Vale
  install target to the command package path used by the CI job.
- Added the planned threshold, tipping-point, and robust VOI contract
  scaffold, including versioned input/output schemas, illustrative example
  payloads, and a frontier doc note that frames the surface as
  sample-by-strategy-by-threshold-profile analysis.
- Added the planned model-validation VOI contract scaffold, including versioned
  input/output schemas, illustrative example payloads, and a frontier doc note
  that frames the surface as sample-by-strategy-by-validation-profile analysis.
- Added the planned preference heterogeneity and individualized care contract
  scaffold, including versioned input/output schemas, illustrative example
  payloads, and a frontier doc note that frames the surface as
  sample-by-strategy-by-preference-profile analysis.
- Added a SOTA VOI frontier track and research note, led by Value of
  Perspective and including planned distributional/equity VOI,
  implementation-adjusted VOI, preference-information VOI, validation VOI,
  threshold/tipping-point VOI, robust VOI, dynamic real-options VOI, and
  adjacent causal, data-quality, computational, and elicitation VOI extensions.
- Added an experimental Value of Perspective API, high-level
  `DecisionAnalysis` wrapper, CLI command, regret-matrix plot helper, and v1
  contract scaffold for comparing decisions across multiple perspectives.
- Added deterministic fixture manifests and exact input/output payloads for
  Value of Perspective, distributional/equity VOI, and implementation-adjusted
  VOI.
- Added a repository-level frontier fixture manifest validation test so the
  experimental contract directories are checked against real artifacts.
- Added a top-level frontier fixture registry manifest so the committed
  experimental contracts can be discovered from one place.
- Added a registry schema for the frontier fixture discovery layer and a test
  that validates the registry schema alongside the fixture manifests.
- Added a reusable `scripts/validate_frontier_contract.py` entrypoint for the
  frontier fixture registry and family manifests.
- Updated the top-level README and frontier research note to describe the
  registry-backed frontier contract layer.
- Wired the frontier contract validator into tox and CI so registry-backed
  frontier fixtures are checked in the normal quality gates.
- Updated the migration guide so the frontier table and roadmap context reflect
  the registry-backed experimental contracts.
- Added frontier examples to the canonical advanced VOI guide for Value of
  Perspective and distributional/equity VOI.
- Added explicit experimental API warnings to the frontier research note and
  advanced VOI guide.
- Added experimental distributional/equity VOI and implementation-adjusted VOI
  APIs, together with `DecisionAnalysis` wrappers, curated exports, and
  regression tests.
- Added deterministic experimental frontier contract files for distributional
  and implementation-adjusted VOI, including versioned schemas, README
  scaffolds, and example result payloads.
- Added shared CHEERS-VOI reporting payload helpers and attached them to the
  experimental Value of Perspective, distributional/equity VOI, and
  implementation-adjusted VOI results.
- Added the same CHEERS-VOI reporting payload helper to Value of Heterogeneity
  so the distributional frontier retains a consistent reporting baseline.
- Added the same CHEERS-VOI reporting payload helper to CEAF and dominance
  outputs so the core cost-effectiveness summary layer also carries the shared
  reporting baseline.
- Added shared CHEERS-VOI reporting payloads to the core scalar CLI outputs for
  EVPI, EVPPI, EVSI, and ENBS.
- Updated the tox coverage job to run the repository's verified pytest
  coverage command directly, so tox and the local suite now use the same
  verification path.
- Added Vale prose linting for Markdown and reStructuredText docs, with the CI
  job pointed at the real documentation paths and contributor guidance for the
  local command.
- Added ISPOR VOI and CHEERS-VOI references to the SOTA frontier track so the
  remaining frontier methods are anchored to method guidance and reporting
  standards.
- Added the HEOR module naming brainstorm track for `calibrate`, `evidence`, `process`, `report`, `registry`, `workflow`, `quality`, `engines`, and `heoml`, with PM4Py kept as an ecosystem-only process-mining capability.
- Linked the roadmap phases to their corresponding Conductor tracks so the
  remaining missing features are visible in one place.
- Refreshed the migration guide feature comparison so it reflects the current
  implemented method set, cross-language scaffolds, and roadmap context.
- Refreshed the top-level README feature matrix and roadmap summary to match
  the current implementation state and the active Phase 5/6 roadmap.
- Added the ecosystem incubation contract outline under `specs/ecosystem/`,
  including the HEOML `voiage` extension scaffold and planned fixture
  families for `lifecourse`, `innovate`, and `mars`.
- Added a compact deterministic `lifecourse` compatibility fixture bundle
  with net-benefit, parameter-sample, EVPI, and EVPPI reference payloads under
  `specs/integrations/lifecourse/v1/fixtures/normative/`.
- Added an illustrative `lifecourse` result-envelope contract with shared
  metadata fields for EVPI, EVPPI, EVSI, and ENBS payloads under
  `specs/integrations/lifecourse/v1/fixtures/illustrative/` and
  `specs/integrations/lifecourse/v1/schemas/`.
- Added explicit compatibility versioning and artifact-exchange validation
  guidance to the `lifecourse` integration docs, fixture manifest, and
  illustrative result envelope.
- Added the `specs/integrations/lifecourse/v1/` scaffold with examples,
  fixtures, and schema overlay placeholders for the future `lifecourse`
  compatibility contract.
- Added a planned `lifecourse` integration contract track and strategy document
  for consuming `lifecourse` PSA outputs through stable VOI artifacts rather
  than package internals.
- Documented the `lifecourse` integration profile as portable-artifact based,
  with pickle excluded from the shared compatibility contract.
- Identified HEOML as the candidate shared health-economic interchange profile
  for the future `lifecourse` and `voiage` handoff.
- Added an ecosystem-module incubation track and strategy for positioning
  `voiage` alongside `lifecourse`, `innovate`, `mars`, HEOML, and future
  sibling modules through optional artifact-first HEOR contracts.
- Expanded the core public API docstrings for EVPI, EVPPI, EVSI, ENBS, CEAF, CEAC, and the main schema/analysis types with NumPy-style sections and examples.
- Expanded the dominance, heterogeneity, portfolio, and dominance-plot public docs with full parameter, return, and notes sections.
- Expanded the adaptive-trial and structural-VOI public docs with full NumPy-style parameter and return sections.
- Expanded the calibration, observational, NMA, and VOI-curve public docs with fuller API-style sections and examples.
- Expanded the `DecisionAnalysis` wrapper docs so the top-level analysis methods now describe parameters, returns, and formulas in the public API surface.
- Expanded the network-meta-analysis implementation docs so the NMA entrypoints and helpers now have fuller NumPy-style sections.
- Documented the polyglot binding release matrix with tooling parity, registry/versioning expectations, and the logging policy for CLI/library output.
- Added a CLI-wide `--quiet` option that suppresses confirmation chatter while keeping the result output intact.
- Added a CLI-wide `--verbose` option that emits debug diagnostics to stderr without changing stdout result formatting.
- Added `voiage generate-config` so the CLI can emit JSON templates for common analysis inputs, starting with EVSI.
- Added working examples to the CLI help output for the result, plotting, and config-generation commands.
- Added a dedicated CLI e2e smoke-test file that exercises the full command surface through `CliRunner`.
- Added a CLI-wide `--format` option for text, JSON, and CSV output, with formatter coverage across the main result commands and plot summaries.
- Added a `--parameters-of-interest` alias for `voiage calculate-structural-evppi`, plus CLI wrappers for the existing CEAC, CEAF, VOI-curve, and dominance plotting APIs.
- Added `voiage plot-ceac`, `voiage plot-ceaf`, `voiage plot-voi-curves`, and `voiage plot-dominance` CLI commands for the existing plotting APIs.
- Added `voiage calculate-adaptive-evsi`, `voiage calculate-portfolio-voi`, and `voiage calculate-sequential-voi` as thin CLI wrappers over the adaptive, portfolio, and sequential VOI methods.
- Added `voiage calculate-enbs` with direct or file-backed EVSI input parsing and optional result-file output.
- Added curated top-level `voiage` package exports for `DecisionAnalysis`, schema types, and core methods, plus a `__version__` attribute from package metadata.
- Added `DecisionAnalysis` wrappers for CEAF, dominance, Value of Heterogeneity, and portfolio VOI so the new methods are reachable through the high-level analysis surface.
- Added exact dynamic-programming portfolio VOI optimization with budget-constrained selection, dependency-group value discounting, and regression coverage against greedy misses.
- Added Value of Heterogeneity calculation for subgroup-specific decisions, numeric subgroup binning, and subgroup VOH plotting.
- Added strong/extended dominance analysis, ICER calculation helpers, cost-effectiveness frontier extraction, and a dominance plot helper.
- Added Cost-Effectiveness Acceptability Frontier (CEAF) calculation and plotting helpers with uncertainty bands and export coverage.
- Added efficient PSA-regression and moment-based EVSI approximation methods, including a `voiage calculate-evsi` CLI command with method selection.
- Added a built-in observational study VOI modeler for explicit net-benefit or cost/effect PSA samples, including sample-size and bias-strength uncertainty adjustment.
- Added targeted branch coverage for schema, backend, structural VOI, config, financial-risk, healthcare, memory-optimization, and network meta-analysis paths, and enabled branch-aware coverage gating at 90%.
- Added initial TypeScript, Go, Rust, Julia, and .NET 11 binding package scaffolds, plus GitHub Actions CI/release workflows for npm, Go modules, crates.io, Julia package validation, NuGet, and the existing R package.
- Archived and marked complete the Python cleanup against spec track.
- Added regression coverage confirming chunked EVPPI evaluation matches the unchunked path.
- Added regression coverage for the TreeAge invalid XML fail-soft path and its warning emission.
- Added regression coverage for the curated `__all__` exports on `voiage.backends` and `voiage.methods`.
- Added regression coverage for the full versioned core API schema/example contract matrix.
- Removed an invalid EVPPI regression that assumed raw dictionary parameter samples were supported.
- Added deterministic provenance metadata validation for normative core API fixtures in the versioned manifest.
- Added normative and illustrative CEAC conformance fixtures under `specs/core-api/fixtures/v1/` and registered them in the versioned manifest.
- Added a stable core API CEAC result contract with a versioned schema, example payload, and validator coverage.
- Synchronized the core API v1 README indexes with the CEAC result schema and example contract.
- Synchronized the core API v1 schema README index with the full contract matrix.
- Seeded the illustrative EVSI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the normative and illustrative ENBS conformance fixtures under `specs/core-api/fixtures/v1/` and registered them in the versioned manifest.
- Seeded the illustrative EVPPI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the first illustrative EVPI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the first normative EVPI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Seeded the normative EVPPI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Seeded the normative EVSI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Added the versioned core API fixture scaffold under `specs/core-api/fixtures/v1/`, including the initial manifest contract and validator coverage for the normative and illustrative subtrees.

- **Structural Uncertainty VOI Methods**:
  - `structural_evpi()`: Calculate Expected Value of Perfect Information for Model Structure
  - `structural_evppi()`: Calculate Expected Value of Partial Perfect Information for Model Structure
  - `structural_evpi_jit()`: JAX-accelerated version with JIT compilation
  - `structural_evppi_jit()`: JAX-accelerated version with JIT compilation
  - CLI commands: `voiage calculate-structural-evpi` and `voiage calculate-structural-evppi`
  - JSON config file support for defining multiple model structures

- **Network Meta-Analysis VOI Methods**:
  - `NetworkMetaAnalysisData`: Data structure for NMA inputs with validation
  - `calculate_nma_evpi()`: Calculate EVPI for Network Meta-Analysis
  - `calculate_nma_evppi()`: Calculate EVPPI for Network Meta-Analysis
  - CLI command: `voiage calculate-nma-voi`
  - Support for willingness-to-pay thresholds
  - Dictionary-to-NMA data conversion for ease of use

### Changed
- Registered the 32 genuine Conductor tracks in an explicit dependency-aware
  execution order, separated 10 externally blocked tracks from automatic
  selection, and archived two superseded umbrella/duplicate tracks.
- Completed and archived the strict CI/CD quality-gates Conductor track after
  refreshing its evidence against the full tox matrix and hosted security,
  benchmark, documentation, and CI checks.
- Reconciled the Conductor registry with all 45 active track directories in
  their original dependency-aware order and added a regression test requiring
  the registry and active directory set to remain identical.
- Made Astro/Starlight the sole documentation toolchain across CI, tox, nox,
  contributor guidance, and GitHub Pages; removed the legacy Sphinx build files
  and dependencies, repaired Astro 6 content discovery, and added a harness
  guard against reintroducing Sphinx build configuration.
- 🧹 Code Health: Removed a stale commented-out example block from `voiage/exceptions.py`.
- Avoided importing `ValueArray` at runtime when it is used only for static
  annotations in the basic VOI methods.
- Updated the protected `main` ruleset for a single-maintainer repository:
  pull requests, signed commits, linear history, resolved review threads, and
  required CI/security checks remain enforced without an impossible
  independent-approval requirement. The CodeQL required-check context now
  matches the workflow's emitted `CodeQL Analysis (python)` check.
- Added Dependabot cooldowns and replaced the third-party GitHub release action
  with the runner-provided `gh release` command across package release jobs.
- Hardened the release and verification checks by making the R binding release
  checklist test resilient to current wording, aligning the minimum-version tox
  environment with the declared JAX-compatible dependency floor, and preventing
  transient intersphinx inventory fetch failures from breaking the backup Sphinx
  docs gate.
- Added explicit TinyTeX setup to the R binding CI and release workflows so the
  required PDF manual artifact can be built on GitHub-hosted Linux runners.
- Made the calibration VOI modeler optional by defaulting to the built-in calibration modeler when no custom modeler is supplied.
- Replaced the JAX two-loop EVSI placeholder/fallback path with a real JAX-assisted posterior update and resampling implementation.
- Replaced the sequential VOI step-level EVPI variance heuristic with the standard `E[max NB] - max(E[NB])` calculation for explicit strategy payoff samples.
- Replaced stale legacy type-checker references with `ty` across the active tooling, contributor docs, and Conductor infrastructure plan.
- Consolidated security linting into Ruff's selected `S` rules and removed the standalone Bandit gate from active tooling.
- Stabilized the curated package exports for `voiage.core`, `voiage.methods`, and `voiage.plot`, and added regression coverage for package-level imports.
- Stabilized the top-level `voiage` package facade so importing `voiage` now exposes the main submodules directly, with regression coverage for the export surface.
- Added focused regression coverage for NMA CLI config validation and error branches.
- Added focused regression coverage for NICE HTA scoring and decision thresholds.
- Added central pytest test categorization in `tests/conftest.py`, automatically marking collected tests as `unit`, `integration`, or `benchmark` based on file naming conventions.
- Declared the `unit` pytest marker in project configuration to support marker-based collection and selection.
- Updated the legacy pytest section in `setup.cfg` to `tool:pytest` so modern pytest versions invoked via `tox` can parse repository configuration.
- Declared missing runtime dependencies for `psutil` and `typing_extensions` so tox-installed environments can import the shipped package successfully.
- Added `pytest-benchmark` to the tox test environment so benchmark-marked tests have their required fixture during suite execution.
- Updated `ValueArray.values` to return the underlying `xarray.DataArray` and added `numpy_values`, copy, subset, and equality helpers for schema-level interoperability.
- Updated decision-analysis and downstream method code paths to use raw NumPy access where numerical kernels require ndarray semantics.
- Fixed structural JAX EVPI aggregation, GPU backend detection/mockability, and memory-budget handling so the shipped test environment runs cleanly under `tox`.
- Added regression coverage for deterministic GPU helper paths in the backend layer, including GPU detection, memory-info reporting, batch flushing, and advanced-backend delegation.
- Normalized health-economic trial outputs to Python floats and added lightweight optional-dependency fallbacks for GAM and BART metamodels when the heavy native stacks are unavailable.
- Removed the temporary GPU-test xfail and hardened metamodel diagnostics and cross-validation edge cases so the `tox` suite completes without warning noise.
- Hardened the core analysis and clinical-trial kernels for JAX tracing, removed a NumPy alias warning, and kept the full `tox` suite green without warning output.
- Reorganized the Conductor track layout into spec-first tracks to support the planned core API, fixtures, and future language bindings.
- Clarified the EVPI/EVPPI validation notebook and marked the benchmark notebook TODO complete.
- Expanded regression coverage across deterministic public modules, including backend helpers, ecosystem-import/export paths, health-economics utilities, CLI flows, schema validation, and the fluent API.

- Migrated from pip/tox to uv for 10-100x faster dependency resolution
- Expanded Ruff configuration with comprehensive rule sets
- Standardized static type checking on ty
- Enhanced pre-commit hooks with ty, commitlint, shellcheck, vulture
- Added integration and E2E test structure with pytest markers
- Modernized CI/CD with uv caching, CodeQL, benchmark tracking
- Added Renovate configuration for automated dependency updates

### Fixed
- Moved the blocking code-scanning gate into a separate least-privilege job so
  OpenSSF can verify and publish Scorecard results without rejecting the custom
  policy step in its privileged analysis job.
- Removed tracked generated artifacts from version control, including pytest
  temp output, coverage reports, macOS metadata, legacy egg-info files, and R
  CMD check output, and added repo-local ignore rules for those generated
  paths.
- Removed tracked backup files and added repo-local ignore/test coverage so
  editor or one-off `.bak` snapshots do not re-enter version control.
- Removed root-level legacy `test_*.py` scratch files and `test_results_*`
  logs so the maintained test suite is anchored under `tests/`.
- Removed stale root-level completion and coverage reports, obsolete local-path
  mutation/demo scripts, and generated result outputs that duplicated the
  canonical roadmap, changelog, docs, and maintained test suite.
- Removed generated `*.nbconvert.ipynb` notebooks and converted remaining
  `file:///Users/.../voiage` documentation links to portable repository paths.
- Reconciled roadmap status labels so completed spec, polyglot, ecosystem,
  frontier, and Rust-core foundation phases no longer appear as planned while
  external registry, hardware, and speedup evidence gates remain explicit.
- Removed stale Qoder, repository-map, and legacy roadmap mirror documents that
  duplicated the canonical roadmap/todo files and described outdated placeholder
  or v0.1-era state.
- Removed remaining root-level Qoder quest files plus duplicate planning,
  branch-architecture, CLI-changelog, repository-structure, and outdated todo
  mirrors so project state stays anchored in `roadmap.md`, `todo.md`, and
  `changelog.md`.
- Removed legacy root JAX audit/setup, coverage-analysis, benchmark,
  verification, review, and testing-approach scaffolding that was not wired
  into the maintained package, docs, tox, or CI surfaces.
- Fixed the Taskipy `profile` command to point at the tracked profiling script
  and updated that script to use the current public `ValueArray` and
  `ParameterSet` APIs.
- Removed root-level generated CLI sample CSVs and pointed the README and CLI
  example script at the maintained `examples/cli_samples/` fixtures.
- Added the XML parser used by the ecosystem integration module to the base
  runtime dependencies and made the CLI example fail fast on command failures.
- Rewrote the stale root product, technology stack, workflow, and product
  guideline mirrors so they match the current roadmap, AGENTS workflow,
  Sphinx/Starlight documentation boundary, and >90% coverage gate.
- Fixed Sphinx formatting in the HPC developer-guide pages so the docs gate
  passes with warnings treated as errors.
- Replaced the sequential CLI's empty step stub with a pass-through progression model and covered it with focused CLI regression tests.
- Restored the E2E CLI job to install the test runner dependencies it needs
  and converted the remaining comprehensive CLI help checks to the in-process
  Typer runner so the matrix stays stable across Python versions.
- Restored the GitHub Actions tool split so the CI jobs that invoke `tox`,
  `ruff`, `bandit`, and `vulture` install the dev toolset they need, and
  cleaned up the strategy orchestration guide table so the docs site builds
  without warnings.
- Regenerated the Rust core `Cargo.lock` so the benchmark workflow can run the scalar baseline with `--locked` again.
- Switched the adaptive EVSI help assertion to the in-process Typer runner so the CLI help test stays stable across Python versions in CI.
- Cleaned up the frontier family README wording so the experimental contracts now use current fixture-backed status language instead of stale "planned" phrasing.

- Added strategy Conductor tracks for SOTA packaging/review readiness, HPC
  distribution and acceleration, Rust-core ABI and migration policy, and
  polyglot repo/docs architecture.
- Extended the roadmap with a SOTA HPC and community-review program, plus
  current-state and future-state mermaid architecture diagrams.
- Reflected the current docs/release state in the conductor setup, including
  community-review targets, HPC distribution tooling, and optional ABI
  tooling.
- Split the CI and test dependency slices so linting, e2e, coverage, and prose
  checks each run with the tools they actually need, then added unit coverage
  for the new validation and threshold CLI error paths so the repository-wide
  coverage gate stays above 90%.
- Disabled coverage enforcement for benchmark-only runs so the benchmark
  workflow and task alias exercise the benchmark suite without tripping the
  repository-wide coverage gate.
- Archived the Starlight documentation-platform track after recording the
  versioning policy, plugin baseline, migration boundary, and future
  validation gates in the conductor setup.
- Aligned the validation and threshold frontier docs, schemas, examples, and
  runtime maturity labels so the contract surface now consistently reports
  fixture-backed status across the user guide, specs, and runtime output.
- Refreshed the Conductor product, tech-stack, and roadmap setup docs so they
  reflect the Rust-core direction, the current language matrix, and the
  updated project date.
- Added a Starlight documentation-platform track and recorded the initial
  versioning and plugin baseline in the roadmap and conductor setup.

- Archived the Rust EVSI stochastic-kernel follow-on track after landing the
  seeded bootstrap kernel, the fixture-backed parity checks, and the benchmark
  baseline.
- Clarified that the Rust EVSI stochastic-kernel follow-on track is kernel-only
  and that the EVSI summary envelope already lives in the Rust core contract.
- Added a new Conductor track for the Rust EVSI stochastic-kernel follow-on,
  keeping the already-owned EVSI summary contract separate from the future
  kernel implementation.
- Archived the Rust-core numerics engine track after making the EVSI summary
  boundary explicit and documenting that the stochastic kernel remains a
  future follow-on decision outside the current Rust core surface.
- Archived the Rust-core performance and profiling track after documenting the
  handoff, parallelism, and accelerator feasibility boundaries, and updated
  the Conductor registry to point at the archive.
- Added Rust-core developer guidance for the handoff boundary, Rayon/SIMD
  parallelism policy, and GPU/TPU/custom-circuit feasibility limits, with
  links from the developer guide index and the Rust binding README.
- Added a deterministic Rust EVPPI summary envelope in the `voiage-core` crate, alongside a full round-trip test for the new partial-information contract.
- Closed the Rust profiling memory/throughput measurement checkpoint and reconciled the numerics plan around the new EVSI summary contract boundary.
- Added a deterministic Rust EVSI summary envelope and expanded the Rust profiling baseline with a machine-readable memory/throughput artifact plus CI upload for the scalar benchmark.
- Archived the completed Rust-core migration foundation, domain model, bindings/release, and SOTA VOI frontier tracks after the latest review and implementation slices landed, and reconciled the Conductor registry to the archived paths.
- Advanced the Rust-core migration with the migration foundation handoff, the Rust domain-model API boundary, deterministic summary methods in `voiage-core`, a scalar CPU benchmark baseline, and the new preference heterogeneity surface in the Python API.
- Added the first Rust-core migration slices in the `voiage-core` crate, including the domain model, deterministic EVPI/ENBS kernels, and a scalar CPU benchmark baseline, then aligned the migration, binding, and release docs with that core-first shape.
- Added a repo-level smoke layer for the polyglot binding walkthrough READMEs, linked the tutorial entry points from the top-level docs and release guidance, and archived the completed R manual/vignette and polyglot tutorial-surface tracks.
- Added the R package narrative vignette and a deterministic PDF manual build helper, then documented the non-interactive build and verification path in the release and contributor guidance.
- Reworded the roadmap, backlog, README, and polyglot release docs so the R manual/vignette track and the polyglot tutorial surface track now reflect the current doc state: package help pages, the vignette, and the manual build path are in place, while verification and non-Python walkthrough alignment remain tracked work.
- Re-scoped the roadmap and backlog so the Rust-core migration is explicit about a Rust execution core, Python façade, thin bindings/adapters, and scalar-first profiling, and aligned the release docs with that ownership model.
- Reworked the R package's Python-module cache to use a private environment and normalized EVSI prior-sample forwarding, which removed the remaining `R CMD check` warning path and let `r-package/voiageR` pass the full build-and-check flow cleanly.
- Tightened the R binding metadata and namespace, and updated the package tests so `r-package/voiageR` now passes `R CMD build` and `R CMD check --no-manual`.
- Clarified the R package's current release channel and caveats in the polyglot publishing docs so the public guidance now distinguishes GitHub Releases from the external CRAN and r-universe registry flows.
- Archived the remaining completed Conductor tracks for the first external bindings release matrix, dynamic real-options VOI, adjacent frontier extensions, and the lifecourse integration contract so the active registry now only shows live work.
- Closed out the remaining documentation and CLI integration bookkeeping by archiving the docs-developer-experience and cli-integration-testing tracks after the docs, notebook, benchmark, and branch-coverage work landed.
- Refreshed the README with the current feature matrix, a working quick-start, comparison table, documentation links, and help/value sections so the landing page matches the implemented surface.
- Restored the repository verification gate by fixing the remaining style issues in the regression tests and getting the full `ruff`, `ty`, and `pytest --cov=voiage --cov-fail-under=90` suite back to green.
- Closed the remaining docs-developer-experience onboarding checklist items and the CLI completeness bookkeeping after the backend-consistency and command-surface coverage landed.
- Added an executed advanced methods tutorial notebook covering structural VOI,
  NMA VOI, adaptive trial VOI, portfolio VOI, sequential VOI, efficient EVSI,
  CEAF, and extended dominance.
- Wired the benchmark regression checks into the main CI workflow so the committed core metrics regression suite now runs on every push and pull request.
- Archived the completed numerics diagnostics and extension-model track after adding the extension-evolution contract, and removed duplicate live track entries for the already-complete cross-language and ecosystem bookkeeping.
- Marked the Python cleanup phase for public imports, result payload shapes,
  and diagnostic behavior as complete after confirming the curated package
  exports and compatibility warning paths already match the stable contract.
- Added a deprecation warning to the lower-level EVPPI wrapper when raw dict
  parameter samples are passed, and documented the `ParameterSet` path as the
  stable contract-facing form.
- Added a developer-guide page documenting the polyglot tooling split, release versioning contract, and logging policy for contributors and maintainers.
- Aligned the Python release metadata and conda update path so the feedstock
  recipe uses the PyPI sdist hash, the changelog filename matches the on-disk
  file, and the release docs describe the live automation accurately.
- Restored the CI and tox lint path for Bandit by adding it to the dev tool
  set and wiring the security check into the local lint env, and committed the
  Vale configuration files that the prose job expects at the repository root.
- Relaxed two CLI test assertions so they accept the rich help and error
  rendering used by Typer across local and CI environments.
- Fixed the coverage-report workflow by installing tox through the locked dev
  environment and relaxed the main CLI help assertion so it no longer depends
  on a specific rich-rendered option line in GitHub Actions.
- Marked the CLI E2E suite as `e2e` in pytest collection and removed the
  coverage gate from the integration and E2E CI jobs so those jobs exercise
  their own scope instead of failing on the global unit-test threshold.
- Split the coverage gate into its own CI job so the unit matrix can collect
  coverage for Codecov without enforcing the repo-wide threshold twice.

## [0.2.0] - 2025-08-02

### Changed
- **Major Refactoring of Core API and Data Structures**:
  - Introduced a new object-oriented interface (`voiage.analysis.DecisionAnalysis`) as the primary entry point for VOI calculations (EVPI, EVPPI). This encapsulates the state of a decision problem.
  - Implemented a new computational backend system (`voiage.backends`) with NumPy as the default backend. This will allow for future extensions to support other backends like JAX or PyTorch.
  - Centralized and standardized all core data structures (e.g., `ValueArray`, `ParameterSet`, `TrialDesign`) in `voiage.schema.py`.
  - Provided backward-compatible wrappers and aliases in `voiage.core.data_structures` to ensure that existing code continues to work.
  - The functional API in `voiage.methods.basic` now uses the new `DecisionAnalysis` class, providing a consistent implementation.

## [0.1.0]

### Fixed
- Fixed GitHub Pages deployment workflow by adding required permissions (`pages: write`, `id-token: write`) and environment configuration. Removed deprecated `github_token` parameter from `deploy-pages@v4` action.

### Added
- Initial project structure and placeholder files.
- Core dependencies in `pyproject.toml`.
- Pre-commit hooks for `black`, `flake8`, and `ty`.
- `NetBenefitArray`, `PSASample`, `TrialArm`, `TrialDesign`, `PortfolioStudy`, `PortfolioSpec`, and `DynamicSpec` data structures.
- `evpi()`, `evppi()`, `evsi()`, and `enbs()` methods.
- Unit tests for `evpi()`, `evppi()`, `evsi()`, and `enbs()`.
- Initial documentation in the `docs/` directory.
- Synchronized the core API README indexes with the current schema, example, fixture, and validator layout.
- Added regression coverage for callable import resolution and schema round-trips.

### Changed
- Updated dependency management with pip-tools
- Improved code coverage reporting with Codecov
- Enhanced CI/CD pipeline with additional checks

### Fixed
- Hardened the Rust benchmark baselines so they no longer depend on timing assertions, versioned the Rust core `Cargo.lock` so locked CI builds work reliably, split the CI test stack into a lighter `ci` extra to avoid disk exhaustion, and improved the benchmark workflow to print captured Cargo output when a baseline fails in CI.
* Add a fixture-backed causal-identification and transportability VOI runtime.
### Added

- Added fixture-backed capacity- and budget-constrained VOI with runtime,
  diagnostics, CLI, frontier fixtures, and Astro documentation.
- Added fixture-backed federated and privacy-preserving VOI with secure
  summary aggregation, privacy-budget diagnostics, CLI, frontier fixtures,
  and Astro documentation.
- Added fixture-backed AI-assisted evidence-triage VOI with human-in-the-loop
  audit, reviewer-time, extraction-error, model-drift, and decision-impact
  diagnostics, plus CLI, frontier fixtures, and Astro documentation.
- Added fixture-backed explainability and transparency VOI with adoption, trust, governance, audit-cost, CLI, Astro documentation, and frontier-contract diagnostics.
- Added fixture-backed interoperability and standardization VOI with harmonization, evidence reuse, transformation-error, CLI, Astro documentation, and frontier-contract diagnostics.
- Added fixture-backed regulatory and market-access VOI with approval, reimbursement, label, pricing, coverage, delay-cost, CLI, Astro documentation, and frontier-contract diagnostics.
- Added fixture-backed replication and reproducibility VOI with replication, audit, reanalysis, credibility, CLI, Astro documentation, and frontier-contract diagnostics.
- Added fixture-backed evidence obsolescence and refresh VOI with evidence age, drift, cadence, living-review, model-refresh, CLI, Astro documentation, and frontier-contract diagnostics.
- Added fixture-backed strategic behavior and game-theoretic VOI with equilibrium, incentive, disclosure, bargaining, regret, adversarial, CLI, Astro documentation, and frontier-contract diagnostics.
- Added unified accelerator evidence packet validation and deterministic indexing for passed GPU and blocked TPU/Metal runs, preserving CPU fallback and external-gate reasons.
## Unreleased

- Added a reproducible Conda-Forge feedstock handoff recording release,
  workflow, recipe, credential, maintainer-merge, and package-index evidence.
- Archived the Conda-Forge follow-through track with feedstock merge and channel
  indexing retained as explicit external gates.
- Added R CRAN/r-universe readiness evidence with reproducible package build,
  CRAN-style check, PDF manual, and explicit manual submission/indexing gates.
- Added Julia General readiness evidence with passing `Pkg.test()`, Compat and
  TagBot/release workflow verification, and explicit registration/approval gates.
- Archived the Julia registry follow-through track with General registration
  and maintainer approval retained as external gates.
- Archived the R registry follow-through track with CRAN review and r-universe
  indexing retained as external gates.

- Added a machine-readable TPU production-scale evidence contract with CPU
  fallback, EVPI parity, compile/transfer overhead, deterministic indexing, and
  an explicit Colab/gcloud availability gate.
- Added a 13-channel external registry publication evidence manifest and
  validator that separates in-repo readiness from publication, indexing,
  approval, credentials, and maintainer gates.
- Completed the repository-owned TPU production evidence track while retaining
  authenticated Colab/gcloud allocation and reviewed speedup as external gates.
- Completed the repository-owned external registry publication evidence track;
  live publication, indexing, credentials, and maintainer approval remain
  explicitly external per channel.
