# voiage - Product Definition

## Vision

`voiage` aims to be the premier cross-domain, high-performance Value of
Information (VOI) library for researchers and decision-makers across health
economics, clinical research, finance, environmental policy, engineering, and
adjacent uncertainty-analysis domains.

## Mission

Provide a comprehensive, open-source VOI analysis toolkit that is rigorous,
reproducible, contract-first, and usable from Python while keeping the path open
for stable polyglot bindings and native execution-core expansion.

## Target Users

- **Health Economists and HTA Agencies**: cost-effectiveness and reimbursement
  analyses.
- **Clinical Researchers**: adaptive trial design, sample-size optimization,
  and calibration-study planning.
- **Policy Analysts**: uncertainty-reduction decisions in public,
  environmental, and implementation policy.
- **Financial Analysts**: research portfolio and investment decisions under
  uncertainty.
- **Academic Researchers**: VOI teaching, method validation, and methodology
  development.

## Current Product Shape

- **Core VOI Surface**: EVPI, EVPPI, EVSI, ENBS, structural VOI, NMA VOI,
  calibration VOI, adaptive trial VOI, portfolio VOI, sequential VOI,
  CEAF/dominance, heterogeneity, and plotting are implemented in-repo.
- **Frontier Surface**: perspective, preference, validation, threshold,
  distributional/equity, implementation-adjusted, and adjacent frontier
  contracts are fixture-backed and explicitly maturity-labelled.
- **Cross-Domain Support**: healthcare, financial, environmental, engineering,
  and domain-agnostic APIs share the same core data contracts.
- **Polyglot Readiness**: R, Julia, TypeScript, Go, Rust, and .NET bindings are
  aligned around shared conformance fixtures and registry-aware release paths.
- **HPC Enablement**: CPU parallelism, scheduler adapters, Apple Metal, discrete
  GPU, TPU, FPGA, and ASIC lanes have contracts and evidence gates. Production
  speedup claims remain gated by benchmark and hardware evidence.
- **Documentation**: Astro and Starlight provide the authoritative documentation
  site, local docs gate, and GitHub Pages deployment path.

## Architecture

- **Python Reference Facade**: Python remains the primary user-facing package.
- **Contract-First Core**: schemas, deterministic fixtures, result envelopes,
  and diagnostics define compatibility before binding expansion.
- **Backend Abstraction**: NumPy, JAX, parallel/distributed adapters, and future
  native kernels must preserve public API behavior.
- **Rust-Core Boundary**: Rust foundations and deterministic kernels exist, but
  broader production-kernel migration remains evidence-gated.
- **CLI-First Workflow**: core and frontier methods expose command-line entry
  points for reproducible batch analysis.

## Completion Boundaries

- In-repo roadmap and Conductor tracks can be repository-complete while registry
  approvals, external indexing, fabricated hardware, and accelerator speedup
  evidence remain external gates.
- FPGA and ASIC support is currently pre-silicon evidence only unless real
  board, shuttle, or fabricated-silicon evidence is added later.
- Binding publication status should distinguish generated artifacts and
  automated release workflows from registry-side approval or indexing.

## Success Metrics

- Reliable >90% local test coverage with CI-enforced lint, type, docs, and test
  gates.
- Stable conformance fixtures across Python and supported external bindings.
- Clear external-gate reporting for registries, HPC distributions, and hardware
  acceleration.
- Method coverage competitive with BCEA, dampack, `voi`, and commercial VOI
  tools.
- Review-ready documentation, examples, and reproducible evidence packets.
