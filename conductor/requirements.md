# VOIAGE requirements

This repository implements the production consumer side of the VOP–VOIAGE
programme. The canonical cross-repository requirements are maintained in
`vop_poc_nz/conductor/requirements.md`.

## MoSCoW priorities

### Must have

- Directional current-information EVoP and perspective methods retain versioned,
  deterministic, public contracts.
- The pinned VOP compatibility contract, Arrow schema fingerprint, IPC/Parquet
  fixtures, and PyArrow/Polars round trips validate in hosted CI.
- Every archived Conductor track remains discoverable in `conductor/tracks.md`
  and is represented in the cross-repository GitHub historical ledger.
- Python 3.12–3.14, current compatible dependencies, security checks, coverage,
  repository harnesses, and benchmark regression gates remain green.
- External maturity, data, hardware, registry, and publication gates remain
  explicit even when repository implementation is complete.
- Cargo-authoritative dynamic versions with fail-closed release-tag validation,
  Pydantic v2 logging settings, structured
  run context, and uv/Pixi parity are enforced as production contracts.
- Ruff, `ty`, BasedPyright, package builds, unit/property/integration/E2E tests,
  security checks, and benchmark regression remain visible fast gates.
- Strict Pydantic v2 contracts give concerns, parameters, numerical policies,
  kernels, backend capabilities, run contexts and results stable typed
  identities with deterministic schemas and compatibility adapters.
- **M14 / planned v1.2.0:** estimation-focused `EVPPI_var` and `EVSI_var`
  declare scalar/vector target shape, component units, variance or covariance
  functional, conditioning and sampling models, estimator assurance and
  provenance.
- **M15 / planned v1.2.0:** COSS returns evaluated designs, feasible range/set,
  signed ENBS curve, deterministic tie policy, optimum, uncertainty and
  plotting inputs; EVSI/EVPI has common-unit, zero-EVPI and bounds behavior;
  the initial portfolio slice allocates governed optima under additive signed
  ENBS, capacity, dependency, exclusion and guardrail constraints while
  declaring metrics, interference, multiplicity, sequential/stopping rules,
  duration/delay, opportunity cost, policy changes, and gross/net outputs;
  model effects and disjoint incremental costs require provenance assurances,
  and tolerance ties are anchored to the fixed global maximum.
- **M16 / planned v1.2.0:** #595 represents utility, wealth/reference state,
  risk attitude, payoff units, information and cost location, current and
  informed policies, stakeholder scope, EUI, CEI, BPI, SPI, anchored PPI,
  policy switches, root diagnostics, direction/normalization and explicit
  cross-problem comparability. VoC is a presentation of the same
  clairvoyant-policy result, not a duplicate kernel; monetary EVPI reduction
  requires verified positive-affine utility.
- **M17 / planned v1.2.0:** the canonical C16 public projection keeps managed
  issue/subissue and Project 28 fields synchronized across every explicitly
  registered consumer repository, preserving human content and failing closed
  on conflicts or missing credentials.

### Should have

- New interchange profiles reuse the shared compatibility schema and canonical
  logical-field fingerprint algorithm.
- Free-threaded Python remains a bounded observational lane until the required
  wheels are published.
- Pull requests and historical development eras remain represented in the
  VOP–VOIAGE GitHub Project.
- Scalene, mutation, dependency-audit, and experimental lanes emit bounded
  scheduled/manual evidence rather than slowing every pull request.
- GitHub governance projections use stable markers, bounded managed sections,
  dry-run plans and conflict detection while preserving human-authored content.
- Specialized v1.2.0 methods should have independent references, accessible
  plots and explicit Rust/Python/R/Julia/Mojo dispositions before promotion.

### Could have

- Cross-language consumers and accelerators validated by the same fixtures.
- Automated synchronization of archived tracks and project fields.
- Signed release attestations for promoted interchange bundles.
- Deterministic governance traceability tables and Mermaid graphs.
- Reviewed vector-target covariance functionals beyond the initial declared
  trace, determinant and weighted-quadratic choices.

### Won't have now

- Automatic external publication or maturity promotion.
- Direct imports from the VOP source tree or repository consolidation.
- Production accelerator claims without parity and hardware evidence.
- Publication of credentials, private evidence, or local-only agent state.
- Automated acceptance of risk, irreversible decisions, or human-controlled
  issue closure.
- Duplicate VoC kernels, silent COSS extrapolation or relabeling
  `total_voi / total_cost` as EVSI/EVPI.

## Planned-version traceability

| Planned version | MoSCoW | Canonical requirement | VOIAGE track | GitHub |
|---|---|---|---|---|
| v1.2.0 | Must | M14 | `estimation_focused_variance_voi_20260727` | #619 under #318 |
| v1.2.0 | Must | M15 | `study_design_efficiency_20260727` | #571 under #318 |
| v1.2.0 | Must | M16 | `risk_adjusted_information_pricing_20260731` | #595 and #694–#697 under #318 |
| v1.2.0 | Must | M17 | canonical C16 plus the four specialized delivery tracks above | #313/#318 and Project 28 |
