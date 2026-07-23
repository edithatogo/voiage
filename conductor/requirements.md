# VOIAGE requirements

This repository implements the production consumer side of the VOP–VOIAGE
programme. The canonical cross-repository requirements are maintained in
`vop_poc_nz/conductor/requirements.md`.

## MoSCoW priorities

### Must have

- The comprehensive Rust-first programme is represented by GitHub parent issue
  #313, native subissues #314--#323, Project 28 items, and one matching active
  Conductor track per issue.
- Maintained open-source VOI library features are implemented independently or
  carry an evidence-backed exclusion; proprietary and web-only parity is never
  claimed beyond reproducible public behavior.
- The canonical method registry distinguishes estimands, estimators,
  applications, visualizations, aliases, and maturity.
- The software census searches CRAN, R-universe, PyPI, crates.io, Julia
  General, available Mojo channels, GitHub, GitLab, published supplements,
  web tools, commercial documentation, and adjacent Bayesian-design and
  active-learning ecosystems on a recorded date.
- Every observed software feature maps to one or more stable canonical method
  identifiers and one of `native`, `equivalent`, `adapter`, `planned`,
  `excluded`, or `not-reproducible`.
- The generated feature matrix reports versions, maintenance, licenses,
  evidence links, method class, MoSCoW priority, VOIAGE state, and gaps.
- Direct decision VOI, Value of Perspective, estimators, visualizations,
  general modeling, information-theoretic design, and active-learning
  acquisition remain explicitly distinguishable.
- Registry snapshots expire within 93 days and before every minor release;
  stale matrices fail validation.
- Stable numerics are Rust-authoritative and surfaced consistently through
  Rust, Python, R, Julia, and Mojo.
- ML, LLM, and agent analyses distinguish decision VOI from entropy-only
  information gain and define utility, cost, posterior action, and stopping.
- Human CRediT and material AI-assistance records are reviewable, privacy-
  preserving, and release-linked.
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
- The documentation site pins the currently reviewed Astro and Starlight
  releases and a commit-pinned `edithatogo/astro-polyglot` source dependency;
  CI initializes submodules recursively and fails closed if extraction,
  generated MDX validation, link validation, or the production build fails.
- Generated API pages are reproducible, excluded from version control, limited
  to public members, and written only beneath the configured Starlight content
  directory.
- A canonical Decision Problem interchange contract represents alternatives,
  uncertain states, information actions, utility or loss, perspective,
  population, time horizon, units, currency year, provenance, and posterior or
  predictive draws without binding users to a particular inference engine.
- Every stochastic estimator reports an estimator-specific assurance envelope:
  uncertainty or Monte Carlo error, convergence and effective-sample
  diagnostics where applicable, replication count, RNG algorithm and seed,
  computational budget, stopping reason, and numerical-error budget.
- The literature census explicitly dispositions Blackwell informativeness,
  value of signals, clairvoyance, control and flexibility, rational
  inattention, Bayesian persuasion and strategic information design, causal
  discovery, model discrimination, and value of measurement or test accuracy.
- Capability tables, binding availability, public documentation, and release
  claims are generated or checked against the canonical registries and fail
  closed on unsupported or maturity-inflated claims.
- ML, LLM, retrieval, verifier, and agent analyses model correlated failure,
  prompt injection, data or retrieval poisoning, tool exfiltration, reward
  hacking, evaluation contamination, adaptive overfitting, and human override
  where those risks can change the preferred information action.
- Stable kernels define deterministic parallel reduction, recorded splittable
  RNG streams, streaming or out-of-core behavior, and bounded memory, latency,
  and energy evidence for advertised scalability profiles.

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
- Registry refresh automation proposes evidence-preserving diffs and never
  overwrites a reviewed exclusion, license decision, or human scientific note.
- Archived or inactive software with unique capabilities remains discoverable
  with its maintenance state and closest supported VOIAGE workflow.
- Each `planned` parity row is converted into an implementation, fixture, or
  reviewed exclusion before the relevant release closes.
- Native Rust, R, Julia, and Mojo documentation extractors should be enabled
  only after their toolchains and generated pages pass the same deterministic
  conformance checks as the initial Python lane.
- Material estimand, estimator, ABI, backend, exclusion, and deprecation
  decisions should have versioned architecture decision records.
- Every worked example should include a reproducibility card, assumptions,
  estimator uncertainty, sensitivity analysis, failure cases, accessible
  tables or plots, and deterministic offline instructions.
- Quarterly and pre-release drift proposals should cover software and
  literature registries, language toolchains, lockfiles, actions, and the
  source-pinned documentation plugin without automatically changing scientific
  dispositions.

### Could have

- Cross-language consumers and accelerators validated by the same fixtures.
- Automated synchronization of archived tracks and project fields.
- Signed release attestations for promoted interchange bundles.
- Deterministic governance traceability tables and Mermaid graphs.
- SPDX normalization, citation-identifier validation, and automated registry
  metadata refresh for records with authoritative machine-readable endpoints.
- A documented candidate-submission template for libraries missed by the
  reproducible search.
- A machine-readable gap report that opens or updates bounded method-triage
  issues without creating duplicate issues.
- A conformance corpus for adversarial and safety-sensitive information
  actions, including poisoned retrieval, dependent judges, and abstention or
  escalation under correlated errors.

### Won't have now

- Unverifiable implementation parity with proprietary or web-only tools.
- A claim that the landscape is universally exhaustive; registries, search
  indexes, terminology, and private software make that claim impossible to
  substantiate.
- API cloning, trademark imitation, or copying license-incompatible source.
- Treating entropy reduction, BALD, uncertainty sampling, or acquisition
  scores as economic VOI without an explicit action and utility or loss.
- AI systems listed as authors or CRediT contributors.
- Required network or model-provider access for deterministic examples.
- Automatic external publication or maturity promotion.
- A claim that `astro-polyglot` is registry-published, or that every language
  extractor is production-ready, before the corresponding external and
  conformance evidence exists.
- Direct imports from the VOP source tree or repository consolidation.
- Production accelerator claims without parity and hardware evidence.
- Publication of credentials, private evidence, or local-only agent state.
- Automated acceptance of risk, irreversible decisions, or human-controlled
  issue closure.
