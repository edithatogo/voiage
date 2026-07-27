# Track Implementation Plan: Supported Frontier Method Completion

## Phase 1: Family audit

- [ ] Add failing completeness, maturity, placeholder, RNG, exception, and
  dependency-boundary tests.
- [ ] Map every module and contract to the canonical method registry.
- [ ] Record duplicates, scientific gaps, and promotion requirements.
- [ ] Freeze the DSA estimand and schema under
  [#556](https://github.com/edithatogo/voiage/issues/556), method ID
  `deterministic-sensitivity-analysis`: baseline, one-way/multi-way/scenario
  inputs, units and ranges, deterministic evaluator, switching points,
  rankings, diagnostics, tie policy, and explicit separation from PSA, global
  sensitivity, and VOI.
- [ ] Freeze the VDI estimand and schema under
  [#557](https://github.com/edithatogo/voiage/issues/557), method ID
  `value-of-distributional-information`: candidate distributions, model
  probabilities, conditioning order, baseline/resolved decisions, gross/net
  VDI, estimator assurance, and explicit separation from distributional-equity
  VOI.
- [ ] Freeze the qualitative VoI schema and review state machine under
  [#558](https://github.com/edithatogo/voiage/issues/558), method ID
  `qualitative-voi`: decision, uncertainties, impacts, feasibility, timing,
  equity/ethics, information actions, burdens, confidence, provenance, dissent,
  missingness, redaction, accessibility, audit history, and prohibition on
  fabricated quantitative VOI.
- [ ] Freeze the VoF estimand and compatibility mapping under
  [#559](https://github.com/edithatogo/voiage/issues/559), method ID
  `value-of-flexibility`: constrained and flexible policy sets, commitment
  baseline, timing, discounting, irreversibility, lock-in, exercise rules,
  value decomposition, and the versioned relationship to dynamic real-options
  and sequential contracts.
- [ ] Freeze supported MCDA decision-rule families and MCDA-VOI schemas under
  [#560](https://github.com/edithatogo/voiage/issues/560), method ID `mcda-voi`:
  criteria, scales, directions, value functions, preferences, correlations,
  aggregation, thresholds/vetoes, alternatives, uncertainty, information
  actions, normalization, rankings, regret, rank acceptability, and estimator
  assurance.
- [ ] Record a Rust/Python/R/Julia/Mojo disposition for #556--#560 using only
  `implemented`, `adapter`, `contract-only`, `unsupported`, or
  `upstream-blocked`; fail governance when capability claims exceed installed
  execution.
- [ ] Disposition adjacent information-ordering, strategic-information,
  measurement-value, causal-discovery, model-discrimination, value-of-control,
  and value-of-flexibility families without conflating them with decision VOI.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 1: Family audit'
  (Protocol in workflow.md).

## Phase 2: Completion and consolidation

- [ ] Implement missing validated behavior and typed diagnostics.
- [ ] Implement #556 from its frozen DSA contract with a deterministic evaluator
  boundary, baseline and incremental curves, decision-switch detection,
  one-way/multi-way/scenario execution, typed failures, canonical
  serialization, and explicit CLI/docs/plotting disposition.
- [ ] Implement #557 from its frozen VDI contract with model-mixture
  conditioning, gross/net value, baseline and resolved decisions, typed
  assurance, and fail-closed probability/draw/model validation.
- [ ] Implement #558 as a versioned portable assessment, validator, deterministic
  renderer/serializer, disagreement-preserving audit log, incomplete and
  unverified states, redaction controls, and human-review boundary; do not add
  a numerical kernel merely to satisfy polyglot symmetry.
- [ ] Reconcile #559 with the existing dynamic real-options and sequential
  runtime: either expose a versioned VoF alias/view or document a reviewed
  exclusion, while returning the constrained/flexible values and separating
  flexibility, waiting, control, and information components.
- [ ] Replace the mock-only MCDA references for #560 with installed execution
  for each accepted decision-rule family, including decision/ranking output,
  criterion/preference information value, regret, rank acceptability,
  dominance diagnostics, and fail-closed model-family validation.
- [ ] Consolidate duplicates and remove placeholders with compatibility aliases.
- [ ] Add Rust-owned numerical kernels for accepted quantitative estimands,
  thin Python/R/Julia adapters, an explicit Mojo upstream disposition, and
  contract-only portability for qualitative VoI where appropriate.
- [ ] Add, for every accepted workstream, an analytical or enumerable example,
  independent reference, metamorphic properties, invalid/pathological fixtures,
  deterministic serialization, capability/registry entries, executable docs,
  and estimator or audit assurance. VDI must include distribution-mixture
  examples; VoF finite-horizon enumeration; MCDA scale/weight/tie cases; DSA
  switching/range cases; qualitative VoI disagreement and incomplete cases.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Completion and consolidation'
  (Protocol in workflow.md).

## Phase 3: Maturity review

- [ ] Run per-family, Rust, binding, frontier, docs, and full quality gates.
- [ ] Verify the native GitHub #318 subissue set contains #556--#560, each issue
  points back to this track, Project 28 metadata is complete, and local
  metadata/index/plan references match live state.
- [ ] Verify each accepted quantitative workstream through the public Rust
  facade and every advertised language binding; record explicit
  unsupported/upstream-blocked evidence rather than treating omission as parity.
- [ ] Require human usability/accessibility and audit-boundary review for
  qualitative VoI; automated schema checks cannot approve qualitative
  judgements.
- [ ] Reconcile every maturity claim and external public-data gate.
- [ ] Record separate v1.2/v1.3 implementation, scaffold, exclusion, and
  maturity decisions for #556, #557, #558, #559, and #560. Issue closure
  requires estimand-specific executable evidence or a reviewed exclusion; a
  citation, mock, schema-only scaffold, or adjacent method is insufficient.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Maturity review'
  (Protocol in workflow.md).
