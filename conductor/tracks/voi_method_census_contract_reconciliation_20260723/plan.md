# Track Implementation Plan: VOI/VOP Method Census And Contract Reconciliation

## Phase 1: Registry contract

- [x] Add failing schema, identifier, citation, and traceability tests.
  (`2630e39`)
- [x] Define the versioned registry and review/disposition vocabulary.
  (`2630e39`)
- [x] Record search protocol, evidence hierarchy, and source-rights boundary.
  (`2630e39`)
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update. (`2630e39`)
- [x] Automated review and validation checkpoint. (`c4c4fab`; added executable
  method-to-source coverage after self-review found that feature URLs alone did
  not satisfy method-level citation traceability)
- [ ] Conductor - User Manual Verification 'Phase 1: Registry contract'
  (Protocol in workflow.md).

## Phase 2: Census and reconciliation

- [x] Inventory repository methods and initial literature candidates.
  (`2630e39`; the search remains refreshable rather than universally exhaustive)
- [x] Verify primary citations and classify estimand versus estimator.
  (`ab55c1e`; 41 source-verified and 19 explicitly contract-verified records,
  with no unresolved triage state)
- [x] Reconcile code, schemas, maturity, docs, roadmap, and fixtures.
  (`3112d69`; corrected unsupported expected-loss and nested-MC EVPPI claims,
  and added generated implementation, test, authority, and remaining-gate
  evidence for every native method)
- [x] Generate capability and method matrices. (`2630e39`)
- [x] Triage buying-price versus expected-utility VOI, constructed-scale VOI,
  robust EIG, validation-study EVSI, and other genuinely distinct estimands
  found by the search; do not promote aliases as new methods. (`6e3ebb1`)
- [x] Triage Blackwell informativeness, value of signals, clairvoyance, control
  and flexibility, rational inattention, Bayesian persuasion, strategic
  information design, causal discovery, model discrimination, and value of
  measurement or test accuracy as VOI, related analysis, application, or
  reviewed exclusion. (`6e3ebb1`)
- [x] Define the canonical Decision Problem interchange contract and map every
  included estimand and estimator to its required fields. (`6e3ebb1`)
- [x] Reconcile VOP against preference, equity, heterogeneity, scenario, and
  robust-decision methods so perspective is not collapsed into those concepts.
  (`6e3ebb1`)
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update. (`3112d69`)
- [x] Automated review and validation checkpoint. (`3112d69`; 202 mapped
  Python tests and the complete `voiage-numerics` crate test suite passed;
  no additional correctness finding remained)
- [ ] Conductor - User Manual Verification 'Phase 2: Census and reconciliation'
  (Protocol in workflow.md).

## Phase 3: Review and freeze

- [x] Run registry, citation, SourceRight, docs, and full repository gates.
  (`f32573d`; generated registries and the freeze candidate were current,
  SourceRight reported zero diagnostics on a temporary normalized CSL
  projection, all 13 tox environments passed with 91.00% coverage, and the
  live GitHub programme validator and 27-workflow repository harness passed)
- [x] Generate a deterministic, hash-bound scientific-review candidate that
  joins all method maturity, evidence, implementation-authority, remaining-gate,
  and DecisionProblemV2 compatibility records without recording approval.
  Candidate digest:
  `9f437ea0b0521297b81f66adfac980e537db3c0ebf63823445f3bff2d285c3f9`.
  (`7a9d6b9`)
- [x] Add citation-identifier validation and an evidence-preserving quarterly
  refresh job with a 93-day freshness limit. (`6e3ebb1`)
- [x] Add a missed-library/missed-method contribution template and duplicate-
  resistant triage automation. (`6e3ebb1`)
- [x] Add a fail-closed, append-only approval recorder that binds an accountable
  human decision to the exact candidate digest and refuses stale candidates,
  incomplete evidence, or overwrites. (`c596380`)
- [x] Obtain human scientific review of stable definitions and dispositions.
  (`c7faac41`; `edithatogo (repository maintainer)` approved candidate digest
  `9f437ea0b0521297b81f66adfac980e537db3c0ebf63823445f3bff2d285c3f9`)
- [x] Freeze the v1.1 registry revision and record remaining research gates.
  (`c7faac41`; the separate approval artifact retains implementation,
  numerical-validation, binding-conformance, release, publication, and
  external gates)
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Review and freeze'
  (Protocol in workflow.md).

## Review fixes

- [x] Require every canonical method to resolve to one or more registered
  sources and an explicit review state. (`c4c4fab`)
- [x] Keep repository-defined VOP and LLM/agent applications visibly separate
  from primary-verified established methods. (`c4c4fab`)
- [x] Add executable required-field mappings after checkpoint review found that
  a boundary label alone did not satisfy the DecisionProblemV2 mapping claim.
  (`6e3ebb1`)
- [x] Correct the census after the external-software refresh found that the
  existing calibration VOI runtime and primary calibration-target literature
  were missing from the canonical method registry. (`7443253`)
- [x] Apply Ruff formatting to the trusted-publishing workflow assertion after
  the complete repository gate exposed pre-existing formatting drift.
  (`f32573d`)

## Phase 4: Additive industry DecisionProblem contract

- [ ] Under [#566](https://github.com/edithatogo/voiage/issues/566), freeze
  `industry-decision-problem-contract` as an additive v2 revision or a new
  version; do not mutate the accepted v2 meaning in place.
- [ ] Add failing schemas and semantic tests for decision criteria, risk
  measures, constraints, policies, action eligibility, observation models,
  privacy, latency, freshness, sequential state/stopping, metrics, guardrails,
  and versioned artifact references.
- [ ] Define typed predictive-distribution, CATE/uplift, forecast, optimization,
  experiment, model-registry, metric, and lineage adapter protocols without
  making external engines stable dependencies.
- [ ] Add deterministic JSON and Arrow fixtures, migration cases, invalid and
  pathological cases, capability requirements, and Rust/Python/R/Julia/Mojo
  dispositions.
- [ ] Reconcile the expanded contract with the canonical method registry and
  the supported-frontier, ML/causal, binding, example, and automation tracks.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 4: Additive industry DecisionProblem contract'
  (Protocol in workflow.md).

## Phase 5: Residual method classification

- [ ] Re-run the recorded literature protocol for #593--#600 and preserve
  primary definitions, equations, assumptions, aliases, adjacent concepts,
  software observations, and search limitations.
- [ ] Classify `implementation-information-decomposition` (#593), including
  expected value of perfect implementation (EVPIM), expected value of specific
  implementation (EVSIM), realizable EVPI, implementation-adjusted EVSI,
  expected value of perfection (EVP), and proposed EVEIm/EVSEIm terminology
  against `implementation-voi`.
- [ ] Classify `uncertainty-modelling-value` (#594), including EVIU, EEV, VSS,
  wait-and-see, EVPI, DVSS, and VMS, without treating modelling uncertainty as
  information acquisition.
- [ ] Classify `risk-adjusted-information-pricing` (#595), including EUI, CEI,
  BPI, SPI and constructed-scale prices, and reconcile the adjacent
  `buying-price-voi` record.
- [ ] Classify `event-localized-information-value` (#596), including perfect
  and imperfect event/tail-event information and information density.
- [ ] Classify `belief-state-sequential-information-value` (#597) against
  sequential VOI, real options, monitoring, knowledge gradient, active
  learning, agent information, POMDP observation value and dual control.
- [ ] Classify `signed-social-information-value` (#598) against strategic
  sharing, privacy/federated VOI, rational inattention, Bayesian persuasion,
  team decisions, information avoidance, overvaluation and harmful
  information.
- [ ] Classify `heterogeneity-value-decomposition` (#599) against
  heterogeneity VOI, individualized care, preference, equity, policy/uplift
  VOI, descriptive segmentation, and CATE estimation.
- [ ] Classify `outcome-conditional-sample-information-value` (#600),
  including delta-EV, VSI, sigma-VSI and rVSI, against EVSI, EVSI estimator
  uncertainty, risk-sensitive VOI and tail-event information.
- [ ] Generate an additive, hash-bound review candidate and require named
  scientific approval before changing canonical registry or maturity claims.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 5: Residual method classification'
  (Protocol in workflow.md).
