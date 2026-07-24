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

- [ ] Run registry, citation, SourceRight, docs, and full repository gates.
- [x] Add citation-identifier validation and an evidence-preserving quarterly
  refresh job with a 93-day freshness limit. (`6e3ebb1`)
- [x] Add a missed-library/missed-method contribution template and duplicate-
  resistant triage automation. (`6e3ebb1`)
- [ ] Obtain human scientific review of stable definitions and dispositions.
- [ ] Freeze the v1.1 registry revision and record remaining research gates.
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
