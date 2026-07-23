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
- [~] Verify primary citations and classify estimand versus estimator.
- [~] Reconcile code, schemas, maturity, docs, roadmap, and fixtures.
- [x] Generate capability and method matrices. (`2630e39`)
- [ ] Triage buying-price versus expected-utility VOI, constructed-scale VOI,
  robust EIG, validation-study EVSI, and other genuinely distinct estimands
  found by the search; do not promote aliases as new methods.
- [ ] Reconcile VOP against preference, equity, heterogeneity, scenario, and
  robust-decision methods so perspective is not collapsed into those concepts.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Census and reconciliation'
  (Protocol in workflow.md).

## Phase 3: Review and freeze

- [ ] Run registry, citation, SourceRight, docs, and full repository gates.
- [~] Add citation-identifier validation and an evidence-preserving quarterly
  refresh job with a 93-day freshness limit. (source identifiers and complete
  method coverage landed in `c4c4fab`; scheduled refresh remains open)
- [ ] Add a missed-library/missed-method contribution template and duplicate-
  resistant triage automation.
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
