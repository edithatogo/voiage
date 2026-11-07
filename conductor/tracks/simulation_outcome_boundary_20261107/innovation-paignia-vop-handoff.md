# Innovate, Paignia and VOP provider handoff

Date: 2026-11-08. Status: implementation pending. This supplement belongs to the existing simulation-outcome boundary track and issue #1286, with #1283/#1221 and scientific-alignment ownership preserved. Read spec.md and plan.md first; do not create a duplicate workstream or advance provider pins here.

## Ownership

Voiage owns reusable decision uncertainty, value of perspective/equity and VOI. `vop_poc_nz` is a proof of concept: reusable methods are candidates for verified promotion into Voiage, while NZ data, institutional/clinical assumptions and research scenarios remain there.

Innovate owns innovation modelling/prediction, including policy diffusion, competition, substitution, hype impact and medicines utilisation. **Paignia** is the working name of the generic Rust game provider currently incubated in `new-drug-reimbursement-game` Track T17; proposed package names are not published dependencies. Kairos owns simulation execution. Voiage must not absorb those providers' scientific engines.

Voiage may optionally consume valued joint outcomes. Application orchestration returns chosen policy/study designs as versioned artefacts. Neither Innovate nor Paignia core gains a reverse import of Voiage. Arbitrary strategic payoffs, scripts, patients or adoption counts are not automatically utilities or net benefits.

## Bounded execution leaves

Each leaf changes one behaviour, touches at most five files and records an actual failing assertion, passing focused test, exact source/artifact hashes and nonzero test count. Proposed paths below are future targets, not commands claimed to pass today. Reconcile existing Rust domain/numerics APIs before adding a new type or duplicate method.

### [ ] VP01 — explicit decision binding

Files: existing `specs/ecosystem/README.md`; proposed `specs/ecosystem/decision-binding-v1.md`, `tests/test_decision_binding_contract.py`; resolve the authoritative Rust domain type and adapter file in the leaf before implementation.

Define a DecisionBindingV1 contract with schema version, source/fit status, joint_draw_id, strategy_id, scenario_id, perspective, outcome_unit, population, time_basis, discount_basis, valuation_mapping_id and valued outcomes. Test missing/duplicate draw keys, swapped labels, invalid fits and absent valuation mapping before accepting an Innovate/Paignia artefact. Raw prescription units must fail. Preserve dependence by joining on draw identity, never independently resampling marginal columns. Run `python -m pytest tests/test_decision_binding_contract.py` only after creating the test and observing its distinguishing failure; record the actual Rust test target after API inventory.

### [ ] VP02 — independent numerical and native consumer fixture

Files: proposed `tests/test_innovation_outcome_evpi.py`, a native fixture module in the inventoried numerical crate, fixture JSON, installed-consumer script and receipt.

For equal-probability joint draws with strategy A net benefits [4,0] and B [1,3], require EVPI=1.5 in the declared arbitrary net-benefit units; test exact joint-row permutation invariance, strategy-label preservation, all-equal zero EVPI and invalid-probability rejection. Use the independent formula E[max_a NB]-max_a E[NB] as oracle, not a Voiage-generated expected value. Test actual native dispatch in a clean installed consumer. Namespace/metadata success is not numerical acceptance.

### [ ] VP03 — promote VOP one method at a time

Files: proposed `docs/validation/vop-promotion-ledger.json` plus at most four provider/test/fixture files per method. Inspect the actual VOP implementation and existing Voiage counterpart; classify duplicate, correct-reusable, domain-specific or unsupported. Test a tiny independently calculated reference and known failure cases before implementing the provider slice. Start with decision/perspective/VOI methods, not a wholesale copy of a health-economic framework. Record method/version equivalence or justified differences. Keep invalid historical results labelled; do not make parity with a flawed POC the oracle.

After the provider passes, issue the exact package/hash and acceptance fixture to the POC's local migration owner. Only the consumer changes its adapter and pin. Preserve study/scenario reproducibility and a documented rollback; do not mark migration complete from this handoff alone.

### [ ] VP04 — prove optionality and coherent feedback

In the existing dependency/installed-consumer suite, seed a reverse core dependency and require rejection. Core Voiage must remain installable without Innovate, Paignia or Kairos. A supported optional adapter may consume their versioned artefacts. Test a study-design feedback payload with an explicit observation model; a generic variance reduction is not evidence of a real study. Preserve the existing native/scientific/release gates, hardware evidence and deferred HPC limits.

## Dependency and release rules

Align with the central Rust/Polars/Arrow direction and current per-method acceptance boundaries. Unsupported host fallback remains explicit; isolated reference environments are not supported production execution. Use newest qualified candidates, not untested blanket upgrades. No package publication, public submission, completed VOP migration or named Paignia namespace reservation is included in this planning change.

Related: Innovate PR #570; repository-standards PR #120 and `docs/paignia-library-boundary.md`; reimbursement-game PR #101/issue #100. All remain separately qualified.
