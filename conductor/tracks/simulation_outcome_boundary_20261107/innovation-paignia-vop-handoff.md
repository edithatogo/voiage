# Innovate, Paignia and VOP provider handoff

Date: 2026-11-08. Status: implementation pending. This supplement belongs to the existing simulation-outcome boundary track and issue #1286, with #1283/#1221 and scientific-alignment ownership preserved. Read spec.md and plan.md first; do not create a duplicate workstream or advance provider pins here.

## Ownership

Voiage owns reusable decision uncertainty, value of perspective/equity and VOI. `vop_poc_nz` is a proof of concept: reusable methods are candidates for verified promotion into Voiage, while NZ data, institutional/clinical assumptions and research scenarios remain there.

Innovate owns innovation modelling/prediction, including policy diffusion, competition, substitution, hype impact and medicines utilisation. **Paignia** is the chosen generic Rust game provider at [edithatogo/paignia](https://github.com/edithatogo/paignia). Its repository is created and is being populated with the renamed `paignia-core`/`paignia-solve` seed from `new-drug-reimbursement-game`; package publication and provider qualification remain pending. T17 retains its seventeen IDs through transfer: provider work belongs to Paignia, while application .09/.10 consumer and .11 reimbursement-valuation handoffs retain their receiving owners. Kairos owns simulation execution. Voiage must not absorb those providers' scientific engines.

Voiage may optionally consume valued joint outcomes. Application orchestration returns chosen policy/study designs as versioned artefacts. Neither Innovate nor Paignia core gains a reverse import of Voiage. Arbitrary strategic payoffs, scripts, patients or adoption counts are not automatically utilities or net benefits.

## Crosswalk to the existing execution leaves

VP01/VP02/VP04 are provider-specific acceptance details for the existing O01–O08 leaves, not additional completion counters. Execute each owning leaf's file budget, distinguishing tests and commands in [plan.md](plan.md); all eight remain pending. Reuse one receipt at the applicable contract/numerical boundary. Proposed targets remain future until implemented. Reconcile existing Rust domain/numerics APIs before adding a type or duplicate method.

| Handoff | Existing implementation owner |
|---|---|
| VP01 decision binding | O01 profile, O02 pairing, O03 value conventions, O04 parameter alignment, O06 interchange |
| VP02 numerical and native consumer fixture | O05 conversion, O07 independent EVPI, O08 installed use |
| VP03 VOP method promotion | Existing scientific-alignment owner and issue #1286, coordinated with Innovate E04; not an extra prerequisite for every outcome producer |
| VP04 optionality and feedback | O08 dependency isolation with O05's adapter boundary; study-design feedback remains issue #1286/Innovate E03 work |

### VP01 — explicit decision binding (O01–O04/O06)

Use O01's `specs/ecosystem/simulation-outcome/v1/profile.md` and inventoried Rust domain type, plus O02–O04/O06's named tests and fixtures. DecisionBindingV1 describes the producer projection into that profile or the compatible existing VOIBinding; it is not a second schema or parallel Python validator by default.

Preserve schema version, source/fit status, joint_draw_id, strategy_id, scenario_id, perspective, outcome_unit, population, time_basis, discount_basis, valuation_mapping_id and valued outcomes. Test missing/duplicate draw keys, swapped labels, invalid fits and absent valuation mapping before accepting an Innovate/Paignia artefact. Raw prescription units must fail. Preserve declared dependence by joining on draw identity, never independently resampling marginal columns; the producer must justify that dependence. Run the owning O leaf's focused target after creating its distinguishing test.

### VP02 — independent numerical and native consumer fixture (O05/O07/O08)

Use O07's `native_evpi_equivalence.rs`, `evpi-fixtures.json` and numerical qualification record, then O08's installed-consumer script and fixture. These are the targets already specified in the plan; do not add a competing `test_innovation_outcome_evpi.py` gate for the same behavior.

For equal-probability joint draws with strategy A net benefits [4,0] and B [1,3], require EVPI=1.5 in the declared arbitrary net-benefit units; test exact joint-row permutation invariance, strategy-label preservation, all-equal zero EVPI and invalid-probability rejection. Use the independent formula E[max_a NB]-max_a E[NB] as oracle, not a Voiage-generated expected value. Test actual native dispatch in a clean installed consumer. Namespace/metadata success is not numerical acceptance.

### VP03 — promote VOP one method at a time (existing issue #1286/E04 owner)

Files: proposed `docs/validation/vop-promotion-ledger.json` plus at most four provider/test/fixture files per method. Inspect the actual VOP implementation and existing Voiage counterpart; classify duplicate, correct-reusable, domain-specific or unsupported. Test a tiny independently calculated reference and known failure cases before implementing the provider slice. Start with decision/perspective/VOI methods, not a wholesale copy of a health-economic framework. Record method/version equivalence or justified differences. Keep invalid historical results labelled; do not make parity with a flawed POC the oracle.

After the provider passes, issue the exact package/hash and acceptance fixture to the POC's local migration owner. Only the consumer changes its adapter and pin. Preserve study/scenario reproducibility and a documented rollback; do not mark migration complete from this handoff alone.

### VP04 — prove optionality and coherent feedback (O08 and existing E03 owner)

In O08's dependency/installed-consumer suite, seed a reverse core dependency and require rejection. Core Voiage must remain installable without Innovate, Paignia or Kairos. O05's optional adapter consumes versioned already-valued artifacts rather than requiring their producer engines. Keep study-design feedback with issue #1286/Innovate E03: test an explicit observation model, since generic variance reduction is not evidence of a real study. Feedback and VOP promotion do not become prerequisites for accepting an unrelated static outcome fixture. Preserve the existing native/scientific/release gates, hardware evidence and deferred HPC limits.

## Dependency and release rules

Align with the central Rust/Polars/Arrow direction and current per-method acceptance boundaries. Unsupported host fallback remains explicit; isolated reference environments are not supported production execution. Use newest qualified candidates, not untested blanket upgrades. No package publication, public submission, completed VOP migration or named Paignia namespace reservation is included in this planning change.

Paignia core, supported static solvers and optional dynamic execution have
independent qualification. Voiage's valued-input acceptance does not require a
Kairos adapter; Paignia extraction does not require this Voiage integration.
All method claims still require their own applicable numerical evidence.

Related: Innovate PR #570; repository-standards PR #120 and `docs/paignia-library-boundary.md`; reimbursement-game PR #101/issue #100. All remain separately qualified.
