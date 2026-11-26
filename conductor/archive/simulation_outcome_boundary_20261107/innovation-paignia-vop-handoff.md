# Innovate, Paignia and VOP provider handoff

Date: 2026-11-08. Status: implementation pending. This supplement belongs to the existing simulation-outcome boundary track and issue #1286, with #1283/#1221 and scientific-alignment ownership preserved. Read spec.md and plan.md first; do not create a duplicate workstream or advance provider pins here.

## Ownership

Voiage owns reusable decision uncertainty, value of perspective/equity and VOI. `vop_poc_nz` is a proof of concept: reusable methods are candidates for verified promotion into Voiage, while NZ data, institutional/clinical assumptions and research scenarios remain there.

Innovate owns innovation modelling/prediction, including policy diffusion, competition, substitution, hype impact and medicines utilisation. **Paignia** is the chosen generic Rust game provider at [edithatogo/paignia](https://github.com/edithatogo/paignia). Its renamed `paignia-core`/`paignia-solve` seed from `new-drug-reimbursement-game` is merged at [12b22ce](https://github.com/edithatogo/paignia/commit/12b22ce7571b9c590405bf53c2c1c5862b49cd64); package publication, method and installed-consumer qualification remain pending. T17 retains its seventeen IDs through transfer: provider work belongs to Paignia, while application .09/.10 consumer and .11 reimbursement-valuation handoffs retain their receiving owners. Kairos owns simulation execution. Voiage must not absorb those providers' scientific engines.

Voiage may optionally consume valued joint outcomes. Application orchestration returns chosen policy/study designs as versioned artefacts. Neither Innovate nor Paignia core gains a reverse import of Voiage. Arbitrary strategic payoffs, scripts, patients or adoption counts are not automatically utilities or net benefits.

## Crosswalk to the existing execution leaves

VP01/VP02/VP04 are provider-specific acceptance details for the existing O01-O08 leaves, not additional completion counters. Execute each owning leaf's file budget, distinguishing tests and commands in [plan.md](plan.md); all eight remain pending. Reuse one receipt at the applicable contract/numerical boundary. Proposed targets remain future until implemented. Reconcile existing Rust domain/numerics APIs before adding a type or duplicate method.

| Handoff | Existing implementation owner |
|---|---|
| VP01 decision binding | O01 profile, O02 pairing, O03 value conventions, O04 parameter alignment, O06 interchange |
| VP02 numerical and native consumer fixture | O05 conversion, O07 independent EVPI, O08 installed use |
| VP03 VOP method promotion | Existing scientific-alignment owner and issue #1286, coordinated with Innovate E04; not an extra prerequisite for every outcome producer |
| VP04 optionality and feedback | O08 dependency isolation with O05's adapter boundary; study-design feedback remains issue #1286/Innovate E03 work; strategic responses reuse SI01-SI04 in PR #1288 |

### VP01 — explicit decision binding (O01-O04/O06)

Use O01's `specs/ecosystem/simulation-outcome/v1/profile.md` and inventoried Rust domain type, plus O02-O04/O06's named tests and fixtures. DecisionBindingV1 describes the producer projection into that profile or the compatible existing VOIBinding; it is not a second schema or parallel Python validator by default.

Preserve schema version, source/fit status, joint_draw_id, strategy_id, scenario_id, perspective, outcome_unit, population, time_basis, discount_basis, valuation_mapping_id and valued outcomes. Test missing/duplicate draw keys, swapped labels, invalid fits and absent valuation mapping before accepting an Innovate/Paignia artefact. Raw prescription units must fail. Preserve declared dependence by joining on draw identity, never independently resampling marginal columns; the producer must justify that dependence. Run the owning O leaf's focused target after creating its distinguishing test.

Amend those same tests to distinguish parameter, structural, simulation, measurement, strategic and numerical uncertainty. Map parameter-draw, model, equilibrium/selection, measurement/valuation version and simulation-replicate identities into the existing profile or an explicitly versioned extension. They describe roles and dependence, not independent random variables. An analyst's state draw is not automatically visible to the simulated players. Reject ambiguous simulation rows labelled as posterior/parameter draws. Preserve separate execution, convergence, identification, predictive-evidence and causal-assumption assessments; weak parameter identification is not automatically a failed execution, and a converged fit is not causal proof.

### VP02 — independent numerical and native consumer fixture (O05/O07/O08)

Use O07's `native_evpi_equivalence.rs`, `evpi-fixtures.json` and numerical qualification record, then O08's installed-consumer script and fixture. These are the targets already specified in the plan; do not add a competing `test_innovation_outcome_evpi.py` gate for the same behavior.

For equal-probability joint parameter draws with strategy A net benefits [4,0] and B [1,3], require EVPI=1.5 in the declared arbitrary net-benefit units; test exact joint-row permutation invariance, strategy-label preservation, all-equal zero EVPI and invalid-probability rejection. Use the independent formula E[max_a NB]-max_a E[NB] as oracle, not a Voiage-generated expected value. Test actual native dispatch in a clean installed consumer. Namespace/metadata success is not numerical acceptance.

Add a separate finite negative control with ONE parameter state and two equally weighted conditional simulation replicates: A=[-1,1], B=[1,-1]. Conditional means are both0, so conventional parameter EVPI is0. Treating simulation rows as parameter states yields1 and must fail. This distinguishes the information estimand, not merely an input dtype. For parameter VOI, estimate the conditional expected net benefit over appropriate simulation randomness before decision maximisation; carry inner/outer budgets, coupling and Monte Carlo standard error/bias diagnostics. A problem where process outcomes are revealed before action is a different explicitly labelled information problem.

Add unequal-mean and heteroscedastic fixtures, justified positive unit scaling and strategy permutation. Test error reduction over replicated increasing numerical effort with prespecified precision bounds, not deterministic monotonicity of every noisy run. Common random numbers are a declared coupling policy; equal seeds alone do not prove matched outcomes. Do not hide estimator bias or failed precision criteria by clipping values to zero.

Implement these as bounded O05/O07 leaves: first run the new assertion against an intentionally naive row-as-parameter conversion, record failure, then correct only the conversion/aggregation contract and rerun the existing native target plus installed-consumer test. Stop if the producer cannot identify what a draw represents; do not infer the missing hierarchy from array shape.

### VP03 — promote VOP one method at a time (existing issue #1286/E04 owner)

Files: proposed `docs/validation/vop-promotion-ledger.json` plus at most four provider/test/fixture files per method. Inspect the actual VOP implementation and existing Voiage counterpart; classify duplicate, correct-reusable, domain-specific or unsupported. Test a tiny independently calculated reference and known failure cases before implementing the provider slice. Start with decision/perspective/VOI methods, not a wholesale copy of a health-economic framework. Record method/version equivalence or justified differences. Keep invalid historical results labelled; do not make parity with a flawed POC the oracle.

After the provider passes, issue the exact package/hash and acceptance fixture to the POC's local migration owner. Only the consumer changes its adapter and pin. Preserve study/scenario reproducibility and a documented rollback; do not mark migration complete from this handoff alone. Independent VOP method inventory/promotion does not wait for an unrelated Innovate or Paignia adapter.

### VP04 — optionality, information estimands and coherent feedback (O08/E03/SI owners)

In O08's dependency/installed-consumer suite, seed a reverse core dependency and require rejection. Core Voiage must remain installable without Innovate, Paignia or Kairos. O05's optional adapter consumes versioned already-valued artifacts rather than requiring their producer engines. Check separate minimal/default/advertised-extra standalone consumers and resolved normal/build feature closures; a workspace all-features build is not isolation evidence. Keep study-design feedback with issue #1286/Innovate E03: test an explicit observation model, since generic variance reduction is not evidence of a real study.

Reuse the existing signed-social-information contract and SI01-SI04/SI-V01-SI-V08 proposal in [PR #1288](https://github.com/edithatogo/voiage/pull/1288), coordinated with [Paignia's handoff](https://github.com/edithatogo/paignia/pull/1). Do not introduce another strategic-value evaluator. Distinguish conventional EVPI/EVSI, net benefit of research, strategic information value and social information value. Record recipient, timing, public/private disclosure, feasible commitments, observation model, solution concept, equilibrium-selection rule and application-owned welfare weights. Preserve/version legacy helper semantics rather than silently changing a thresholded score into an equilibrium VOI result.

A fixed-policy finite case must reduce to conventional VOI under its assumptions. A supported recomputed-response case can have a signed strategic value: retain a negative reference value, do not clip it or relabel it conventional EVPI. Unsupported or nonconverged is not economic zero. Multiple-equilibrium ranges/selection remain explicit; solver restart frequencies are not behavioural probabilities. Test that hidden analyst states do not leak into player policies and that averaging independently solved fully observed games cannot substitute for the intended information structure.

Feedback, strategic research and VOP promotion are not prerequisites for accepting an unrelated static outcome fixture, ordinary VOI release or initial software paper. Preserve existing numerical/native/release gates, hardware evidence and deferred HPC limits. The application proposes the actual design; Voiage does not fabricate an innovation game or scheduler.

## Dependency and release rules

Align with central Rust/Polars/Arrow direction and current per-method acceptance boundaries. Isolated reference environments are not supported production execution. Qualify newest candidates separately from reproducible release locks. No package publication, public submission, completed VOP migration or named Paignia namespace reservation is included here.

Paignia core, supported static solvers and optional dynamic execution have independent qualification. Voiage's valued-input acceptance does not require a Kairos adapter; Paignia extraction does not require this Voiage integration. All claimed methods still require applicable numerical evidence. Each implementing leaf records concrete input/expected output, prohibited shortcut and stop condition before coding; unresolved scientific choices are not delegated to an invented default.

Related: Innovate PR #570; repository-standards PR #120 `docs/scientific-result-contracts.md`; Paignia PR #1; reimbursement PR #101/issue #100 and strategic pilot PR #103. These are coordinated proposals, not merged implementation evidence.
