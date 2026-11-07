# Rust-authoritative VOI Frontier Implementation Plan

> For agentic workers: use `superpowers:executing-plans`, or the approved subagent workflow, to implement one admitted packet at a time. Checkboxes below are pending work, not completion evidence.

**Goal:** Deliver a validated Rust numerical authority and a reproducible evaluation programme before advancing new VOI methods.

**Architecture:** Consolidate supported semantics in existing Rust crates. Use thin bindings and versioned producer artifacts; keep independent comparators outside the production core.

**Tech stack:** Existing Rust workspace, C ABI/PyO3, Python orchestration, R/Julia bindings, Arrow/Parquet, Cargo/Pytest/tox and current Conductor evidence controls.

**Spec:** [Programme contract](README.md).

## Global constraints

- Preserve Rust 1.85 support unless a separately reviewed change revises it; preserve the current Python 3.15.1 support policy rather than broadening it incidentally.
- Preserve offline R builds, promised public API/C ABI behaviour and explicit compatibility migrations.
- HPC remains deferred except bounded MLX/Metal work; optional accelerators cannot block an accepted CPU slice.
- Reuse V01–V12/F01–F08 and existing registries. Do not create a second generic scheduler, inference engine, trainer, dataset registry or publication client.
- Source admissions require exact revision, rights, provenance and a declared decision problem. Candidate links are not qualified datasets.
- No automated scientific promotion, external submission, new external repository, authenticated upload or authorship certification.

## Review focus

1. Correlated draws, changed weights or reordered labels must not silently change the estimand: Q01/Q03/Q08.
2. Extreme finite values, ties and near-zero decision differences must retain declared numerical behaviour: Q02/Q04/Q07.
3. Wrong study likelihoods, incoherent prior/posterior draws and degenerate importance weights must be diagnosed: Q05.
4. Overlapping evidence, delay, adoption and capacity must not be double-counted: Q06/R02.
5. Interrupted work, unavailable optional providers and copied release claims must not masquerade as completed results: Q07/Q08/Q09.

## Worker protocol

For each packet: inspect its exact baseline and current tests; reserve its write scope;
write a focused failing witness; demonstrate the expected failure; implement the
smallest change; run affected tests; retain outputs and review the diff; then open a
single-scope PR. First reuse the acceptance tests named below. Proposed test modules
are explicitly labelled; no command here is evidence that a check has run.

Before implementation admission, freeze the exact types/signatures in the existing
contract owner, case-specific error budget, acceptance commands and rollback.
Do not invent competing interfaces in parallel. A source or rights investigation
can proceed before the dependent numerical interface is frozen.

### Q01 — Reconcile authority, capabilities and scientific claims

Maps to V01/V02/F01/F08. Dependencies: none.

**Inspect/modify:** `README.md`, `specs/`, `voiage/contracts/capabilities.py`,
`voiage/contracts/capability_report.py`, method docs under
`docs/astro-site/src/content/docs/methods/`, binding manifests, the root historical
gap/literature documents. Reuse existing capability registries, not a new runtime list.

**Produces:** a method × estimator × study-model × binding × backend inventory with
source owner, maturity, assumptions, reference case and unsupported disposition.

- [ ] Add failing coverage for a native claim backed only by a Python bridge and for a capability missing its study-model/reference contract.
- [ ] Reconcile actual installed behaviour with registry, README, papers and release docs; mark historical reviews as historical with links rather than deleting evidence.
- [ ] Freeze the minimum decision/sample/result contracts and supported weight/unit conventions; record any signature changes before downstream work.
- [ ] Run existing capability-report and consumer-matrix tests and the relevant docs/registry checks. Require every advertised capability to have a verification family or an explicit exclusion.

### Q02 — Consolidate the Rust numerical authority

Maps to V12/F01/F04. Depends on Q01.

**Inspect/modify:** `rust/crates/voiage-numerics/`, `rust/crates/voiage-domain/`,
`rust/crates/voiage-ffi/`, `rust/crates/voiage-python/`, and the bundled native sources
and build scripts under `r-package/voiageR/`. Keep source moves smaller than algorithm changes.

**Consumes:** Q01 contracts. **Produces:** one shared implementation with a minimal
Rust-only consumer and reproducible offline R packaging.

- [ ] Add independent EVPI witnesses for payoff rows `[[0,100],[100,0]]`: equal weights yield 50; weights 0.9/0.1 yield 10 where weighted support is declared. Unsupported weighted paths must reject explicitly.
- [ ] Add translated, scaled, tied, extreme-finite and invalid-input cases; test installed R/native agreement and shared-source identity. Do not inflate tolerances to preserve a faulty implementation.
- [ ] Choose a dependency-light shared crate or deterministic vendoring of the same source; preserve offline source builds and document the choice.
- [ ] Run Cargo numerical/FFI tests and offline R build/check plus installed smoke tests. A Rust-only consumer must have no Python/PyO3 runtime requirement.

### Q03 — Admit the independent reference corpus

Maps to V02/V05/V08/F08. Depends on Q01 for contract freeze; source/rights preflight may start immediately.

**Inspect/extend:** existing `specs/numerical-reference/`, `specs/core-api/fixtures/`,
`specs/ecosystem/fixtures/`, dataset registry and `validation/` infrastructure.

**Produces:** versioned case manifests and independent reference artifacts, not only raw tables.

- [ ] Start with exact finite decisions and conjugate studies; add chemotherapy and BCEA, then DARTH only after provenance and reproduction admission.
- [ ] Test rejection of missing units, broken draw alignment, absent licences/checksums, unpinned sources and references computed by the implementation under test.
- [ ] Record the decision, study, expected answer/error bound, generator, seed, source rights, transformation history and reference implementation separately.
- [ ] Reproduce each admitted case offline from pinned inputs. Hold out cases from estimator tuning; disclose synthetic and same-author application cases.

### Q04 — Qualify nonlinear EVPPI

Maps to V02/V03/F04. Depends on Q01/Q02/Q03.

**Inspect/modify:** `rust/crates/voiage-numerics/src/evppi.rs`, `voiage/methods/basic.py`,
existing metamodel ports, `rust/crates/voiage-numerics/tests/evppi*`, and Python EVPPI tests.

**Produces:** at least one qualified nonlinear route alongside the linear reference,
with an explicit applicability and surrogate-diagnostic contract.

- [ ] Add held-out nonlinear, interaction, correlation and near-boundary problems where a linear fit is inadequate; test invariants and expected decision losses.
- [ ] Compare maintained Rust candidates and a qualified MARS artifact route; keep training producer-owned. Do not require every GAM/GP/BART/INLA alternative before a useful validated release.
- [ ] Compare with pinned `voi` flexible estimators and independent nested references; report approximation error separately from sampling error.
- [ ] Test held-out prediction/decision diagnostics, reproducibility and supported failure states before changing maturity.

### Q05 — Qualify coherent EVSI estimators

Maps to V03/V04/V05/V07/F02. Depends on Q01/Q02/Q03; nonlinear surrogate variants also depend on Q04.

**Inspect/modify:** `rust/crates/voiage-numerics/src/evsi*.rs`,
`voiage/methods/sample_information.py`, study-design contracts and their existing tests.

**Produces:** estimator-specific study contracts and comparison results; generic external
inference remains behind a narrow posterior-sample/diagnostic interface.

- [ ] Reproduce normal–normal and beta–binomial cases, then gamma–Poisson with controlled quadrature/truncation bounds; distinguish per-arm from total sample size.
- [ ] Add failures for mismatched priors, invalid joint updates, malformed callbacks and importance-weight degeneracy. Retain raw estimates and diagnostics rather than silently clamping convergence failures.
- [ ] Qualify nested, regression, moment-matching and importance-sampling routes individually against the same study. A sample-size proxy or seeded-bootstrap summary is not automatically a published EVSI estimator.
- [ ] Report bias, RMSE, interval coverage where intervals are supplied, effective sample size where applicable, decision error, runtime and memory across prespecified seeds/budgets.
- [ ] Admit multilevel/QMC/unbiased extensions only after reference correctness and a separate algorithm/accuracy comparison; existing methods are not novelty claims.

### Q06 — Verify research-design and portfolio decisions

Maps to V06/V10/F06. Depends on Q03/Q05.

**Inspect/modify:** `rust/crates/voiage-numerics/src/enbs.rs`,
`rust/crates/voiage-numerics/src/study_design.rs`, study-design/portfolio contracts,
`voiage/experimental/study_design.py` and associated tests.

**Produces:** reproducible sample-size/allocation/follow-up choices and bounded portfolio results.

- [ ] Verify ENBS 50 × 100 − 6000 = −1000 under an explicitly single-period, undiscounted fixture, including the no-study option.
- [ ] Enumerate small design/portfolio optima independently, including ties, binding budgets, shared outcomes and overlapping evidence; reject unsupported dependence rather than summing incompatible EVSI values.
- [ ] Test population timing, discounting, recruitment delay, implementation and capacity without unit mixing or double counting.
- [ ] Score selected-design regret as well as scalar error; document when an algorithm is exact, approximate or unsupported.

### Q07 — Thin bindings and measured computation

Maps to V12/F04/F05. Depends on Q02/Q03; qualification follows each accepted method rather than waiting for the entire catalogue.

**Inspect/modify:** existing FFI/PyO3 adapters, `bindings/julia/`, `r-package/voiageR/`,
`voiage/backends/`, `voiage/parallel*`, existing benchmark and installed-consumer tests.

**Produces:** installed cross-language evidence and accuracy-matched CPU performance results.

- [ ] Exercise the same positive, invalid and unavailable-method cases through installed Rust/Python/R/Julia consumers; no silent Python fallback for a native claim.
- [ ] Profile `.tolist()` crossings, allocation/sorting and serialization. Test buffer ownership, lifetimes, shape/stride handling and copy counts before claiming zero-copy.
- [ ] Benchmark deterministic block/streaming reductions against existing numerics. Record thread count, cancellation, checkpoint/restart and partial-result semantics.
- [ ] Separate same-estimator implementation, matched-accuracy algorithm and same-workflow hardware comparisons. Include cold/warm execution, transfers, compilation and memory.
- [ ] Preserve the HPC deferral. A bounded Metal/MLX experiment needs real-device accuracy, CPU fallback and end-to-end evidence; lack of hardware cannot block CPU acceptance.

### Q08 — Qualify one cross-library workflow at a time

Maps to F08/F06/F07. Depends on Q01/Q03 and the numerical packets used by each pilot.

**Inspect/extend:** `specs/integration/vop-voiage/`, existing ecosystem fixtures and
installed-consumer tracks. Producer changes are separate PRs in the capability owner's repository.

**Produces:** a pinned producer export and installed VOI result with independent reference.

- [ ] Reuse VOP/Tobac evidence first; choose one Innovate adoption workflow next, without describing same-author integration as independent adoption.
- [ ] Test different provider APIs that export equivalent semantics, then wrong units/weights/timebases, missing providers and changed export versions.
- [ ] Qualify MARS fitted-artifact replay before assuming Rust training or importing Python-linked code into the core.
- [ ] Keep Kairos/lifecourse/careops-sim simulation and MCHS/atlas/rare-burden evidence production outside Voiage. Record maturity and rights for each admitted producer.

### Q09 — Comparative evaluation and publication packet

Maps to V02/V12/F08; coordinates with the existing release/venue track. Depends on
Q01/Q03 and completed evidence for the exact Q04–Q08 subset claimed by each paper.

**Inspect/extend:** `validation/`, existing release/venue packets and reproducibility
artifacts; manuscript edits must first load the repository's manuscript instructions.

- [ ] Freeze a protocol before confirmatory runs: estimands, cases, comparator revisions, budgets, tolerances, seeds, failure accounting and decision-quality outcomes.
- [ ] Run the matrix in clean supported environments; retain failed and inconclusive cases. Arrange an external reproduction without falsely certifying it in advance.
- [ ] Prepare pyOpenSci then eligible JOSS software evidence; prepare the distinct Medical Decision Making evaluation. Preserve rOpenSci and deferred arXiv gates.
- [ ] Prepare licence-compatible Hub artifacts and DOI-backed release bundles, but publish only after destination, rights and owner approval.

### R01–R03 — Research gates, not automatic feature activation

- [ ] R01: after Q05–Q07, test decision-aware simulation allocation/stopping against fixed-budget and appropriate published methods. Require valid sequential error control and lower time-to-decision-accuracy.
- [ ] R02: after Q05/Q06 and the Innovate Q08 pilot, compare joint research–implementation–capacity optimisation with sequential and uptake-multiplier baselines. Use exact small cases and held-out scenarios; forecasting fit is not causal validation.
- [ ] R03: after Q05/Q08, investigate transport, confounding/bias, equity and perspective-sensitive evidence choices. Require identified assumptions, independent structural examples and sensitivity to misspecification.

Each research gate starts with a current primary-literature novelty check, explicit
hypothesis, feasibility protocol and accountable scientific decision. A negative
result is reportable. No plan, coding-agent opinion or benchmark rank alone establishes novelty.

## Delivery order and verification authority

Start Q01 and source-only Q03 preflight. Freeze the minimal contract, then deliver
Q02/Q03; Q04 and the analytic parts of Q05 can proceed under disjoint scopes. Q06
and method-specific Q07/Q08 follow acceptance, not elapsed time. Q09 collects evidence
throughout; R01–R03 remain separately gated. No calendar completion guarantee is made.

Use current repository commands, including focused Cargo/Pytest/binding tests,
`tox -e frontier-contract` where affected, and the final resource-guarded
`tox run-parallel -p 2` on the required interpreter. Record exact tree, environment,
commands and exit status; reuse only valid unchanged evidence. Current-head hosted
checks, security, signature and merge requirements remain mandatory. Do not merge
on the strength of this planning document or a static planning check.
