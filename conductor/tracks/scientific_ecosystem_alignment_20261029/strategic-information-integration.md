# Strategic-information integration implementation plan

Date: 8 November 2026. Status: approved roadmap direction; all implementation and
scientific acceptance remain pending. This supplements the existing frontier
programme and optional simulation-outcome track, not a new runtime registry.

**Goal:** Value evidence when strategic responses matter, without making Voiage a
game solver or blocking its ordinary VOI methods and initial software release.
**Architecture:** Two Rust numerical authorities: the game packages own supported
response solving; Voiage owns information-value arithmetic. Applications construct
and value games. UOGTO owns semantics; Kairos is optional for dynamic execution.
**Tech stack:** Existing Rust domain/numerics, thin language bindings, paired
Arrow/Parquet outcomes and versioned metadata. No new dependencies or pins here.
**Spec and dependencies:** This document and the adjacent
[planning manifest](strategic-information-manifest.json). Preserve the existing
[Q01–Q09/R01–R03 plan](frontier_20261108/plan.md) and its completion requirements.

## Current direction and alternatives

Use the existing finite signed-social-information contract and paired-outcome
boundary. Do not create a universal schema or duplicate Nash/CE/CFR algorithms.
A monolithic game/simulation/VOI engine would violate ownership; exchanging only
unlabelled payoff matrices would lose the information and selection assumptions.
The chosen approach is an optional typed adapter or versioned artifact exchange.
A callback request cycle must not become a Cargo dependency cycle.

[Paignia T17](https://github.com/edithatogo/paignia/tree/main/conductor/tracks/t17_game_runtime_ecosystem_20261107)
is the canonical provider track. The historical seed came from the reimbursement
incubator at f6c1c87b; its initial transfer is merged at
[12b22ce](https://github.com/edithatogo/paignia/commit/12b22ce7571b9c590405bf53c2c1c5862b49cd64).
Package, method and installed-consumer qualification remain pending. A bounded
static release can be qualified by two real application cases, including
reimbursement plus Innovate; CareOps is another candidate. Both can be static.
Do not restore the older mandatory CareOps/Innovate pair, require every advanced
method, or let a Voiage wrapper stand in for a second real application. This
planning supplement changes no package release or provider pin.

Preserve three distinct claims: fixed-policy valued outcomes; equilibrium-aware
information comparisons; and separately gated information/mechanism-design research.
An ontology profile or completed adapter is not evidence that a solver supports
Bayesian games, Bayes correlated equilibrium or another unimplemented concept.

## Ownership and release independence

| Owner | Owns | Excludes |
| --- | --- | --- |
| UOGTO | Semantic identifiers and declared executable profiles | Numerical correctness or equilibrium existence certification |
| Rust game core/solve | Representations, supported solution concepts, policies and solver diagnostics | VOI and application-specific economics |
| Kairos / optional adapter | Native time/events/state/RNG, causal execution and replay | Mandatory static solving or a second scheduler |
| Applications / valuation providers | Players, priors, incentives, units, perspectives and welfare assumptions | Copied generic solvers or VOI formulas |
| Voiage | Signed information-value estimands, study comparison and research choice | Generic game solving or implicit equilibrium selection |

Default Voiage must not require the game library or Kairos. Static game packages
must have no direct or transitive Kairos dependency. Voiage's bounded numerical-job
lifecycle is not Kairos simulation time. Preserve current language support,
offline R builds, ABI/API commitments and the HPC deferral. No non-Rust engine
fallback, external submission or Hub upload is authorised by this planning change.

## Contracts to map before changing schemas

Reuse existing types and extensions; first produce a field-level compatibility
map, then freeze exact signatures and versioning in the existing contract owner.
Carry game specification/instance, player/role, policy, session and trace IDs;
analyst uncertainty draw, hidden state/type, signal, replication and solver-run IDs;
public/private recipients, timing, likelihood, commitment and disclosure assumptions;
solution concept, representation, algorithm/version, termination, tolerance and
appropriate feasibility/incentive/best-response diagnostics; selection rule or
catalogue coverage; actor payoffs, policymaker objective and any social aggregator;
units, weights, population, horizon, discounting, transfers and source hashes.

An analyst's sampled state is not automatically visible to players. Do not solve
each hidden state as a complete-information game and average the results. Policies
must be nonanticipative for the declared observations/information sets. Equal row
counts or equal seeds do not prove paired worlds: preserve explicit joint coupling.

Keep solver analysis of unchosen branches separate from causal simulation traces.
A selected equilibrium is not a probability; restart frequencies are not behavioural
frequencies without a model. Multiple equilibria require a declared selection rule
or reported ranges/sensitivity; an incomplete catalogue is not exhaustive.

Preserve signed actor and social values. The legacy strategic_behavior.py helper
is a fixture-backed, thresholded additive scenario score with nonnegative clipping,
not a general equilibrium-aware VOI estimator. Inventory its compatibility promises
and propose a versioned warning/deprecation path; do not silently change results.
The experimental signed-social-information contract is the preferred foundation,
but its existing Python implementation is not proof of Rust or binding completion.
Do not sum individually rescalable utilities without declared cardinal comparability
and welfare weights, or double-count transfers as social resource costs.

Report Monte Carlo, surrogate, numerical-solver and behavioural/selection uncertainty
separately. Small residuals need not bound value error near equilibrium switches.
Solver failures are neither economic zeroes nor observations to discard. Cache and
warm-start identity includes game/information regime, solution concept, selection,
valuation, algorithm and relevant tolerances; test path-dependent selection.

### Field map for reimbursement T17.11

| Boundary | Current source fields and handoff |
| --- | --- |
| Game and actor identity | Paignia's `GameSpecification.id` identifies the description. Its result identity record separately carries the specification, instance, solver run, and optional session, policy artifact, and solver trace. Map the declared player to `agents[].agent_id` and `policies[].decision_agent_id` explicitly. Preserve the original identifiers and their roles in the producer receipt. UOGTO uses full IRIs, while signed-information v1 uses local identifier strings with a restricted syntax. A provider adapter needs a versioned one-to-one mapping. |
| Observations and disclosure | UOGTO declares the player and decision nodes in each information set. Paignia admits the complete partition, ordered legal actions, and that player's own recall. The application supplies `worlds[].signal`, `policies[].decisions[].observation`, topology and design recipients, and the purpose-bound rights receipts. Group equally observed worlds under one policy action. The sharing record determines which decision agents can use it. |
| Execution and coverage | Carry the provider's declared representation, information structure, solution concept, algorithm version, status, tolerance, selection rule, and catalogue claim. Voiage separately declares `designs[].selection_mode`, `selector`, the chosen policy, and any `equilibrium_receipt`. The signed-information v1 receipt accepts `declared_finite_nash` and binds the selected policy to the evaluated catalog and the receipt. Use the provider's scope and completeness evidence to support a coverage claim. |
| Joint draws and repetitions | Preserve producer draw namespace, scenario, strategy, parameter draw, replication, perspective, and subgroup where defined. The installed paired-draw helper binds `draw_id`, `strategy_id`, `perspective`, `subgroup_id` and `net_benefit` with explicit weights when supplied. Retain the wider scenario and replication keys in the application handoff. The application supplies the finite `world_id`, probability and signal law. Use the producer's declared coupling for the join. |
| Payoffs and valuation | The application supplies `worlds[].action_utilities`, the value unit, welfare weights and cardinal comparability, perspective, and provenance. Retain the declared population, horizon, discounting, and currency or price year where applicable. Carry `designs[].transfers` and `costs` independently, with the declared welfare ledger stage. Voiage computes signed actor, role and welfare values. |
| Trace and provenance | Preserve the native result artifact, source and version hashes, and independent evidence. Retain the solver diagnostic trace ID separately from optional execution session context and the application's realized trace. The signed-information input declares the world-law, utility, policy, and rights sources. |

The provider evidence is tied to Paignia's result and information-set contracts
and UOGTO's executable declaration. [Paignia result contract](https://github.com/edithatogo/paignia/blob/a09c3f8359363b228154c9e07af20a36bf0dcd4e/docs/architecture/game-result-contract-v1.md) · [Paignia information-set contract](https://github.com/edithatogo/paignia/blob/a09c3f8359363b228154c9e07af20a36bf0dcd4e/docs/architecture/game-information-contract-v1.md) · [UOGTO executable declaration](https://github.com/edithatogo/UOGTO/blob/3000c231b2c102e18bf7c3294bec498f00f1976c/docs/contracts/executable-information-set-profile-v1.md)

The current Voiage inputs are the experimental finite-world schema and the
experimental paired-draw helper. The simulation-outcome plan tracks the wider
key contract. [Signed-information input](../../../specs/frontier/signed-social-information/v1/schemas/input.schema.json) · [Paired draws](../../../voiage/experimental/paired_draws.py) · [Simulation-outcome specification](../simulation_outcome_boundary_20261107/spec.md)

## Pending bounded work

Each implementation slice first binds exact current types/tests/tolerances, writes
and observes a failing witness, makes the smallest change, runs affected checks and
records a scoped review. The following are acceptance obligations, not executed tests.

### SI01 — semantic reconciliation (Q01/Q03)

- [ ] Inspect voiage/methods/strategic_behavior.py, the signed-social-information
  methods/contracts, specs/frontier/signed-social-information/v1 and the paired
  simulation-outcome contract. Keep legacy score and strategic estimand identities distinct.
- [ ] Map information/selection/valuation fields to T17 and UOGTO. Reject unknown
  regimes, hidden-state leakage, unspecified selection and incomplete-coverage claims.
- [ ] Review proposed schema/API changes with their existing owners before coding;
  add distinguishing cases SI-V03/04/06 to existing contract tests.

### SI02 — Rust-authoritative finite signed evaluation (Q02/Q03/Q07)

- [ ] After SI01 and Q02/Q03, qualify the finite signed evaluator in existing Rust
  domain/numerics, without a game dependency. Proposed numerical module:
  rust/crates/voiage-numerics/src/signed_social_information.rs.
- [ ] Start with SI-V01/05 and independent equations, not a translation-only oracle.
  Preserve weights, feasible policies, signed differences, units and transfer accounting.
- [ ] Verify Rust-only execution and installed parity for each exposed binding;
  unsupported methods/bindings stay explicit. Historical Python compatibility is versioned.

The finite v1 kernel evaluates admitted policy payoff tables, transfers, costs and
comparator-relative signed values in Rust. The Python and CLI entry points use
that kernel after admitting world laws, observations and rights receipts. The
standalone Rust interface accepts indexed numerical inputs. The portable v1
schema, normative result and binding dispositions apply to the full contract.

SI-V01 checks values of 50 under equal weights and 10 under a 0.9/0.1 prior.
SI-V05 checks agent values of −2, 0 and 4 and declared social value of 2, with
separate controls for transfers, costs and welfare stages. The historical
[Python evaluator](https://github.com/edithatogo/voiage/blob/eafe1d41a896aca630d7ba29aca980e1047f614a/voiage/methods/signed_social_information.py)
records the previous execution path.

### SI03 — optional provider admission (Q08)

- [ ] After SI01/SI02 and the paired-outcome contract, consume only the qualified
  T17 representation/method. T17.01–03 supply bounded static packaging contracts.
- [ ] Require T17.05 for mixed normal-form claims, T17.06 for general CE, and
  T17.07 plus T17.17 for the supported CFR/information-set profile. None alone
  qualifies Bayesian games or Bayes correlated equilibrium; those require separate contracts.
- [ ] Dynamic claims additionally require T17.08 and relevant T17.13–16 evidence
  for conversion, lawful observations, RNG isolation and policy/native replay.
- [ ] Proposed tests/test_strategic_game_provider_contract.py rejects stale hashes,
  wrong actor/draw identities, unknown concepts, nonconvergence and invalid catalogues.
  Cover SI-V02/03/04/06/07, cold/warm selection and clear failure dispositions.
- [ ] Test installed Rust consumers outside provider checkouts and resolved feature
  closures. No accidental default dependencies or silent host-language fallback.

### SI04 — application and comparative evaluation (Q08; R01/R02/R03)

- [ ] Begin with a tiny exactly solvable static case; then an application-owned
  reimbursement evidence/price-response or evidence-contingent-contract pilot.
  Innovate adoption/incentives and CareOps information/capacity games follow as needed.
- [ ] Require Q05/Q06 only when claiming EVSI/ENBS/design selection; require native
  replay only for dynamic claims. Keep synthetic and empirical validation distinct.
- [ ] Compare fixed responses, recomputed responses, research-only and disclosure-only
  designs, with no-study/avoidance options only where genuinely feasible.
- [ ] Score signed actor/welfare outcomes, wrong-design probability, decision regret,
  value error, runtime, memory and failures. Freeze confirmatory cases and assumptions.
- [ ] R01 may allocate computation across sampling and response solving with justified
  stopping; R02 examines joint research/implementation/capacity; R03 separates learning
  about behaviour from changing information/incentives. Each needs its own novelty review.

## Reference cases extending existing verification families

| Case | Required reference or failure |
| --- | --- |
| SI-V01 | One-player rows [[0,100],[100,0]]: equal-weight EVPI 50; weights 0.9/0.1 give 10. Unsupported weighted routes reject. |
| SI-V02 | Matching pennies A=[[1,-1],[-1,1]], B=-A: uniform mixed equilibrium, value 0. Pure-only absence is not absence of all equilibria. |
| SI-V03 | Coordination AA=(3,1), BB=(1,3), off-diagonal=(0,0): two pure equilibria and mixed probabilities row A=3/4, column A=1/4. Selection and catalogue scope are explicit. |
| SI-V04 | Hidden states with identical permitted observations cannot select different deterministic actions. Private signals do not become public on export. |
| SI-V05 | Preserve the existing constructed negative-private/positive-social fixture and accounting assumptions; add an independently verified endogenous game before claiming equilibrium harm. |
| SI-V06 | Stale receipts, unknown schemas/concepts, nonconvergence, broken pairings and incomplete coverage fail distinctly rather than becoming zeroes. |
| SI-V07 | Reuse T17-owned S01–S04 numerical obligations; licensed pinned Gambit/OpenSpiel references stay isolated, never runtime dependencies. |
| SI-V08 | Installed source-to-decision application with exact artifacts, independent small reference and optional qualified native replay. |

Extend D01/D10/D11/D13/D16/D18 rather than duplicating the benchmark registry.
Pin case definitions, reference errors, licences, sources and algorithm versions.
Distinguish same-estimator implementation, matched-accuracy algorithm and same-workflow
hardware comparisons. No empirical forecasting fit is proof of causal effectiveness.
Rights-cleared artifacts may use the already proposed owner-controlled benchmark
collection; no resource is created or published here.

## Delivery and publication

Keep SI dependencies one-way into the existing programme: do not add SI or game
completion to Q09, the initial software paper, static package eligibility or native
runtime acceptance. Preserve pyOpenSci/eligible-JOSS and the separate MDM evaluation;
a later strategic-information methods paper needs a precise new contribution,
primary-literature and venue review, and independent reproduction.

This is a follow-on branch to PR #1285, not a rewrite of its pending merge. The
adjacent manifest records inspected revisions, not new ecosystem pins. Companion
repo plans retain T17, I07/G01–G04, K01, semantic-track and R76.5 owners. No runtime,
public schema, task completion, lease takeover or scientific promotion is made.
Full local and exact-head hosted gates remain required; static plan checks are not
Cargo, tox, native replay, binding, numerical or journal acceptance evidence.
