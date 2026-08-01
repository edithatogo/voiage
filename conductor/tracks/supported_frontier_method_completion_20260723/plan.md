# Implementation Plan

## Todo-to-issue-to-phase mapping

The following mapping is the canonical traceability boundary for the
experimental frontier entries in `todo.md`. Each phase is scoped to the
corresponding GitHub issue or native sub-issue; implementation status is
evidenced by the phase marker and receipt, while panel review, parity,
promotion, release, and issue closure remain separate gates.

| `todo.md` workstream | GitHub issue | Conductor phases |
| --- | --- | --- |
| Uncertainty-modelling value | #594; #774–#776 | dedicated track `uncertainty_modelling_value_20260801`: F594-1–F594-5 |
| Implementation-information decomposition | #593; #766–#768 | F593-1–F593-4 |
| Forecast and signal information VOI | #572; #759, #760, #762 | F572-1–F572-3 |
| Risk-sensitive and constrained VOI | #570; #757, #758, #761 | F570-1–F570-3 |
| Signed, social and strategic information value | #598; #783–#785 | F598-1–F598-5 |
| Value of Flexibility | #559 | F559-1–F559-4 |
| Deterministic sensitivity/scenario analysis | #556; #724–#728 | F556-1–F556-5 |
| Distribution-Family Information | #557; #731–#735 | F557-1–F557-5 |
| Portable qualitative VOI | #558; #738–#742 | F558-1–F558-5 |
| Finite additive MCDA information value | #560; #746–#750 | F560-1–F560-5 |
| Dependent information-source portfolio | #582; #763–#765 | dedicated track `information_source_portfolio_voi_20260801` |

The Rust-first programme and its remaining workstream tracks are governed by
the separate root track `rust_polyglot_voi_completion_20260723`; this table
covers the experimental frontier entries only.

## Phase 1 — Governance and contract reconciliation

- [x] **G1:** Verify the owning issue, native parent/children, Project 28,
  metadata, registry and cross-reference manifest. (AC-01) `e5462426`
- [x] **G2:** Reconcile existing repository artifacts and prior evidence
  without converting issue status into implementation evidence. (AC-01, AC-02)
  `b7836020`
- [x] **G2a:** Integrate the merged #619 scalar estimation-variance delivery,
  refresh exact-head assurance and retain vector covariance, scientific review,
  stable promotion, release and closure as separate gates. (AC-01–AC-03)
  `43ec5868`
- [x] **G3:** Freeze workstream estimands, contracts, maturity boundaries and
  explicit exclusions. (AC-02, AC-05, AC-06) `c622e859`
- [x] **G3a:** Reconcile VoC as #595's expected-utility/clairvoyant-policy
  alias or presentation, including affine-utility reduction conditions, and
  prohibit a duplicate numerical kernel. (AC-02, AC-08) `91770f6a`
- [x] **G4:** Run automated contract review and full Conductor validation.
  (AC-01, AC-07) `57c385bd`

## Phase 2 — Evidence before positive claims

- [x] **F557-1 / #731:** Freeze #557's distribution-family estimand, terminology,
  MoSCoW requirements, Mermaid flow, independent reference review and failing
  analytical/property/pathology tests. (AC-02, AC-03, AC-09) `b78120e1`
- [x] **F557-2 / #732:** Add strict versioned schemas, exact normative fixtures,
  provenance, comparability, estimator status and language dispositions.
  (AC-02–AC-06) `7976e49a`
- [x] **F557-3 / #733:** Implement the exact experimental Python evaluator with
  family-first conditioning, complete ties and a signed net-value result.
  (AC-02–AC-05) `ec39ed2a`
- [x] **F557-4 / #734:** Add exact-schema CLI, documentation, discovery and generated
  governance surfaces without claiming stable or polyglot support. (AC-04–AC-06)
  `f154cb67`
- [x] **F557-5 / #735:** Run independent implementation review, focused/full local and
  hosted exact-head assurance; reconcile #557 without closing scientific,
  stable-promotion, release or issue gates. (AC-03–AC-07) `2aa5d2eb`

- [x] **F558-1 / #738:** Freeze #558's portable non-cardinal assessment
  contract, planned v1.3.0 MoSCoW requirements, Mermaid flow, independent
  reference review and failing worked/disagreement/incomplete/adversarial
  tests. (AC-02, AC-03, AC-09) `be3631d1`
- [x] **F558-2 / #739:** Add strict versioned assessment, result, audit-event and
  rendering schemas; normative fixtures; provenance, redaction, AI-assistance,
  human-review and language dispositions. (AC-02–AC-06) `3bfea8b8`
- [x] **F558-3 / #740:** Implement the deterministic experimental Python
  validator and state transition workflow with ordinal priorities, complete
  ties, dissent, conflicts, missingness and immutable audit history, without
  cardinal pseudo-scoring. (AC-02–AC-05) `0bd57c17`
- [x] **F558-4 / #741:** Add installed contracts, Python/CLI deterministic JSON
  and accessible text rendering, documentation, examples and discovery without
  claiming quantitative VOI, stable or polyglot execution. (AC-04–AC-06) `4f9ae190`
- [x] **F558-5 / #742:** Run independent implementation, usability and
  accessibility review, focused/full local and hosted exact-head assurance;
  reconcile #558 while retaining practitioner/scientific, stable-promotion,
  parity, release and parent-closure gates. (AC-03–AC-07) `7bbf77cf`

- [x] **F560-1 / #747:** Freeze #560's finite compensatory additive MCDA
  information estimand, planned v1.3.0 MoSCoW requirements, Mermaid flow,
  adjacent-surface/reference review and failing analytical, invariant and
  pathology tests. (AC-02, AC-03, AC-09) `e2dacd0f`
- [x] **F560-2 / #746:** Add strict versioned input/result schemas, finite
  correlated-state fixtures, normalization/provenance assurance and explicit
  Rust/Python/R/Julia/Mojo dispositions. (AC-02–AC-06) `e2dacd0f`
- [x] **F560-3 / #749:** Implement the exact experimental Python additive-value
  evaluator with fixed normalization, complete ties, criterion/preference/joint
  information actions and no-double-counting diagnostics. (AC-02–AC-05) `e2dacd0f`
- [x] **F560-4 / #748:** Add installed Python/CLI execution, accessible plots,
  documentation, examples and discovery without claiming AHP, outranking,
  non-compensatory or imperfect-sample support. (AC-04–AC-06) `e2dacd0f`
- [x] **F560-5 / #750:** Run independent implementation/usability review,
  focused/full local and hosted exact-head assurance; reconcile #560 while
  retaining scientific, stable-promotion, parity, release and parent-closure
  gates. (AC-03–AC-07) Local review and assurance: `e2dacd0f`; PR #751 exact
  head `60297ba3` passed every hosted check, including Python 3.12–3.14,
  installed-wheel, aggregate and changed-branch coverage, then squash-merged as
  `e8aaba82`. VOP run `30684893440` and VOIAGE receiver run `30684980076`
  verified the canonical v1.3 projection without opening a drift PR.

- [x] **F594-1 / #774:** Freeze #594's exact finite EVIU/VSS estimand,
  planned v1.3.0 Must requirements, point-estimate comparator, shared-history
  nonanticipativity, information-acquisition boundary and DVSS/VMS disposition.
  (AC-02, AC-05, AC-09) `c04d4c33`
- [x] **F594-2 / #774:** Add strict schemas plus deterministic two-stage
  nonlinear, multistage, infeasible-recourse and pathology fixtures with
  independent finite references. (AC-02–AC-06) `c04d4c33`
- [x] **F594-3 / #775:** Implement the exact experimental Python EV/EEV/RP/WS,
  direction-aware VSS/EVIU/EVPI, complete ties, feasibility and policy audit.
  (AC-02–AC-05) `c04d4c33`
- [x] **F594-4 / #775:** Add deterministic API/CLI discovery, documentation and
  explicit Python/Rust/R/Julia/Mojo dispositions without stable claims.
  (AC-04–AC-06) `c04d4c33`
- [~] **F594-5 / #776:** Run independent implementation review, 100% changed
  branch coverage, full local and hosted exact-head assurance; retain
  scientific, stable-promotion, parity, release and parent-closure gates.
  Independent implementation review at signed commit `766e7b59` found no
  unresolved Critical, High or Medium findings. PR #798 exact head `aa5d9fd8`
  completed all 42 hosted checks with 38 successes and four governed skips;
  all six wheel contracts, aggregate coverage, CodeQL and security checks
  passed, and all four review threads were resolved before squash merge
  `c5adca8f`. Repository-delivery subissues #774–#776 may close. Independent
  scientific review, Rust/R/Julia parity, stable promotion, release, parent
  #594 closure and umbrella #318 closure remain pending.
  (AC-03–AC-07)

- [x] **F593-1 / #766:** Freeze #593's four-cell information/implementation
  estimands, v1.3.0 M25 requirements, Mermaid flow, primary-reference review,
  non-independence boundary and terminology-candidate status. (AC-02, AC-03,
  AC-09) `9cc93c63`
- [x] **F593-2 / #767:** Add strict portable v1 input/result schemas, an exact
  state-dependent normative fixture, current/specific/perfect and signal-
  dependent implementation contracts, population/time/cost semantics and
  explicit language dispositions. (AC-02–AC-06) `9cc93c63`
- [x] **F593-3 / #767:** Implement the exact experimental Python evaluator,
  current/perfect-information by current/perfect-implementation matrix, EVSIM,
  IA-EVSI, interaction identities, complete ties and CLI/API surfaces without
  an independence assumption. (AC-02–AC-05) `9cc93c63`
- [~] **F593-4 / #768:** Complete implementation review and the relevant
  subagent scientific-review panel,
  hosted exact-head and installed-wheel assurance, then reconcile #593 without
  claiming polyglot parity, stable promotion, release or parent closure.
  Independent repository implementation re-review passed with no open findings
  at signed review-artifact commit `f945f87b`, and the patch-equivalent rebased
  review boundary is `f52feb28`. PR #787 exact head `de31458b` passed all 42
  hosted checks (38 successes, one neutral aggregation and three governed
  skips), including installed-wheel and changed-coverage assurance, and both
  review threads were evidence-resolved before squash merge `20e0c606`.
  Named panel review, Rust/R/Julia parity, stable promotion, release,
  parent #593 closure and umbrella #318 closure remain pending.

- [x] **F596-1 / #777:** Freeze C18/M27's exact finite event-localized
  estimands, canonical policy-relative EUI density, explicitly signed centered
  diagnostic, strict schemas, normative fixture, complete ties, source review,
  exclusions and language dispositions. (AC-02–AC-06, AC-09) `8014c75f`
- [x] **F596-2 / #778:** Implement perfect event/complement value, the
  imperfect symmetric binary-channel accuracy curve, finite univariate and
  multivariate density atoms, integral assurance, modes/directions and
  deterministic Python/API/CLI execution. (AC-02–AC-05) `8014c75f`
- [x] **F596-3 / #778:** Add result-only accessible density and accuracy plots,
  installed discovery, documentation and governance projections without
  claiming continuous density estimation, monetary BPI or stable/polyglot
  execution. (AC-04–AC-06, AC-09) `8014c75f`
- [x] **F596-4 / #779:** Complete independent implementation and source review,
  focused/full local and hosted exact-head/wheel assurance; reconcile #596
  while retaining scientific approval, Rust/R/Julia parity, stable promotion,
  release and parent closure as separate gates. PR #804 exact head `e6835358`
  passed all 42 hosted checks (38 successes and four governed skips), including
  installed-wheel and 100% changed-branch assurance, with zero unresolved
  review threads before squash merge `e3a62eba`. (AC-03–AC-07, AC-09)
- [x] **F596-R1:** Preserve empty modes/directions when coordinate information
  value is zero, so tied zero-density atoms do not fabricate a direction of
  concern. `5964df04`
- [x] **F596-R2:** Remediate independent-review findings for executable
  input/result validation, true-max reference optimality, bounded raw-integral
  tolerance and explicit binary-channel symmetry assurance. A fresh
  independent reviewer remains required. `6d712f82`
- [x] **F596-R3:** Remediate the fresh result-integrity review by binding
  auditable partition evidence, reconstructing every event/channel/density
  action marginal, requiring a common reference policy and rejecting ungrouped
  coordinate atoms. Shared focused/oracle/static/governance assurance passed;
  hosted exact-head coverage and wheel evidence remain required. `893a2243`
- [x] **F596-R4:** Complete a third fresh independent review of both
  remediations. The review independently reproduced the normative values,
  exercised the previously bypassed result mutations, and found no unresolved
  Critical, High or Medium finding. Review record:
  `event-localized-information-final-review.md`. Hosted exact-head coverage,
  installed-wheel evidence and later scientific/parity/promotion/release gates
  remain pending.
- [x] **F596-R5:** Remediate exact-head hosted assurance failures without
  weakening the experimental contract. The repository-wide command registry,
  extension-policy projection, promotion checklist and inherited fixture hashes
  were synchronized; result/input fail-closed invariants, defensive numerical
  guards and both CLI output branches now have executable coverage. Focused
  local coverage reports 100% statements and branches for the event contract
  and evaluator. Replacement exact-head `e6835358` passed hosted coverage and
  every other required Action before merge.
  `e447c0e8`

- [x] **F572-1 / #760:** Freeze #572's C18/M23 forecast-signal decision
  estimand, strict input/result schemas, analytical newsvendor fixture,
  provenance, timing, cost, calibration and explicit language dispositions.
  (AC-02–AC-06, AC-09) `8c9a1885`
- [x] **F572-2 / #759:** Implement the exact finite experimental Python/API/CLI
  evaluator with baseline/oracle/deployed policies, signed value, calibration
  loss, regret, maximum price and no-skill/perfect/miscalibrated/late/stale
  limits. (AC-02–AC-05) `8c9a1885`
- [~] **F572-3 / #762:** Complete the relevant subagent scientific/implementation
  review panel,
  hosted exact-head/full-suite assurance and canonical C18 projection
  reconciliation while retaining parity, stable-promotion, release and parent
  closure as separate gates. The independent implementation review in
  `forecast-signal-implementation-review.md` found no blocking issue after
  bounded strictness and hosted-regression repairs. Live C18/M23 projection was
  verified from merged VOP PRs #69 and #70. PR #770 exact head `c110706c`
  passed all 42 hosted checks (38 successes, one neutral aggregation and three
  governed skips), including 223/223 changed lines and 57/57 changed branches,
  before squash merge `4657f94e`. Panel review, parity,
  stable-promotion, release and parent closure remain pending.
  (AC-03–AC-07, AC-09)

- [x] **F570-1 / #757:** Freeze C18/M22's finite risk-sensitive constrained
  perfect-information estimand; add strict v1.3.0 experimental input/result
  schemas, a fully enumerable normative fixture, risk-neutral/CVaR/regret
  reductions, pathological tests, provenance and explicit exclusions.
  (AC-02, AC-03, AC-09) `2da92322`
- [x] **F570-2 / #758:** Implement exact bounded Python enumeration of matched
  current and perfect-state feasible policy problems with expected
  value/declared utility, lower-tail CVaR, minimax regret,
  deterministic/chance constraints, complete ties, infeasibility, gross/net
  value, switches, diagnostics, CLI, documentation and discovery. (AC-02–AC-06)
  `2da92322`
- [~] **F570-3 / #761:** Retain local exact-fixture, type, lint, registry and
  Conductor assurance plus honest Python/Rust/R/Julia/Mojo dispositions;
  complete hosted exact-head/wheel and the relevant subagent review panel before
  reconciling #570. Local implementation evidence: `2da92322`; independent
  implementation review and bounded remediation: `58a119bc`; hosted lint/type
  remediation: `18747c8b`; changed-branch coverage remediation: `bfbfcea3`;
  review record:
  `risk-sensitive-constrained-voi-implementation-review.md`. PR #769 exact
  head `f513416f` passed all hosted checks with 100% changed-line and
  changed-branch coverage before squash merge `c25f3234`. Scientific, parity,
  stable-promotion, release and parent-closure gates remain pending.
  (AC-03–AC-07, AC-09)

- [x] **F598-1 / #783:** Freeze C18/M29's complete finite joint-world law,
  named agents/roles/topology/designs, nonanticipative policy catalogs, signed
  estimands, welfare/cardinal-comparability declaration, selective-sharing
  comparators, strict Blackwell applicability and adjacent-method exclusions.
  (AC-02, AC-03, AC-09) `f04b1627`
- [x] **F598-2 / #783:** Add strict versioned input/result schemas, the
  Li-Pozzi-inspired harmful-private/positive-social fixture, provenance,
  complete ties, transfer/cost ledgers and language dispositions. (AC-02–AC-06)
  `f04b1627`
- [x] **F598-3 / #784:** Implement exact experimental Python/API/CLI execution
  for centralized, fixed, declared-response and receipt-verified finite-
  equilibrium catalogs, signed agent/role/social values, selective sharing,
  harm, avoidance, switches, winners/losers and rights receipts without
  clipping. (AC-02–AC-06) `f04b1627`
- [x] **F598-4 / #784:** Add deterministic discovery, documentation, registry,
  export and governance surfaces while retaining persuasion, mechanism design,
  rational inattention and general game solving as adjacent. (AC-04–AC-06)
  `f04b1627`
- [~] **F598-5 / #785:** Run independent implementation/scientific review,
  changed-line/branch and full local assurance, hosted exact-head/wheel checks,
  then reconcile #598 without claiming parity, stable promotion, release or
  parent closure. The first implementation reviewer found and remediated
  portable derived-result and Blackwell-applicability defects in
  `signed-social-information-implementation-review.md`. A fresh independent
  re-review passed at `d7d569b2` with no remaining Critical, High or Medium
  findings; hosted exact-head/wheel checks and merge remain pending.
  (AC-03–AC-07, AC-09)

- [x] **F598-R1 / #785 — Review Fixes:** Bind portable agent roles and exact
  evaluated identifiers; recompute signed roles, policy/design ties, optimum,
  diagnostics, Blackwell values and assurance counts; exclude infeasible and
  transfer/cost-bearing comparisons from Blackwell applicability; retain
  negative social values without clipping. (AC-03–AC-07) `da119389`

- [x] **F598-R2 / #785 — Hosted extension-policy remediation:** Declare the
  signed/social experimental Python method in its frontier capability contract
  and refresh the governed capability artifact hash. The exact extension-policy
  regression, deterministic export/hash verification, frontier contract,
  GitHub cross-reference and full 148-track Conductor validators pass locally.
  A replacement exact-head hosted run remains mandatory before merge.
  `0a7180ed`

- [x] **F559-1:** Add failing analytical, invariant, permutation, chronology,
  comparability and double-counting tests plus independent reference review for
  #559 Value of Flexibility. (AC-03) `6fd474b1`
- [x] **F559-2:** Add the versioned VoF schema, normative fixture, runtime
  fixture-conformance check, provenance and explicit language dispositions.
  (AC-02–AC-06) `500adfd0`
- [x] **F559-3:** Implement the corrected flexible-versus-commitment evaluator,
  compatibility view, public Python/CLI surface and documentation without
  duplicating a numerical kernel or claiming information value. (AC-02–AC-05)
  `02e182bd`
- [x] **F559-4:** Run independent implementation review, focused repository
  checks and hosted checks; then reconcile #559 without closing scientific,
  stable-promotion, release or issue gates. (AC-03–AC-07) `3fa33653`
- [x] **F556-1 / #724:** Freeze #556 DSA MoSCoW requirements, Mermaid flow,
  independent reference review and failing analytical, metamorphic and
  pathology tests. (AC-02, AC-03, AC-09) `5859f7a1`
- [x] **F556-2 / #725:** Add strict versioned DSA schemas, one-way/two-way/scenario
  fixtures, provenance, reproducibility and language dispositions. (AC-02–AC-06)
  `93e357cb`
- [x] **F556-3 / #726:** Implement the shared Python DSA evaluator, normalized-record
  adapter, deterministic switch/tie policy and experimental public API.
  (AC-02–AC-05) `f72d3c3a`
- [x] **F556-4 / #727:** Add exact-schema CLI, tornado plotting, documentation,
  discovery and generated-surface reconciliation. (AC-04–AC-06) `33135403`
- [x] **F556-5 / #728:** Run independent implementation review, focused/full local
  checks and hosted exact-head checks; reconcile #556 without claiming stable
  promotion, scientific approval, release, merge or closure. (AC-03–AC-07)
  `4c705ce6`
- [ ] **G5:** Add failing conformance, reference, property and pathological
  tests, or the corresponding reproducible review protocol. (AC-03)
- [ ] **G6:** Add versioned schemas, fixtures, diagnostics and provenance
  contracts required by the accepted scope. (AC-02, AC-03)
- [ ] **G7:** Record rights, privacy, scientific, practitioner and external
  evidence gates that apply to this workstream. (AC-05, AC-06)
- [ ] **G8:** Run an independent evidence and boundary review. (AC-03, AC-07)

## Phase 3 — Delivery or reviewed exclusion

- [ ] **G9:** Implement each accepted repository-owned capability or record a
  reviewed exclusion with migration guidance. (AC-02, AC-06)
- [ ] **G10:** Add Rust/Python/R/Julia/Mojo dispositions and installed
  shared-fixture evidence where executable surfaces are advertised. (AC-04)
- [ ] **G11:** Add documentation, examples, generated surfaces and capability
  discovery that match the evidenced maturity state. (AC-05)
- [ ] **G12:** Run automated implementation review, focused validation and the
  repository harness. (AC-03–AC-07)

## Phase 4 — Programme and hosted closeout

- [ ] **G13:** Reconcile child-issue results, roadmap, todo, registries,
  v1.2.0 MoSCoW requirements, Mermaid design, canonical C16, release targets
  and remaining external gates. (AC-01, AC-05, AC-06, AC-08, AC-09)
- [ ] **G14:** Run final full local validation and hosted required checks.
  (AC-07)
- [ ] **G15:** Record repository completion separately from merge, release,
  publication, registry acceptance and issue closure. (AC-02, AC-07)
