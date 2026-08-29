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
| Study-design efficiency and experiment portfolios | #571; #680–#682 | dedicated track `study_design_efficiency_20260727`: S1–S19 |
| Expected-utility information pricing and VoC presentation | #595; #694–#697 | dedicated track `risk_adjusted_information_pricing_20260731`: U1–U17 |
| Signed, social and strategic information value | #598; #783–#785 | F598-1–F598-5 |
| Value of Flexibility | #559 | F559-1–F559-4 |
| Deterministic sensitivity/scenario analysis | #556; #724–#728 | F556-1–F556-5 |
| Distribution-Family Information | #557; #731–#735 | F557-1–F557-5 |
| Portable qualitative VOI | #558; #738–#742 | F558-1–F558-5 |
| Finite additive MCDA information value | #560; #746–#750 | F560-1–F560-5 |
| Belief-state sequential information value | #597; #780–#782 | F597-1–F597-R5 |
| Signed/social information value | #598; #783–#785 | planned C18/M29; delivery remains separate |
| Dependent information-source portfolio | #582; #763–#765 | dedicated track `information_source_portfolio_voi_20260801` |
| Static/dynamic heterogeneity value | #599; #786, #788, #789 | F599-1–F599-4 |
| Outcome-conditional sample-information value | #600; #790–#792 | F600-1–F600-4 |

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
- [x] **G2a-R:** Integrate the dedicated #619 predictive-weighting and replay-
  provenance remediation after fresh exact-head hosted assurance and merge.
  PR #837 exact head `076a29075e839e3cad49d0487dff0c4e2639845f`
  completed 65 terminal checks (60 successes, four governed skips and one
  neutral conclusion), with zero review threads, before squash merge
  `366186b358abd775bea5fd2440d7e0ececb3ebaa` at 2026-08-01T16:58:05Z.
  Scientific review, vector covariance, stable promotion, release and parent
  closure remain separate gates. (AC-01–AC-03, AC-07)
- [x] **G2b:** Reconcile #571's final PR #679 exact-head assurance and merge
  receipt so only delivery subissues #680–#682 become closure-eligible; retain
  scientific review, Rust/R/Julia parity, stable promotion, release, parent
  #571 closure and umbrella #318 closure as separate gates. (AC-01–AC-03)
  `75458e79`
- [x] **G2c:** Reconcile #595's final PR #712 exact-head assurance and merge
  receipt so only delivery subissues #694–#697 become closure-eligible while
  VoC remains a presentation/delegating alias; retain scientific review,
  Rust/R/Julia parity, stable promotion, release, parent #595 closure and
  umbrella #318 closure as separate gates. (AC-01–AC-03, AC-08) `ac6b33c6`
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
- **Migrated:** **F594-5 / #776:** Run independent implementation review, 100% changed
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
- **Migrated:** **F593-4 / #768:** Complete implementation review and the relevant
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

- [x] **F600-1 / #790:** Freeze C18/M31's exact finite outcome-conditional
  estimands, Equation 10 weighted population standard deviation, expectation-
  only tower identities, tie-aware `rVSI0` boundary, primary-reference review,
  strict schemas, normative fixture and explicit exclusions. (AC-02, AC-03,
  AC-09) `3080c1b4`
- [x] **F600-2 / #791:** Implement deterministic experimental Python/API/CLI
  execution for predictive/posterior outcome laws, `delta-EV_x`, `VSI_x`, EVSI,
  weighted `sigma-VSI`, `rVSI_delta`, quantiles/tails, cost placement, complete
  ties and prospective/retrospective presentation. (AC-02–AC-06) `6548dda7`
- [x] **F600-3 / #792:** Add portable input/result assurance, independent result
  reconstruction, probability/Bayes/identity/calibration diagnostics,
  adversarial/tie/permutation/pathology tests, documentation, discovery,
  fixture manifests and Python/Rust/R/Julia/Mojo dispositions. (AC-03–AC-06)
  `6548dda7`; focused contract/evaluator coverage is 100% statements and
  branches, with frontier, cross-reference, static and full Conductor
  validation passing locally.
- [x] **F600-4 / #792:** Run a fresh independent implementation review and
  hosted exact-head/installed-wheel assurance. Retain continuous-outcome and
  fitted-estimator validation, independent scientific review, risk-sensitive
  composition, Rust/R/Julia parity, stable promotion, release, parent #600 and
  umbrella #318 closure as separate gates. Fresh independent implementation
  review found and fixed the value-unit invariance defect at signed commit
  `aaf77aaf`. A second independent PR review found a High probability-
  normalization defect and a Medium reference-policy wording inconsistency;
  signed remediation `6eac2ba2` normalizes accepted near-unit vectors with
  explicit assurance and aligns M31/schema/README wording on the exact
  reference extremum. Final independent re-review of exact head `027f59d2`
  found no remaining Critical, High or Medium finding after the inclusive
  probability-boundary remediation. PR #831 exact implementation head
  `eb5a201d82350631dc3ba0b636dfdf43563ea64f` completed all 42 terminal-
  allowed hosted checks with 38 successes, three governed skips and one neutral
  CodeQL aggregation. All five CodeQL review threads were resolved before
  squash merge `ac1d31bf900c3ee6e817047202cae4229918d48f`. Delivery subissues
  #790, #791 and #792 may close. Continuous-outcome and fitted-estimator
  validation, independent scientific review, risk-sensitive composition,
  Rust/R/Julia parity, stable promotion, release, parent #600 closure and
  umbrella #318 closure remain pending. (AC-03–AC-07, AC-09)

- [x] **F572-1 / #760:** Freeze #572's C18/M23 forecast-signal decision
  estimand, strict input/result schemas, analytical newsvendor fixture,
  provenance, timing, cost, calibration and explicit language dispositions.
  (AC-02–AC-06, AC-09) `8c9a1885`
- [x] **F572-2 / #759:** Implement the exact finite experimental Python/API/CLI
  evaluator with baseline/oracle/deployed policies, signed value, calibration
  loss, regret, maximum price and no-skill/perfect/miscalibrated/late/stale
  limits. (AC-02–AC-05) `8c9a1885`
- **Migrated:** **F572-3 / #762:** Complete the relevant subagent scientific/implementation
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
- **Migrated:** **F570-3 / #761:** Retain local exact-fixture, type, lint, registry and
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

- [x] **F599-1 / #786:** Freeze #599's M30 four-value static/dynamic
  heterogeneity estimand, optional EVSI extension, planned v1.3.0 Should
  requirements, Mermaid flow and primary-reference boundary. (AC-02, AC-03,
  AC-09) `1a075383`
- [x] **F599-2 / #786:** Add strict v1 schemas, deterministic opposing
  static/dynamic fixture, declared subgroup selection, weights, eligibility,
  fairness/privacy, research model, provenance and language dispositions.
  (AC-02–AC-06) `1a075383`
- [x] **F599-3 / #788:** Implement exact experimental Python/API/CLI execution
  for `C0`, `Cf`, `P0`, `Pf`, static/dynamic value, subgroup EVPI, optional
  population/subgroup EVSI, complete ties and exact identities without
  replacing the stable heterogeneity helper. (AC-02–AC-06)
  `1a075383`
- [x] **F599-4 / #789:** Run independent implementation/scientific review,
  complete changed-branch, local and hosted exact-head assurance; retain
  scientific validity, selection-bias/sparse-subgroup review, Rust/R/Julia
  parity, stable promotion, release, parent #599 and umbrella #318 closure as
  separate gates. Independent review found and remediated model-unbound result
  assurance in `4b5d8d08`; the final review record is
  `heterogeneity-value-implementation-review.md`. Local focused, mutation,
  100% changed statement/branch, static, frontier, cross-reference and full
  Conductor gates pass. PR #809 exact implementation head
  `b0fc8db75796ffac9e66720ab45fdcf341c0b516` completed all 42 hosted checks
  with 38 successes, three governed skips and one neutral CodeQL aggregation;
  there were zero review threads before squash merge
  `1a37526af0ee87acc57dd14a629eb52aef2e182c`. Delivery subissues #786, #788
  and #789 may close. Scientific review, selection-bias and sparse-subgroup
  validity review, Rust/R/Julia parity, stable promotion, release, parent #599
  closure and umbrella #318 closure remain pending. (AC-03–AC-07)

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
- [x] **F598-5 / #785:** Run independent implementation/scientific review,
  changed-line/branch and full local assurance, hosted exact-head/wheel checks,
  and merge the bounded experimental delivery without claiming scientific
  approval, parity, stable promotion, release or parent closure. The first
  implementation reviewer found and remediated
  portable derived-result and Blackwell-applicability defects in
  `signed-social-information-implementation-review.md`. A fresh independent
  re-review passed at `d7d569b2` with no remaining Critical, High or Medium
  findings. PR #808 exact implementation head
  `4d121b29bb50492bcc84b1cdfa6fb46df9e5e51c` completed all 42 hosted checks
  with 38 successes, three governed skips and one neutral CodeQL aggregation;
  all 10 review threads were resolved before squash merge
  `d649c344ef2493abe445fb9e3ef20da89c53fb75`. Delivery subissues #783–#785 may
  close. Scientific review, Rust/R/Julia parity, stable promotion, release,
  parent #598 closure and umbrella #318 closure remain pending.
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
  The replacement exact-head hosted run passed before merge. `0a7180ed`

- [x] **F598-R3 / #785 — Hosted changed-branch remediation:** Exercise the
  signed/social CLI's non-object rejection and human-readable output receipt
  branches without changing runtime behavior or weakening coverage policy. The
  hosted-equivalent full suite passes with 3627 tests, 94.14 percent aggregate
  coverage, 444/444 changed executable lines and 198/198 changed branches. A
  replacement exact-head hosted run passed before merge. `7d68ce19`

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
- [x] **G5:** Map each accepted family to its conformance, reference, property
  and pathological tests or reproducible review protocol in
  `g5-g13-evidence-map.json`. (AC-03) `786b5229`
- [x] **G6:** Map each family to its versioned schemas, fixtures, diagnostics
  and provenance contracts. (AC-02, AC-03) `786b5229`
- [x] **G7:** Record the rights, privacy, scientific, practitioner and external
  gates without promoting repository evidence into external approval.
  (AC-05, AC-06) `786b5229`
- [x] **G8:** Complete genuine independent evidence and boundary review for
  every family. #571's fresh independent review found and drove remediation of
  probability-tolerance and accessible tie/uncertainty-plot gaps; #595's
  independent phase reviews and #619's fresh exact-commit remediation re-review
  report no remaining Critical, High or Medium finding. This engineering-
  evidence gate does not satisfy later scientific/design/classification review.
  (AC-03, AC-07)
- [x] **G8-R1 — Review Fixes:** Align complete/partial selection-probability
  validation with declared absolute plus relative tolerance and expose complete
  tied-optimum and selection-uncertainty availability states in accessible COSS
  plots, with under/over-tolerance and rendering regressions. Fresh independent
  re-review is recorded in `study-design-efficiency-independent-review-20260802.md`.
  (AC-03, AC-07) `b0078969`

## Phase 3 — Delivery or reviewed exclusion

- [x] **F597-1 / #780:** Freeze the exact finite belief-MDP contract, strict
  schemas, normative nonmyopic intervention fixture, chronology, estimands,
  null-sensor reduction and complete-tie policy. (AC-02, AC-03, AC-09)
  `bdcb78b1`
- [x] **F597-2 / #781:** Implement the experimental Python Bellman evaluator,
  public API and CLI with closed-loop/no-information comparators, gross/net,
  myopic/nonmyopic, conditional-sensing, policy-tree, regret and horizon
  results. (AC-03–AC-06) `bdcb78b1`
- [x] **F597-3 / #782:** Record posterior-martingale, exact-bound,
  action-dependent-learning and dual-control diagnostics, explicit language
  dispositions, independent implementation review, full local validation and
  hosted exact-head assurance without claiming stable or scientific closure.
  Initial review found and remediated false dual-control classification,
  unbounded exact-enumeration work, open nested result contracts and
  inconsistent probability tolerance. The remediator's engineering record is
  `belief-state-information-remediation-review.md`. The third fresh review
  found and remediated residual result-assurance drift, recorded in
  `belief-state-information-third-review-remediation.md`. The fourth fresh
  review found and remediated remaining model-unbound assurance claims in
  `belief-state-information-fourth-review-remediation.md`. The fifth fresh
  independent review approved the bounded remediation in
  `belief-state-information-fifth-review.md`. PR #807 exact head
  `35cfe522c1b23b8dae3542442a8900b14f9bbcc0`
  passed all 42 terminal-allowed checks (38 successes, three governed skips
  and one neutral CodeQL aggregation), including installed-wheel, full-suite
  and exact-head assurance, with zero unresolved review threads before squash
  merge `39de9c6ab2079b55a4666243baff2a5db7f10604`. Delivery subissues
  #780–#782 may close; scientific review, Rust/R/Julia parity, stable promotion,
  release, parent #597 closure and umbrella #318 closure remain separate gates.
  (AC-03–AC-07)

- [x] **F597-R1 / #782 — Review Fixes:** Add conservative Bellman expansion
  preflight, repeated-belief memoization, strict recursive input/result
  validation, semantic identity and martingale checks, and state-informative
  usable-response dual-control diagnostics with adversarial regressions.
  (AC-03–AC-07) `b9e8bd92`

- [x] **F597-R2 / #782 — Review Fixes:** Extend the 50,000-call preflight
  across adaptive full/horizon/myopic/conditional, no-information and fully
  observed regret recursion; add high-cardinality rejection and accepted
  horizon-boundary regressions while preserving memoization. (AC-03–AC-07)
  `eaa2f194`

- [x] **F597-R3 / #782 — Review Fixes:** Bind the public result validator to a
  complete stage-zero fixed-horizon policy tree, successful exact-assurance
  flags and the governed 50,000-call budget; add adversarial early/late tree,
  false-assurance and enlarged-budget regressions. (AC-03–AC-07) `6d696ca4`

- [x] **F597-R4 / #782 — Review Fixes:** Commit the strict source input model
  inside each result and reconstruct the exact bounded evaluation during
  standalone validation, binding the expansion estimate, horizon values,
  policy selections/ties and transition/learning diagnostics to their source.
  (AC-03–AC-07) `08c371af`

- [x] **F597-R5 / #782 — Independent Review:** Independently reproduce the
  normative exact evaluation and reject altered expansion estimates, final
  horizon cells, policy selections/ties, diagnostic flags and committed input
  models. The review found no unresolved Critical, High or Medium finding in
  the bounded repository delivery. (AC-03–AC-07)
  `belief-state-information-fifth-review.md`

- [x] **G9:** Evidence-map every accepted repository-owned capability to its
  merged experimental delivery; no reviewed exclusion or stable claim is
  introduced. (AC-02, AC-06) `786b5229`
- [x] **G10:** Map the explicit Rust/Python/R/Julia/Mojo or binding disposition
  artifact for each family; native parity and installed shared-fixture
  evidence remain later gates. (AC-04) `786b5229`
- [x] **G11:** Map documentation, examples, generated surfaces and capability
  discovery to the experimental maturity state. (AC-05) `786b5229`
- [x] **G12:** Reconcile existing automated implementation reviews and focused,
  repository-harness and hosted delivery receipts; G14 remains the fresh
  exact-head programme check. (AC-03–AC-07) `786b5229`

## Phase 4 — Programme and hosted closeout

- [x] **G13:** Reconcile all 18 live child issue states and merged dispositions
  at source revision `366186b3`; refresh roadmap, todo, registries, existing
  v1.2.0/v1.3.0 MoSCoW and Mermaid mappings, canonical C16/C17/C18 governance,
  release targets and remaining external gates. Record Project-normalization
  eligibility for closed #558 and #724–#728/#731–#735/#738–#742 without
  mutating GitHub or Project 28. (AC-01, AC-05, AC-06, AC-08, AC-09)
  `786b5229`
- [x] **G14:** Final PR #836 exact head `8f1d70cb` completed 42 terminal
  checks (38 successes, three governed skips and one neutral conclusion), with
  zero failed, cancelled or pending conclusions, zero review threads, and
  `CLEAN`/`MERGEABLE` status before squash merge `163825d8`. (AC-07)
- [x] **G15:** Record repository completion at merge `163825d8` in
  `repository-completion-receipt-20260802.md`, separately from scientific
  review, parity, stable promotion, release/publication, registry acceptance
  and issue closure. Parent #318 and open family-parent issues remain open.
  (AC-02, AC-07)

## Phase 5 — Orchestrated scientific review and adjudication

This phase executes the pending scientific gate without reopening G1–G15.
Repository defects discovered by a panel are delivered through separate
issue-backed remediation slices and rebound to a fresh review candidate.

- [x] **SR1 — Freeze inventory and candidate:** Create #841 as the #318
  scientific-review governance child and #842–#850 as its governed wave,
  remediation, evidence and sampling-harm issue hierarchy; freeze the exact
  issue/Project/roadmap/canonical inventory, candidate commit/tree, artifact
  manifest, tools, references and known limitations. First reconcile live
  GitHub–Project–Conductor version, gate and lifecycle fields, including #570
  v1.3.0/C18/M22, or publish a blocking discrepancy register. Reject dirty,
  moving or unreconciled candidates. (M17-R1–M17-R4; AC-01, AC-09, AC-10)
  `a7f6446`
- [x] **SR2 / #842 — Materialize evidence contracts:** Add versioned schemas and
  validators for review packets, artifact manifests, reviewer attestations,
  role reports, findings, disagreements, dispositions, adjudication,
  scientific approval, promotion receipts and candidate-delta invalidation.
  Receipts require independent decision-maker identity/qualification/conflict,
  candidate/tree and packet hash, family/capability scope, conditions, dissent,
  date, expiry and supersession. Promotion approval must not be satisfiable by
  unbound Boolean flags. Bounded delta review requires deterministic
  classification and independent governance/scientific signatures.
  (M17-R1–M17-R3; AC-03, AC-07, AC-10) `41e5255`
- **Migrated:** **SR3 / #843–#845 — Wave A panel (#619/#571/#595):** The orchestrator commissions
  independent estimand/domain, estimator-assurance, cross-language/API and
  governance/publication reviewers, normalizes their reports and publishes a
  synthesis. Resolve or block on COSS commissioning semantics; estimation
  conditioning/model/provenance, nested uncertainty and vector covariance;
  executable fixtures; portable schemas/capabilities; installed artifacts;
  replayable joint COSS selection uncertainty; maturity discovery; CRRA
  stability near risk aversion one; VoC alias and presentation-digest
  integrity; and auditable promotion evidence.
  (M17-R3–M17-R5; AC-02–AC-06, AC-08, AC-10)
  - [x] Materialize and execute the PR #855 stale-branch, normalization-drift and
    exact-head hosted-evidence closeout plan.
- [x] **SR4 / #850 — Sampling-harm scoping:** Materialize
  `sampling_acquisition_harm_voi_20260802` as the C18/M32 planned v1.3.0 Must
  track for the #570 child dependent on #571, with #851–#853 as governed nested
  follow-through. Primary-source, estimand and automated advisory review define
  a fail-closed research scope only. Candidate-bound role-separated agent
  review and an accountable owner decision,
  runtime implementation, ethics/regulatory authorization, promotion, release
  and closure remain pending; this is not an existing #570 kernel and is not
  #595's VoC presentation.
  (M17-R6; AC-02, AC-03, AC-10) `ea6100d2`
- **Migrated:** **SR5 / #846 — Wave B high-risk panel:** Review #570 and #597–#600 with required
  domain specialists, including #570 v1.3.0/C18/M22 synchronization, sequential
  model validity, signed/social harms, sparse-subgroup validity and continuous
  or fitted outcome-conditional estimator assurance. (M17-R3, M17-R4; AC-01–
  AC-06, AC-09, AC-10)
- **Migrated:** **SR6 / #847 — Wave C remaining-family panel:** Review #556–#560, #572, #582
  and #593–#596 in estimand-coherent cohorts. Preserve a separate report and
  verdict for every family; grouping cannot hide a failed required dimension.
  (M17-R3, M17-R4; AC-02–AC-06, AC-10)
- **Migrated:** **SR7 / #848 — Cross-cutting installed and parity review:** Verify packaged
  schemas/capabilities, clean-wheel execution outside the checkout, shared
  Rust/Python fixtures, governed R/Julia unsupported behavior or execution,
  version consistency, central capability discovery and negative stable-API/
  ABI assertions. Implement M17-X1–M17-X7 through nested #318/#571/#619/#595
  issues. (M17-R3–M17-R5, M17-X1–M17-X7; AC-03–AC-05, AC-10)
- **Migrated:** **SR8 — Remediation and independent re-review:** Implement accepted
  findings only in their owning issue-backed slices; freeze new packets, map
  every prior finding to a permitted disposition and commission fresh affected
  reviewers. No author/remediator self-approves. (M17-R1–M17-R3; AC-03,
  AC-07, AC-10)
- **Migrated:** **SR9 / #849 — Orchestrator synthesis and disagreement adjudication:** Publish
  the family verdict matrix, unresolved finding and dissent registers,
  dependency order and evidence-backed recommendation. Distinguish governed
  pending gates, new defects, research questions, human decisions and release
  gates; never promote by majority vote. (M17-R3, M17-R4; AC-02, AC-10)
- **Migrated:** **SR10 — Accountable decision and synchronized closeout:** Obtain the
  repository owner's accountable scientific verdict and separate maintainer
  maturity decision after the role-separated agent panel. For #850, require
  distinct scientific and domain/ethics agent reports and use an agent chair
  for dissent, dispute, or reviewer remediation. Verify receipt digests,
  scope, expiry, supersession,
  complete finding closure, candidate delta and exact head; synchronize
  issues, Project 28, roadmap, todo, canonical projections and metadata. Record
  readback or set Sync State to `Conflict`. Record parity, stable promotion,
  release, publication, registry acceptance and parent/#318 closure
  independently. (M17-R1–M17-R4, M17-R7–M17-R9; AC-01, AC-07, AC-09,
  AC-10)
  - [x] Record the repository owner's Option A decision for C18 M22–M31:
    scientifically acceptable at experimental maturity, no stable promotion,
    stable-core release accepted with experimental-frontier boundaries, and
    external registry/publication acceptance kept separate. — `4ac2f96a`

## Supersession

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/archive/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
