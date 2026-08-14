# MoSCoW requirements — planned v1.2.0 and v1.3.0

## Must

- **M17-R1:** The planned v1.2.0 scientific-review programme must use a
  separate orchestrating agent and at least four role-specific subagents for
  estimand/domain, estimator assurance, cross-language/API and
  governance/publication review. Record reviewer identity, qualifications,
  contribution history, conflicts and independence; a remediator cannot
  independently approve the remediated slice.
- **M17-R2:** Freeze every review candidate to an exact commit/tree and
  SHA-256 artifact manifest containing contracts, fixtures, algorithms, tests,
  documentation, claims, toolchain, commands, issue/Project state, canonical
  projections and references. Every change invalidates approval by default;
  bounded metadata-only re-review requires deterministic allowlist
  classification independently signed by the governance reviewer and an
  affected scientific reviewer.
- **M17-R3:** Use the governed acceptance matrix and structured finding,
  disagreement, adjudication and synthesis records. Critical/High findings may
  close only as independently verified fixes or reviewed exclusions, and an
  excluded capability cannot be promoted. Scientific dissent blocks
  acceptance; every Medium requires an explicit disposition and affected-role
  re-review. Approval/adjudication receipts bind an independent qualified
  decision-maker, conflicts, candidate/tree and packet hash, family/capability
  scope, conditions, dissent, date, expiry and supersession. Boolean-only
  promotion approval is insufficient.
- **M17-R4:** Run risk-ordered review waves: #619/#571/#595 first; #570/#597–
  #600 next; remaining C17/C18 families next; then cross-cutting installed
  artifact, fixture, capability, parity, reproducibility and promotion review.
  Keep scientific acceptance, installed parity, stable promotion, hosted
  assurance, release, publication, registry acceptance and issue closure as
  separate gates.
- **M17-R5:** Wave A must resolve or explicitly block on the panel's initial
  High findings: COSS no-study versus conditional-on-commissioning semantics;
  #619 conditioning/model/provenance binding, nested-estimator uncertainty and
  unit-safe vector covariance scalarization; executable EVPPI/EVSI fixtures;
  #571 portable schemas/capabilities; installed-wheel execution and resources;
  replayable COSS selection uncertainty; #595 numerically stable CRRA behavior
  near risk aversion one and presentation-bound VoC provenance; unified
  maturity/capability discovery; and auditable promotion receipts.
- **M17-R6:** Sampling-process harm must not be inferred from #570 risk-
  sensitive policy value or #595 expected-utility/VoC. Create a separately
  scoped estimand and primary-source review before making a risk-adjusted
  sampling-harm claim; until then record the capability as unsupported.
- **M17-R7:** Human-confirmation preparation must distinguish Git OIDs from
  SHA-256 content digests, recompute canonical hashes, verify frozen-tree
  bytes, reconcile complete findings and enforce contribution-aware separation
  among orchestrator, author/remediator, reviewer, chair and approver. Pending
  templates cannot masquerade as normative evidence.
- **M17-R8:** The orchestrator must synthesize the panel's findings, dissent,
  options, contingencies, rationale and recommendation without deciding.
  Candidate-bound human receipts must use an auditable authenticated channel,
  record conditions, expiry and supersession, and fail closed after material
  change. #850 requires distinct scientific and domain/ethics humans, with an
  independent chair only when dissent, dispute or remediation requires one.
- **M17-R9:** Enforce complete Critical/High and Medium dispositions,
  capability-bound reviewed exclusions, narrow administrative-only delta
  review and ordered GitHub/Project/Conductor transitions. Partial
  synchronization sets Sync State to `Conflict` and prohibits closure.
- **M17-X1:** Add portable request/result schemas and a capability artifact for
  #571; downgrade unsupported `fixture-backed` classification until present.
- **M17-X2:** Execute #619, #571 and #595 Python callables, packaged discovery
  and failure cases from clean installed wheels outside the source checkout.
- **M17-X3:** Add portable executable #619 EVPPI and EVSI fixtures consumed
  independently by Rust and Python.
- **M17-X4:** Package versioned schemas, capability records and fixture
  metadata in wheels behind an installed resource API.
- **M17-X5:** Create one canonical capability registry for #619, #571 and #595
  covering maturity, contract versions, surfaces and all governed language
  dispositions.
- **M17-X6:** Enforce exact contract, fixture, runtime envelope and capability
  version consistency, including correction of #571's `1.0`/`1.0.0` drift.
- **M17-X7:** Require stable API/ABI negative tests and an evidence-bearing
  promotion validator so no experimental capability is promoted by an unbound
  flag, source-checkout test or missing registry entry.

- **M22-U1:** #570 planned v1.3.0/C18 risk-sensitive constrained information
  value must compare matched current and post-information feasible policy
  problems under one declared expected-value, expected-utility, lower-tail
  CVaR/expected-shortfall or minimax-regret functional.
- **M22-U2:** Fix objective direction and units, state probabilities,
  deterministic or chance-constrained budget, capacity, eligibility, fairness,
  regulation, carbon, liquidity and service-level semantics, plus information-
  cost placement across both policy problems.
- **M22-U3:** Return complete ties, infeasibility, current/informed policies,
  gross/net value, switches, risk/constraint diagnostics and constraint-removal
  evidence without presenting discrete removal effects as local shadow prices.
  Normalize the owning issue, Project gate/version, roadmap and canonical C18
  projection before scientific review entry.

- **M25-U1:** #593 planned v1.3.0 must declare current, specific, perfect and
  post-sample implementation as conditional distributions over realised
  actions, with uncertain states, intended policies, sample likelihoods,
  chronology, value units, population, discounted time factor and costs.
- **M25-U2:** Return the auditable current/perfect-information by
  current/perfect-implementation matrix; specific-implementation and
  sample-information cells when declared; EVPIM, EVSIM, realizable EVPI, EVP,
  IA-EVSI, signed net components, complete policy ties, switches, interaction
  and exact decomposition residuals. Do not assume implementation and
  information are independent.
- **M25-U3:** Treat EVEIm and EVSEIm as review-candidate presentation labels,
  reconcile rather than replace the existing implementation-loss helper, and
  require exact fixtures, zero-uptake/dependence/pathology tests, estimator
  assurance, deterministic serialization and explicit Python/Rust/R/Julia/Mojo
  dispositions before promotion.

- **M21-U1:** #560 planned v1.3.0 MCDA information value must declare named
  alternatives; criteria with raw units, directions and fixed ex-ante value
  functions/normalization anchors; nonnegative normalized preference weights;
  a finite joint uncertainty law that preserves outcome/weight correlation;
  the aggregation direction and aggregate-value unit; and provenance.
- **M21-U2:** The v1 estimand must resolve only explicitly declared
  criterion-performance, preference or joint latent variables while integrating
  residual uncertainty. Return baseline expected aggregate scores/ranking/choice,
  complete ties, conditional states/choices, action-specific gross and signed
  net information value, criterion/preference/joint decompositions, interaction
  and no-double-counting diagnostics, scenario regret and rank acceptability.
- **M21-U3:** Define expected and statewise Pareto diagnostics against raw
  direction-adjusted criterion performance rather than cost-effectiveness
  dominance. Use one shared additive-value kernel; freeze normalization before
  information; preserve correlations; reject non-finite or incoherent scales,
  negative/unnormalized weights, missing uncertainty and unidentified actions.
- **M21-U4:** Keep v1 distinct from qualitative priorities, DSA/PSA, scalar
  net-benefit VOI, Value of Preference and arbitrary weighted scoring. AHP
  elicitation/pairwise consistency, outranking, veto/threshold and other
  non-compensatory methods, per-state/post-information renormalization and
  imperfect-sample EVSI are explicitly unsupported pending separate contracts.
  Require exact fixtures, pathology/property tests, installed execution,
  language dispositions and independent scientific review before promotion.

- **M20-U1:** #558 planned v1.3.0 qualitative VOI must use a versioned portable assessment
  contract that identifies the decision, information questions, uncertainty or
  evidence gaps, potential decision impact, feasibility, timeliness,
  equity/ethics, proposed information action, cost or burden, confidence,
  rationale, sources, missingness and accountable human reviewers.
- **M20-U2:** Return deterministic ordinal priority and recommendation classes,
  complete tie groups, unresolved conflicts, dissent and an explicit
  complete/incomplete/unverified state without converting ordinal judgements
  into probabilities, utilities, currency, weighted scores or quantitative
  VOI. Sorting, tie and missing-data policies must be declared and auditable.
- **M20-U3:** Preserve immutable assessment and audit-event identifiers,
  versioned history, source and transformation provenance, redaction markers,
  actor roles and timestamps. AI-assisted contributions must record provider,
  model/version, prompt or input reference, verification state and human
  override; AI output cannot satisfy human approval.
- **M20-U4:** Keep the workflow distinct from MCDA, expert elicitation, Delphi
  consensus, evidence grading, risk-of-bias assessment and numerical VOI.
  Require deterministic serialization/rendering, conflict and adversarial
  fixtures, accessibility/usability review, explicit language dispositions and
  scientific/practitioner review before promotion.

- **M19-U1:** #557 Value of Distribution-Family Information must identify the
  uncertain model-family index, its evidence-conditioned probabilities, the
  common alternatives, decision direction, comparable value unit and the
  within-family conditional expected value for every family/alternative pair.
- **M19-U2:** The v1 estimand must perfectly resolve only the discrete
  model-family index while leaving within-family parameter and outcome
  uncertainty integrated out. Return current and family-resolved values,
  complete tie sets, probability-weighted contributions, gross VDI,
  information cost and signed net VDI without silent renormalization or
  clipping.
- **M19-U3:** Record the exact conditioning order, family definitions,
  probability and value provenance, comparability diagnostics, estimator
  status and language dispositions. Treat the calculation as a discrete-index
  EVPPI presentation, keep it distinct from full structural EVPI and
  model-discrimination EVSI, and require scientific review before promotion.

- **M18-U1:** #556 deterministic sensitivity analysis must declare a fixed
  baseline, parameter/scenario coordinates, compared alternatives, outcome
  direction and units before evaluating one-way, two-way or scenario surfaces.
- **M18-U2:** Return every evaluated point, baseline and optimal alternatives,
  incremental outcomes, deterministic range/ranking metrics, complete tie sets
  under declared absolute/relative tolerances, and every observed tie or
  bracketing switch interval with exact-versus-bracket status. Tornado ranking
  must name its grid-extrema or endpoint metric; interpolation must be opt-in,
  estimated and assumption-labelled rather than fabricated.
- **M18-U3:** Keep DSA distinct from PSA, EVPPI and global sensitivity; reject
  non-finite/missing baselines, duplicate or unknown coordinates, malformed
  callback results and unsupported extrapolation. Two-way inputs with correlated
  coordinates must declare a feasible mask or path rather than implying that an
  infeasible Cartesian surface is covariance-aware.

- **M16-U1:** Delegate #595 delivery to
  `risk_adjusted_information_pricing_20260731`, which represents named utility,
  wealth/reference state, risk attitude, units, information/cost location,
  policies, scope and deterministic provenance.
- **M16-U2:** Require EUI, CEI, BPI, SPI, anchored PPI, signed values, policy
  switches, root diagnostics and explicit comparability conditions.
- **M16-U3:** Treat VoC as a presentation of the same clairvoyant-policy result
  and permit monetary EVPI reduction only under verified positive-affine
  utility.
- **M17-U1:** Keep canonical C16, issues/subissues and Project 28 synchronized
  through bounded managed projections.

## Should

- **M31-S1:** #600 planned v1.3.0 outcome-conditional sample-information value
  must declare a finite state prior, finite measurement-outcome likelihood,
  named actions, utility/maximization or loss/minimization orientation, common
  value unit, population, horizon, discount basis, baseline reference policy,
  nonnegative low-value thresholds, information-cost placement and prospective
  or retrospective scope. The reference policy must attain the exact baseline
  extremum. The declared complete-tie tolerance is only for tie-set and
  presentation diagnostics; it never admits a non-extremal reference policy.
- **M31-S2:** For each outcome `x`, return its predictive probability, posterior
  state law, action values, complete optimal ties, direction-aware
  `delta-EV_x`, nonnegative `VSI_x`, signed net `VSI_x`, and policy/tie
  diagnostics. Return `EVSI = E_x[VSI_x] = E_x[delta-EV_x]` only as an
  expectation-linear tower identity. Do not infer equality of variances,
  standard deviations, quantiles, tails or outcome-wise estimands.
- **M31-S3:** Implement Equation 10 as the predictive-probability-weighted
  population standard deviation
  `sigma-VSI = sqrt(sum_x p(x) (VSI_x - EVSI)^2)` with `ddof = 0`; never copy
  the unweighted MATLAB/Table 3 calculation. Return monotone
  `rVSI_delta = P(VSI_x <= delta)`, weighted quantiles/tails, and distinguish
  `rVSI_0` from reference-action exclusion, mandatory-policy-switch and
  complete-tie-set-change mass when baseline or posterior policies are tied.
- **M31-S4:** Require exact finite enumeration, deterministic serialization, a
  portable input commitment, independent result reconstruction, probability
  and Bayes calibration residuals, negative `delta-EV_x` and nonnegative
  `VSI_x` evidence, threshold monotonicity, and explicit
  Python/Rust/R/Julia/Mojo dispositions. Continuous outcomes, dynamic/adaptive
  sampling, fitted estimators, risk in underlying system outcomes, scientific
  validity, stable promotion, parity, release and parent closure remain
  separate gates.

- **M30-S1:** #599 planned v1.3.0 must declare a prespecified subgroup
  partition, covariates, population weights, common value unit/direction,
  horizon and discount basis, subgroup policy eligibility, effect-state law,
  selection/multiplicity policy, fairness/privacy constraints and provenance.
- **M30-S2:** Return current-information population-common and subgroup-policy
  values (`C0`, `Cf`), their perfect-information counterparts (`P0`, `Pf`),
  direction-aware static and dynamic value, population-common and subgroup-
  policy EVPI, subgroup policies, complete ties, switches and the exact
  identity `dynamic - static = EVPIf - EVPI0`.
- **M30-S3:** A declared finite sampling model may return `S0`, `Sf`,
  population-common EVSI, subgroup-policy EVSI, sample-informed segmentation
  value, study cost and signed net diagnostics with an analogous identity.
  Keep these distinct from dynamic perfect-information value, subgroup effect
  estimation and estimator uncertainty.
- **M30-S4:** Require enumerable opposing/zero cases, population-weighting and
  direction invariants, strict schemas, deterministic serialization, exact
  estimator assurance and Python/Rust/R/Julia/Mojo dispositions. Selection
  bias, sparse-subgroup validity, causal identification, fairness/privacy
  approval, stable promotion and release remain separate gates.

- **M29-S1:** #598 planned v1.3.0 signed/social information value must declare a
  complete finite joint-world law, named agents and roles, signal topology,
  eligible and actual recipients, bounded nonanticipative policy catalogs,
  comparator-linked sharing designs, value units and provenance.
- **M29-S2:** Return selected policies, complete ties, pre-transfer, transfer,
  cost and post-transfer ledgers, signed agent/role/social comparator values,
  selective-sharing comparisons, harm, avoidance, switches, winners/losers,
  externalities and rights/consent/purpose receipts without clipping.
- **M29-S3:** Require an explicit cardinal-comparability declaration and named
  welfare aggregator. Apply Blackwell nonnegativity only to a verified aligned
  centralized refinement with unchanged preferences and constraints and an
  embedded comparator catalog; otherwise return explicit inapplicability
  reasons.
- **M29-S4:** Limit v1 execution to centralized, fixed, declared-response and
  receipt-verified finite-equilibrium catalogs. Require deterministic exact
  fixtures, adversarial pathology tests, explicit Python/Rust/R/Julia/Mojo
  dispositions, independent scientific review and hosted exact-head evidence;
  keep persuasion, mechanism design, rational inattention and general game
  solving adjacent.

- **M27-S1:** #596 planned v1.3.0 event-localized information value must
  declare a finite probability law, actions and value unit, a nontrivial event
  or threshold and complement, chronology, information cost, binary-channel
  accuracy grid, coordinate names/units, base coordinate and a declared
  true-max baseline-optimal reference action. The v1 objective is
  higher-is-better `maximize`; tie tolerance is in `[0, 1e-6]` and integral
  tolerance is in `(0, 1e-6]`.
- **M27-S2:** Use the canonical policy-relative EUI density
  `i(x) = f(x) [max_a g_a(x) - g_a*(x)]`. Return every probability-mass atom,
  conditional action value and complete tie, the nonnegative density integral,
  modes and directions from the declared base. The optional centered
  `j(x) = f(x) [max_a g_a(x) - V0]` is an explicitly signed diagnostic; its
  integral identity does not make each atom nonnegative.
- **M27-S3:** Return exact event/complement probabilities, conditional
  decisions and values, gross/net perfect-event VOI and an imperfect symmetric
  binary-channel accuracy curve. Verify event/complement partitioning,
  accuracy `p`/`1-p` symmetry, the uninformative `0.5` boundary, deterministic
  serialization and raw-atom density-integral tolerances. A requested grid
  without a complementary pair reports `null`, never a false zero residual.
  Bind the portable result to auditable state/coordinate partition evidence;
  re-evaluate the event definition and reconstruct every baseline, channel and
  density action marginal, rejecting mismatched references or ungrouped equal
  coordinates.
- **M27-S4:** Keep finite event/density EUI distinct from ordinary threshold
  plotting, DSA, parameter EVPPI, forecast accuracy and tail-risk measures.
  Monetary BPI remains delegated to #595. Python may be experimental; Rust, R
  and Julia remain unsupported and Mojo external until separate evidence.

- **M23-S1:** #572 planned v1.3.0 forecast-signal information value must
  consume a declared forecast artifact with outcome prior, signal likelihood,
  reported conditional probabilities, objective/payoff units, feasible
  actions and constraints, acquisition cost, horizon, freshness, latency and
  lead time. It must not train or tune forecasting models.
- **M23-S2:** Return baseline and signal-conditional policies, complete ties,
  timely-oracle and signed deployed values, calibration loss, cost and signed
  net value, nonnegative maximum price, regret avoided and per-signal value
  contributions. Late or stale information has zero operational value while
  retaining its counterfactual timely diagnostic.
- **M23-S3:** Report calibration, Brier and signal-probability coverage
  diagnostics without relabelling accuracy as value. Require analytical
  newsvendor evidence and no-skill, perfect, miscalibrated, late, stale,
  permutation and pathology limits; exact-enumeration assurance; explicit
  Python/Rust/R/Julia/Mojo dispositions; independent scientific review and
  hosted exact-head evidence before promotion.

- Provide additive MCDA analytical enumeration, scale/alternative/state
  permutation invariants, correlation and complete-tie cases, normalization and
  weight pathologies, accessible information-value/rank/regret plots and an
  independently reviewed worked example.
- Provide worked, disagreement, incomplete, redacted and adversarial
  qualitative assessments; schema/property/audit-chain tests; accessible text
  rendering; and reviewer-oriented explanations of every recommendation.
- Provide normalized tabular input, accessible tornado plotting, explicit
  estimated interpolation labels and independent analytical/brute-force tests.
- Provide analytical enumeration, permutation/splitting/scaling invariants,
  loss/minimization equivalence, bound checks against matched structural EVPI
  and explicit Monte Carlo follow-on assurance requirements.
- Provide nonlinear-utility counterexamples, probability-price anchors,
  root-finding diagnostics and explicit polyglot dispositions.

## Could

- Add further non-cardinal presentation views over the same portable
  qualitative assessment contract.
- Add further reviewed presentation labels without new kernels.

## Won't

- Infer weights from qualitative classes, re-normalize after observing a state,
  label AHP/outranking/veto outputs as supported, or describe predictive/ranking
  accuracy as MCDA information value.
- Infer cardinal distances between ordinal classes, aggregate them as numeric
  weights, silently resolve dissent, treat AI output as verified or describe
  the qualitative workflow as an EVPI/EVPPI/EVSI estimate.
- Treat deterministic ranges as probability distributions, uncertainty
  attribution or information value.
- Create a duplicate VoC method or overwrite human-authored issue content.
- Reuse the structural-EVPI kernel when it would also resolve within-family
  uncertainty, infer model probabilities, or claim that VDI terminology is
  scientifically standardized.
- Clip negative private or social information value, infer cardinal welfare
  comparability, omit transfer/cost ledgers or apply Blackwell nonnegativity
  outside a verified aligned centralized refinement.
- Treat the bounded fixed/declared-response/verified-equilibrium catalogs for
  C18/M29 as Bayesian persuasion, mechanism design, rational inattention or a
  general game solver.
