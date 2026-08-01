# MoSCoW requirements — planned v1.2.0 and v1.3.0

## Must

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
