# Sampling-Acquisition-Harm Value of Information

## Overview

Issue [#850](https://github.com/edithatogo/voiage/issues/850) scopes a distinct
v1.3.0 research family for information-acquisition actions that can themselves
cause stochastic harm. It is owned by risk-sensitive issue #570, depends on
#571's study-design/ENBS contract, and remains under scientific-review umbrella
#841 and frontier programme #318.

The family is not an existing #570 runtime, not a #595 VoC presentation, and
not ordinary trial cost. This track first freezes the estimand, safety and
ethical boundary. Runtime implementation is prohibited until candidate-bound
independent scientific/domain review and a named human verdict approve a
narrow contract.

## Requirements

1. Declare the sampling action `d`, explicit no-sampling comparator `d0`,
   candidate set, population and affected parties, time filtration, decision-
   time observable history, horizon, downstream policy, and incremental
   ordinary cost and acquisition harm relative to `d0`.
2. Declare harm types, physical/psychological/privacy/community or other units,
   severity, reversibility, attribution, catastrophic and absorbing outcomes,
   dependence on design/information/state, and missingness or under-reporting.
3. Use one explicitly selected risk treatment: commensurate expected welfare
   loss, constrained expected harm, chance constraint, upper-tail CVaR or
   expected shortfall for positive loss, lower-tail CVaR or expected shortfall
   for signed welfare/value, lexicographic safety constraint, or a separately
   reviewed criterion. Declare sign, confidence level, quantile convention and
   atom handling. Never convert heterogeneous harms to money or utility without
   a declared valuation, perspective, cardinal scale, numeraire, horizon,
   discount convention, source date and affected-party scope.
4. Define gross information value and sampling-acquisition harm on the same
   design-indexed causal probability space. The downstream policy may use only
   its declared observable history. Additive incremental `EVSI - cost - harm`
   is valid only when acquisition does not alter the state/action set, valued
   harm is policy-independent, and value, cost and harm are separable and
   commensurate and a mutually exclusive outcome-component ledger prevents
   double counting; otherwise use total joint welfare once, or a constrained
   or vector result.
5. Include an explicit no-sampling design. Under zero acquisition harm and
   identical feasibility, the scalar separable case must reduce to #571's
   ordinary EVSI/ENBS contract.
6. Return mathematical feasibility as `feasible`, `infeasible` or
   `indeterminate`; expected and tail harm; catastrophe probability;
   constraint margins; all nondominated designs and complete ties; a selected
   design only under a declared ordering; signed net information value where
   defined; party/subgroup diagnostics; uncertainty; and provenance. Keep
   mathematical feasibility separate from accountable ethics/regulatory scope
   authorization.
7. Require enumerable opposing examples: high-information catastrophic harm,
   low-information safe sampling, rare absorbing harm, correlated
   information/harm, heterogeneous parties, zero harm, and no-sampling
   optimality. Sensitivity must cover harm probability, severity, valuation,
   risk tolerance, constraint threshold and under-reporting. Model latent harm,
   reporting, dropout and validation data explicitly; return `not_identified`
   or bounds where the observed law cannot identify the harm law.
8. Keep the capability `unsupported_research_scoping` until primary-source
   review, independent domain/scientific approval, estimator assurance,
   portable contracts, fixtures, bindings and exact-head hosted checks are all
   separately satisfied.

## Acceptance criteria

- **AC-01:** Issue #850, native subissues #851–#853 and governed descendants
  including #867, #870, #873 and #876, parent #570, dependency #571, umbrella #841,
  Project 28, C18/M32, roadmap, todo, track metadata and cross-references agree.
- **AC-02:** The estimand distinguishes sampling harm, ordinary study cost,
  downstream decision harm, risk-sensitive perfect information and VoC.
- **AC-03:** Sampling action and `d0`, design-indexed potential outcomes,
  observable filtration, harm law, parties, timing, units, catastrophe, risk
  criterion, constraints and authorization boundary are explicit.
- **AC-04:** Additive net value is permitted only under declared separability
  and commensurability with a mutually exclusive component ledger; otherwise
  the result uses total joint welfare once or remains constrained/vector.
- **AC-05:** Zero harm reduces to ordinary EVSI/ENBS, and enumerable
  counterexamples prove that positive EVSI need not justify sampling.
- **AC-06:** Primary sources and sensitivity requirements are recorded with
  stable identifiers, scope and limitations.
- **AC-07:** Capability discovery, schemas and docs fail closed and state that
  no sampling-harm runtime currently exists.
- **AC-08:** A named independent human scientific/domain verdict bound to the
  exact candidate is required before any runtime or promotion task may start.
  Independent role reports and a separate orchestrator must supply findings,
  options, contingencies, rationale and recommendation, but cannot satisfy the
  human verdict. A distinct domain/ethics human is additionally required for
  this high-risk family; an independent chair adjudicates disputed findings,
  scientific dissent or reviewer remediation.
- **AC-09:** Repository validation, hosted checks, stable promotion, release,
  publication and issue closure remain separate gates.
- **AC-10:** Human-confirmation evidence cryptographically binds canonical
  packet and artifact bytes, Git OIDs, complete finding history, reviewer
  eligibility, signed receipts, expiry/supersession and fail-closed state
  transitions; stale, moving or partially synchronized evidence is rejected.
- **AC-11:** Current automated source observations remain distinct from the
  immutable H8-C manifest, retain no source bytes, grant no rights or
  applicability, and partition all nineteen pending findings without treating
  repository implementation as an independent disposition.
- **AC-12:** Human commissioning preparation exposes an accountable
  candidate-context decision, binds every source/finding/role prerequisite and
  the existing signed-output schemas, prohibits credentials and unnecessary
  personal data in the repository, and cannot report readiness or authority
  before the bound preconditions are independently satisfied.
- **AC-13:** Preserve the historical unset commissioning preflight and record
  an authenticated candidate-context decision as a separate receipt. Advance
  only `candidate_selected`, remove only the candidate-decision blocker, and
  route every independent source, reviewer, packet and finding prerequisite to
  a separate blocked gate.

## Non-functional constraints

- Preserve Rust as numerical authority for any future accepted kernel.
- Use finite, versioned, deterministic, content-addressed contracts.
- Preserve complete ties, nondominated sets and no-sampling evaluation; never
  assume that no sampling is ethically feasible without the same accountable
  assessment applied to other alternatives.
- Never silently clip signed value or repair safety-constraint violations.
- Treat consent, regulatory and ethics review as external accountable gates,
  not numeric outputs that software can approve.

## External gates

- Candidate-bound independent scientific, domain and ethics review.
- Named human verdict and maintainer maturity decision.
- Any required IRB/research-ethics, regulator, custodian or participant
  authorization for a real study.
- Polyglot parity, exact-head hosted assurance, stable promotion, release,
  publication, registry acceptance and issue closure.

## Out of scope

- Relabelling #570 perfect-information risk adjustment as acquisition harm.
- Relabelling #595 expected utility or VoC as sampling-harm value.
- Treating ordinary monetary research cost as a harm distribution.
- Automatically monetizing death, injury, privacy loss, community harm or
  incomparable stakeholder utilities.
- Claiming that a positive expected net value makes an unethical or infeasible
  study permissible.

The executable H8 plan, decision rights, options, contingencies and transition
model are defined in `human-confirmation-gates.md`.

## Authoritative inputs

- GitHub issue #850, observed 2026-08-02.
- `conductor/tracks/study_design_efficiency_20260727/` (#571 ENBS/COSS).
- `conductor/tracks/supported_frontier_method_completion_20260723/` (#841
  scientific-review protocol and M17-R6).
- HHS, *Belmont Report*, official OHRP record, accessed 2026-08-02:
  https://www.hhs.gov/ohrp/regulations-and-policy/belmont-report/read-the-belmont-report/index.html
- HHS, *45 CFR 46*, official OHRP regulation index, accessed 2026-08-02:
  https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/index.html
- ICH E6(R3), *Good Clinical Practice*, Step 4 final guideline dated
  2025-01-06, especially Principles 1, 2, 3, 6 and 7:
  https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
- Camilleri et al., *Active Learning with Safety Constraints*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/d6929af3791b2cec21c136b573aa87f2-Abstract-Conference.html
- Bottero et al., *Information-Theoretic Safe Exploration with Gaussian
  Processes*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/c628644624c1be9c8cfb1541fa6421fd-Abstract-Conference.html
- Heath, Anna; Baio, Gianluca; Manolopoulou, Ioanna; and Welton, Nicky J.,
  *Value of Information for Clinical Trial Design: The Importance of
  Considering All Relevant Comparators*, PharmacoEconomics 42 (2024), DOI
  10.1007/s40273-024-01372-0.

The versioned retrieval and applicability record is
`primary-source-manifest-20260802.json`. Human-subject and domain requirements
must select jurisdiction- and domain-applicable authorities; these sources do
not establish a universal cross-domain ethics rule.

These sources establish distinct economic-value, safety-constraint and
research-protection considerations. They do not by themselves establish one
universal scalar sampling-harm VOI formula.
