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

1. Declare the sampling action, no-sampling comparator, population and affected
   parties, timing, horizon, information observation, downstream decision
   policy, ordinary research cost and a joint law for acquisition harm.
2. Declare harm types, physical/psychological/privacy/community or other units,
   severity, reversibility, attribution, catastrophic and absorbing outcomes,
   dependence on design/information/state, and missingness or under-reporting.
3. Use one explicitly selected risk treatment: commensurate expected welfare
   loss, constrained expected harm, chance constraint, lower-tail
   CVaR/expected shortfall, lexicographic safety constraint, or a separately
   reviewed criterion. Never convert heterogeneous harms to money or utility
   without a declared valuation and stakeholder scope.
4. Define gross information value and sampling-acquisition harm on the same
   probability space. Additive `EVSI - cost - harm` is valid only when the harm
   value is separable and commensurate; otherwise retain a constrained or
   vector result instead of fabricating a scalar net value.
5. Include an explicit no-sampling design. Under zero acquisition harm and
   identical feasibility, the scalar separable case must reduce to #571's
   ordinary EVSI/ENBS contract.
6. Return feasibility, expected and tail harm, catastrophe probability,
   constraint margins, selected/no-sampling design, signed net information
   value where defined, affected-party diagnostics, uncertainty and
   provenance. Do not imply that a positive economic ENBS overrides an ethical,
   regulatory or safety constraint.
7. Require enumerable opposing examples: high-information catastrophic harm,
   low-information safe sampling, rare absorbing harm, correlated
   information/harm, heterogeneous parties, zero harm, and no-sampling
   optimality. Sensitivity must cover harm probability, severity, valuation,
   risk tolerance, constraint threshold and under-reporting.
8. Keep the capability `unsupported_research_scoping` until primary-source
   review, independent domain/scientific approval, estimator assurance,
   portable contracts, fixtures, bindings and exact-head hosted checks are all
   separately satisfied.

## Acceptance criteria

- **AC-01:** Issue #850, parent #570, dependency #571, umbrella #841, Project
  28, C18/M32, roadmap, todo, track metadata and cross-references agree.
- **AC-02:** The estimand distinguishes sampling harm, ordinary study cost,
  downstream decision harm, risk-sensitive perfect information and VoC.
- **AC-03:** Sampling action, harm law, parties, timing, units, catastrophe,
  risk criterion, constraints and no-sampling comparator are explicit.
- **AC-04:** Additive net value is permitted only under declared separability
  and commensurability; otherwise the result remains constrained or vector.
- **AC-05:** Zero harm reduces to ordinary EVSI/ENBS, and enumerable
  counterexamples prove that positive EVSI need not justify sampling.
- **AC-06:** Primary sources and sensitivity requirements are recorded with
  stable identifiers, scope and limitations.
- **AC-07:** Capability discovery, schemas and docs fail closed and state that
  no sampling-harm runtime currently exists.
- **AC-08:** A named independent human scientific/domain verdict bound to the
  exact candidate is required before any runtime or promotion task may start.
- **AC-09:** Repository validation, hosted checks, stable promotion, release,
  publication and issue closure remain separate gates.

## Non-functional constraints

- Preserve Rust as numerical authority for any future accepted kernel.
- Use finite, versioned, deterministic, content-addressed contracts.
- Preserve complete ties and no-sampling feasibility.
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

## Authoritative inputs

- GitHub issue #850, observed 2026-08-02.
- `conductor/tracks/study_design_efficiency_20260727/` (#571 ENBS/COSS).
- `conductor/tracks/supported_frontier_method_completion_20260723/` (#841
  scientific-review protocol and M17-R6).
- HHS, *Belmont Report*, official OHRP record, accessed 2026-08-02:
  https://www.hhs.gov/ohrp/regulations-and-policy/belmont-report/read-the-belmont-report/index.html
- HHS, *45 CFR 46*, official OHRP regulation index, accessed 2026-08-02:
  https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/index.html
- ICH E6(R3), Principle 7 and risk-proportionate trial processes, Step 4
  presentation dated 2025-01-23:
  https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step%204_Presentation_2025_0123.pdf
- Camilleri et al., *Active Learning with Safety Constraints*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/d6929af3791b2cec21c136b573aa87f2-Abstract-Conference.html
- Bottero et al., *Information-Theoretic Safe Exploration with Gaussian
  Processes*, NeurIPS 2022:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/c628644624c1be9c8cfb1541fa6421fd-Abstract-Conference.html
- Strong et al., *Value of Information for Clinical Trial Design: The
  Importance of Considering All Relevant Comparators*, PharmacoEconomics
  2024, DOI 10.1007/s40273-024-01372-0.

These sources establish distinct economic-value, safety-constraint and
research-protection considerations. They do not by themselves establish one
universal scalar sampling-harm VOI formula.
