# Independent implementation review

Reviewer: Codex independent review agent, separate from the implementation
agent.

Review date: 2026-08-01. Reviewed maturity: experimental exact finite Python
execution planned for v1.3.0 under canonical C18/M26. This is a repository
implementation review, not independent scientific approval, stable promotion,
cross-language parity, release evidence or issue-closure authorization.

## Scope and numerical result

The review covered the exact finite evaluator, semantic and result validators,
input/result schemas, two- and three-stage fixtures, pathology fixtures,
API/CLI discovery, documentation, Conductor governance, global frontier
registration and maturity boundaries.

Independent table enumeration reproduced the normative decompositions. For
the minimization fixture, EEV, RP and WS are respectively 9, 8 and 5, so
`VSS = EVIU = EEV - RP = 1` and `EVPI = RP - WS = 3`. For the maximization
fixture, EEV and RP are 8.5 and WS is 10, so
`VSS = EVIU = RP - EEV = 0` and `EVPI = WS - RP = 1.5`.

EVIU is correctly presented as v1 VSS because its comparator is the declared
point-estimate EV solution. The implementation does not model information
acquisition or sampling. DVSS/VMS, approximate or external solvers and risk
criteria beyond expected value remain explicit deferrals.

## Findings and remediation

No unresolved Critical, High or Medium implementation finding remains.

- **High — fixed:** recourse-stage histories were checked as independent
  partitions but were not required to form a filtration. Later partitions now
  refine the preceding partition, declared information is cumulative, and a
  crossing-history pathology fails closed.
- **High — fixed:** tolerance-equivalent ties could replace the mathematical
  optimum with a lexical near-optimum while assurance still reported zero
  optimality gap. Exact minima/maxima now determine selected values and
  decomposition; declared-tolerance ties remain presentation diagnostics.
- **High — fixed:** Python non-finite numbers can satisfy a generic JSON Schema
  numeric type. Result validation now recursively rejects NaN and infinity.
- **Medium — fixed:** the implementation initially omitted the global stable-
  promotion checklist entry. The experimental family is now registered with
  scientific review, hosted assurance and Rust/R/Julia parity still pending.
- **Medium — fixed:** the full repository suite exposed normalized metadata,
  v1 programme active-track and downstream changelog evidence-hash drift. The
  governed records and hash pin were synchronized, and the three exact
  regression nodes pass.

## Validation evidence

- 30 feature tests passed; changed-module coverage is 100 percent for 200
  statements and 82 branches across the method and contract modules.
- The focused feature, CLI and public-export suite passed; the broader
  frontier/Conductor governance suite passed 54 tests after remediation.
- Ruff check/format passed. BasedPyright reported zero errors and zero
  warnings for the method, contract, exporter and feature tests.
- Frontier-contract validation passed, including
  `value_of_uncertainty_modelling`.
- The full Conductor validator passed across 148 tracks with zero errors and
  zero warnings. Manifest and evidence SHA-256 pins, JSON parsing and
  `git diff --check` passed.
- A full repository pytest run reached completion. Its only three failures
  were the governance/evidence drifts listed above; after remediation, all
  three exact failing nodes and the 76-test focused/governance rerun passed.

## Boundaries and verdict

The reviewed evaluator is exact only for a declared complete finite policy
space and finite scenario law. Precomputed policy outcomes and claimed
candidate-space completeness remain governed inputs, not proof of an external
solver or model. Infeasible induced recourse yields null EEV/VSS/EVIU, while a
recourse problem with no all-state feasible policy fails closed.

Python is the only executable implementation. Rust, R and Julia remain not
implemented and Mojo remains external. Hosted exact-head and installed-wheel
checks, independent scientific review, stable-promotion approval, release and
#594/#318 closure remain separate gates. Subject to those gates, no open
implementation finding prevents PR review of the experimental Python surface.
