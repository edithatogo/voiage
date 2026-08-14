# Belief-state information third-review remediation

## Scope and independence

A third fresh reviewer inspected issue #597 / C18-M28 at signed head
`3d94f97c8546266562a07239a34b908e6543dda1`. The reviewer did not author the
original implementation or either earlier remediation. Review covered the
finite belief-MDP chronology, Bellman comparators, input and result contracts,
schemas, fixtures, tests, documentation, governance records, and the
50,000-call preflight across adaptive, no-information and fully observed
evaluators.

This is an engineering implementation review. It is not independent scientific
approval, polyglot parity, stable-promotion evidence, release evidence, or
programme closure.

## Finding and remediation

**Medium — the result validator did not bind all advertised exact-assurance
claims.** A structurally valid result could start its policy tree after stage
zero, terminate a branch before the declared fixed horizon, continue beyond the
horizon, report failed posterior-martingale, null-sensor, no-information or
complete-tie assurance, or enlarge the governed exact-enumeration budget. This
contradicted the claimed complete fixed-horizon policy tree and fail-closed
exact-assurance result contract.

The reviewer reported the finding before editing, then became the remediator.
Signed commit `6d696ca4` now requires stage-zero roots, consecutive continuation
through every positive-probability branch, termination exactly at the fixed
horizon, successful exact-assurance flags, and the governed 50,000-call public
budget. Adversarial result mutations cover every repaired boundary. Because the
reviewer authored this correction, a fourth fresh reviewer must approve the
resulting head.

## Independent recomputation and runtime-bound audit

- The normative prior, transition, likelihood, reward and sensor-cost inputs
  recompute to closed-loop gross value 7, closed-loop net value 6.5, matched
  no-information value 0, myopic value 0, fully observed value 20 and gross
  partial-observability regret 13.
- Instrumented recursive evaluator wrappers counted 52 adaptive, 12
  no-information and 18 fully observed invocations for the normative fixture.
  The preflight estimate was exactly 82, so it neither omitted a recursive
  evaluator nor relied on a memoization hit for safety.
- The 20-state, one-action, one-observation, four-stage adversary rejects before
  recursion at estimate 168,453. The accepted one-action, null-sensor,
  two-state horizon-twelve boundary reports 24,649, below the 50,000 limit.
- The preflight counts the full adaptive solve, each horizon-prefix solve,
  myopic solve, conditional-sensing continuations, full and horizon-prefix
  no-information solves, and every latent-state branch of the fully observed
  regret comparator. It deliberately ignores adaptive cache hits, so the bound
  remains conservative for numerically distinct belief representations.

## Local assurance

- 114 focused tests passed with 100% statements and branches for
  `voiage.methods.belief_state_information` (566 statements, 264 branches).
- 156 combined belief-state, package-export, CLI and supported-frontier tests
  passed.
- Ruff check and format verification passed. BasedPyright reported zero errors;
  five existing test-only private-usage warnings remain non-blocking.
- Frontier-contract, GitHub cross-reference and full Conductor validation are
  required again after the plan and evidence records are committed.

The Python 3.14 coverage-loader path initially reproduced the known NumPy
`cannot load module more than once per process` error. Pre-importing NumPy
before invoking pytest avoided the loader defect; the same source and tests
then produced the complete coverage result above.

## Remaining boundary

A fourth fresh independent implementation review, hosted exact-head checks and
merge remain pending. Independent scientific review, Rust/R/Julia parity,
stable promotion, release, parent #597 closure and umbrella #318 closure also
remain pending.
