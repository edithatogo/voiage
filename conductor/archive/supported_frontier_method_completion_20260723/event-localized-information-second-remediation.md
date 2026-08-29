# Event-localized information second independent-review remediation

## Boundary

This bounded remediation responds to the fresh independent review of C18/M27
issue #596 at signed head `bf6470a3adc4fc1c1b591e02599f73d1030e5e8f`.
It does not self-approve the result, establish hosted exact-head or installed
wheel evidence, promote maturity, claim language parity, close #596, or close
the umbrella programme. A third fresh reviewer remains required.

## Findings resolved

1. **High — channel-law reconstruction:** Result validation now derives both
   signal probabilities and every action-specific posterior numerator and
   conditional value from the declared accuracy, event probability and
   event/complement conditional action values. A changed accuracy cannot retain
   stale channel summaries or VOI.
2. **High — marginal action identities:** Every baseline action value must
   reconcile independently through the event/complement partition and through
   the grouped coordinate atoms. Nonoptimal action values can no longer be
   changed while leaving only optimal-value summaries intact.
3. **Medium — density binding and grouping:** The density reference action must
   equal the baseline reference action. Coordinates are unique after grouping,
   and their set must exactly cover the coordinates in partition evidence.
4. **Medium — auditable event definition:** Each result now includes the state
   identifier, coordinate and membership flag for every partition member.
   Threshold operators and state-set membership are re-evaluated from this
   evidence, including dimension, coverage and uniqueness checks.
5. **Medium — material-scale assurance:** Numerical identity checks use the
   declared absolute tolerance only; a large-magnitude result cannot acquire an
   unintended magnitude-scaled relative tolerance.

## Local evidence and deviation

- Shared Python 3.14 focused runtime, mutation, schema, plotting, export,
  package and programme suite: 104 passed.
- Independent deterministic oracle: 500 randomized exact finite problems
  matched coordinate VOI, event VOI, `p=0`, `p=0.5`, `p=1` and decimal
  `0.07`/`0.93` symmetry calculations.
- Ruff check/format and BasedPyright: zero findings using cached shared tools.
- Frontier-contract and GitHub cross-reference validators: passed.
- Full bundled Conductor validation: 147 tracks, zero errors and zero warnings.
- `git diff --check`: passed.
- Local branch coverage remains unavailable: the single bounded isolated
  coverage attempt failed during `tests/conftest.py` NumPy import with
  `cannot load module more than once per process`; a no-coverage isolated retry
  stalled during scikit-learn import and was terminated after 60 seconds. No
  further isolated environment was launched. Hosted exact-head changed coverage
  remains required after third-party review and publication of a PR branch.
