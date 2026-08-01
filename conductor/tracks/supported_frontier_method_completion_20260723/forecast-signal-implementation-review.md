# Forecast-Signal Information Independent Implementation Review

Date: 2026-08-01

Scope: issue #572 and native delivery issues #759, #760 and #762; PR #770
range `cb874718..708ea334`, plus the bounded review remediations recorded in
this pull request.

## Outcome

No blocking correctness, contract, security or maturity finding remains after
review remediation. This is an independent implementation review, not the
independent scientific approval required for promotion.

## Estimand and decision semantics

- The baseline is the best feasible action under the declared outcome prior.
- The timely oracle uses the posterior induced by the declared joint law
  `P(outcome) P(signal | outcome)`. The deployed policy is selected from the
  reported conditional probabilities and evaluated under that joint law.
- Gross deployed value is the signed difference from the baseline. Calibration
  loss is the nonnegative difference between timely-oracle value and the timely
  deployed-policy value. Predictive accuracy is retained only as a diagnostic.
- Regret is measured against outcome-clairvoyant feasible action choice. The
  avoided-regret identity therefore equals signed gross deployed value.
- Maximum price is `max(0, gross deployed value)` and net deployed value
  subtracts the declared acquisition cost in the same objective unit.

The analytical newsvendor fixture, no-skill and perfect-signal limits,
miscalibrated harmful forecast, complete ties, action/signal/outcome
permutations and infeasible-action exclusion are consistent with these
definitions.

## Timing and failure boundaries

The result defines horizon as `outcome_time - forecast_origin`, forecast age at
decision (freshness) as `decision_time - forecast_origin`, latency as
`information_available - forecast_origin`, and lead time as
`outcome_time - decision_time`. Information arriving after the decision or
older than the declared maximum freshness falls back to the baseline policy,
so operational gross value is zero while the counterfactual timely-oracle
diagnostic remains visible.

Review added fail-closed semantic checks and tests for non-finite timing,
cost and tolerance values accepted through the Python mapping surface. Existing
checks reject invalid probability mass, inconsistent IDs/maps, unit mismatch,
invalid chronology, non-finite payoffs and zero-probability signal partitions.

## Surfaces, governance and maturity

The portable schemas are strict v1 experimental envelopes; the Python API and
Typer command serialize the same contract. Capability discovery correctly
reports Python as executable, Rust/R/Julia as unsupported and Mojo as external.
No stable or polyglot claim is made.

Canonical requirement M23 is MoSCoW Must for planned v1.3.0 under C18. Live
verification found C18 on `vop_poc_nz/main`, created by merged PR #69 and
corrected by merged PR #70, with M23 and #572 present. VOIAGE records #572 and
native children #759/#760/#762. Hosted exact-head evidence, independent
scientific review, Rust/R/Julia parity, stable promotion, release and issue
closure remain open gates.

## Remediation and validation

Review repaired the comprehensive CLI registry, refreshed all governed hashes
for shared files changed by the new CLI/API/docs surface, and normalized the
supported-frontier metadata. The repository normalizer is now idempotent.

- Five hosted regression tests plus the focused #572/governance suite:
  63 passed.
- Ruff check and format: passed.
- `ty` and BasedPyright on the new contracts/runtime: passed.
- Frontier contract validator: passed.
- Full Conductor validator: 146 tracks, zero errors and zero warnings.
- Conductor registry normalizer dry run: zero changed paths.

A local focused coverage rerun was blocked before collection by the existing
NumPy error `ImportError: cannot load module more than once per process` when
the coverage plugin loaded. Ordinary focused tests pass; restarted hosted
coverage remains the authoritative coverage gate.

## Applicable guide disposition

- Google Python Style Guide summary, language rules/style/naming: Pass for
  `voiage/contracts/forecast_signal_information.py`,
  `voiage/methods/forecast_signal_information.py` and their tests, as evidenced
  by Ruff, formatting and both type checkers.
- Mobile and browser guidance: Not applicable; this change exposes schemas,
  Python and a non-interactive CLI, not an interactive browser interface.
