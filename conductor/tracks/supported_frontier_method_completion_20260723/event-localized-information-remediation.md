# Event-localized information independent-review remediation

## Boundary

This remediation responds to the independent review of C18/M27 issue #596 at
signed pre-remediation head `78237b213bf88b3ec40f23066f5cbf8563b98dd9`.
It does not approve the remediated implementation, establish hosted exact-head
or installed-wheel evidence, promote maturity, claim language parity, close
#596, or close the umbrella programme. A fresh independent reviewer must assess
the remediated head.

## Findings resolved

1. **High — executable schemas and result semantics:** The runtime previously
   accepted schema-invalid values such as Boolean costs and duplicate
   `state_set` identifiers and did not validate generated result identities.
   The embedded Draft 2020-12 input/result schemas are now the export source,
   runtime input validation is mandatory, and the result validator reconciles
   policies, probabilities, values, densities, integrals, modes, directions
   and assurance fields.
2. **High — false baseline under an unbounded tie tolerance:** A materially
   suboptimal reference action could replace the true baseline maximum and
   inflate both event and coordinate VOI. The baseline is now always the true
   maximum; a density reference must attain it exactly; tie tolerance is
   bounded to `[0, 1e-6]`.
3. **Medium — zero/unbounded integral tolerance and premature rounding:** A
   zero tolerance rejected valid floating calculations while a large tolerance
   could erase real VOI. Integral tolerance is now `(0, 1e-6]`; raw atoms drive
   assurance identities and cleanup is presentation-only.
4. **Medium — false binary-channel symmetry assurance:** Exact float-key
   lookup skipped valid decimal `p`/`1-p` pairs and reported zero when no pair
   existed. Pair matching is tolerant, residuals are enforced, and an
   unevaluated grid reports `null`.

## Remediation evidence

- Focused runtime, schema, mutation, plotting, export, package and programme
  tests: 98 passed.
- Branch coverage across the new contract, evaluator and plot modules: 92%
  aggregate; evaluator 96%; defensive impossible-state guards remain the only
  uncovered evaluator lines.
- Independent deterministic oracle: 250 randomized finite problems matched
  direct coordinate VOI, event/complement value, `p=0`, `p=0.5`, `p=1` and
  decimal `0.07`/`0.93` symmetry calculations.
- Ruff check and format: pass. BasedPyright: zero errors, warnings or notes.
- Frontier-contract and Conductor GitHub cross-reference validators: pass.
- Full bundled Conductor validation: 147 tracks, zero errors and zero warnings.
- `git diff --check`: pass.

The local test environment was reused without creating an isolated environment.
No branch was pushed and no pull request was opened by the remediator.
