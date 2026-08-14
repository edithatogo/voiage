# Independent implementation review — outcome-conditional sample information

Issue: [#600](https://github.com/edithatogo/voiage/issues/600) and delivery
subissues #790–#792 under programme #318.

Review date: 2026-08-01. Reviewed maturity: experimental Python implementation
planned for v1.3.0 under canonical Should requirement M31. This is a fresh
independent repository implementation review. It is not independent scientific
approval, hosted exact-head or installed-wheel evidence, continuous-estimator
validation, risk-sensitive composition, polyglot parity, stable promotion,
release evidence, merge authorization or issue closure.

## Scope reviewed

- canonical M31 and M31-U1–M31-U4, the Mermaid design and F600-1–F600-4 plan;
- the frozen primary-reference review and its explicit Equation 10 divergence;
- strict input/result schemas and semantic reconstruction validator;
- the exact finite evaluator, public Python API, CLI and registry surfaces;
- the normative input/result fixture, capability and fixture manifests;
- value units, conditioning chronology, utility/maximize and loss/minimize
  direction, fixed cost placement, prospective/retrospective scope, complete
  ties and policy-switch diagnostics;
- adversarial result mutations, identifier and probability pathologies,
  permutation determinism, outcome splitting, lower-tail endpoints and value-
  unit scaling from `1e-12` through `1e6`.

The review compared executable output with the frozen equations and independently
recomputed finite sums. A committed schema, fixture or method label was not
treated as numerical assurance by itself.

## Equations and assumptions checked

For baseline-optimal reference action `a*`, finite outcome `x`, direction sign
`d` and posterior action values `V_x(a)`, the implementation computes

\[
\Delta EV_x=d\{\max_d V_x(a)-V_0\},\qquad
VSI_x=d\{\max_d V_x(a)-V_x(a^*)\}.
\]

The reference action must be an exact baseline optimizer. Declared tolerance
ties are presentation and switch diagnostics; they do not turn a suboptimal
reference into the exact optimizer required by the frozen estimand.

The reviewed aggregate computes

\[
EVSI=\sum_x p(x)VSI_x=\sum_x p(x)\Delta EV_x,
\]

as an expectation-only tower identity. Negative outcome-level `delta_ev` with
positive `vsi` is retained. Equation 10 is the predictive-weighted population
functional

\[
\sigma_{VSI}=\sqrt{\sum_xp(x)(VSI_x-EVSI)^2},\qquad ddof=0,
\]

not the supplement's unweighted sample standard deviation. Equivalent outcome
splitting preserves the distributional metrics.

`rVSI_delta` uses the inclusive event `VSI_x <= delta`. Its zero-threshold mass
is kept distinct from reference exclusion, mandatory switching and complete-
tie-set change. Weighted quantiles use the finite inverse-CDF convention;
lower-tail mass zero uses the limiting essential minimum. Information cost is
subtracted after the gross distribution, so every outcome's `net_vsi` and the
aggregate `net_evsi` use the same fixed declared cost without changing gross
risks, quantiles or dispersion.

## Findings by severity

### Critical

None.

### High

None open. One adversarial finding was fixed in signed remediation commit
`aaf77aaf15fedcdabc0eab0c91df2eeaafff2fb5`:

- The original evaluator used the dimensionless `probability_tolerance` to
  zero value-unit `delta_ev` and `vsi`, and used a fixed `1e-15` cutoff on
  squared-value Equation 10 variance. An equivalent model scaled to small
  units therefore reported zero dispersion. Value quantities are no longer
  truncated by probability tolerance, variance is retained at its natural
  scale, and tower residual cleanup is scale-aware. Regression cases cover
  scales from `1e-12` through `1e6` and require positive, correctly scaled
  sigma and variance.

A second independent PR review found and signed remediation commit
`6eac2ba2e3896229cd2969591135063e7e12771b` fixed a probability-coherence
defect:

- Prior and likelihood-row totals within `probability_tolerance` were accepted
  but evaluated without normalization, so valid-by-contract near-unit vectors
  could violate the tighter tower identity or leave non-unit predictive mass.
  The evaluator now normalizes every accepted vector before calculation,
  retains the original committed input, and reports the pre-normalization
  residuals and whether normalization was applied. Boundary cases on both
  sides of one are covered; vectors outside tolerance still fail closed.

### Medium

None open. Two unit/contract inconsistencies were fixed with the High finding:

- `tie_tolerance` is an absolute tolerance in `value_unit`; its former
  dimensionless-looking `1e-6` upper cap was removed from runtime and both
  standalone schema projections.
- the frozen exact baseline-reference rule formerly admitted values within an
  undeclared absolute `1e-12`; it now requires the reference's computed
  baseline value to equal the exact optimum. A near-optimal counterexample is
  rejected.

The lower-tail mass-zero limiting convention is now stated explicitly in both
the portable contract README and the method documentation.

The second independent PR review also found that canonical M31-S1 described
the reference as optimal under tie tolerance while runtime correctly required
the exact extremum needed by the EVSI tower identity. M31-S1, the standalone
and embedded input schemas, the portable README and traceability tests now all
state that the reference must attain the exact baseline extremum and that
`tie_tolerance` controls tie-set/presentation diagnostics only.

### Low

None open in the reviewed implementation scope.

## Reconstruction and adversarial assurance

The result embeds the complete input and its canonical SHA-256 digest. The
standalone semantic validator verifies the digest, re-evaluates from that input
and requires canonical equality with the submitted result. Mutations of EVSI,
sigma, net EVSI, outcome VSI/delta, optimal ties, low-value risk, quantiles,
lower tails, switch probability, scope, baseline reference, `ddof` and the input
digest all fail closed. This assurance reconstructs the submitted exact finite
contract; it does not validate the scientific truth of submitted probabilities
or action values.

## Validation evidence

- 75 focused outcome-contract, exact-evaluator, API, CLI and schema tests pass.
- 50 frontier registry, governance, fixture and package-export tests pass.
- Ruff and BasedPyright pass with zero findings on changed Python files.
- frontier-contract and GitHub cross-reference validators pass.
- full Conductor validation passes for 148 tracks with zero errors and zero
  warnings.

An isolated Python 3.13 source environment reports 244/244 evaluator statements
and 92/92 evaluator branches plus 42/42 contract statements and 4/4 contract
branches covered. The remediated CLI save-status line and branch are covered.
The source checkout still lacks the native Rust extension required by unrelated
stable CLI tests, so installed-wheel and broad stable-runtime assurance remain
correctly assigned to hosted exact-head checks.

## Disposition

No Critical, High or Medium implementation finding remains open. The exact
finite experimental Python delivery is ready for hosted exact-head and
installed-wheel review. Continuous outcomes, fitted-estimator calibration,
independent scientific validity, risk-sensitive composition, Rust/R/Julia
parity, stable promotion, release and parent/umbrella closure remain open gates.
