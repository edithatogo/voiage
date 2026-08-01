# #600 outcome-conditional sample-information reference review

## Scope and source

Issue #600 and native delivery children #790–#792 adopt the four metrics in
Akinlotan et al., *Ecological Indicators* 160 (2024) 111828,
doi:10.1016/j.ecolind.2024.111828 (arXiv:2309.09452v2). This is a bounded
primary-source mapping, not an independent scientific approval or systematic
review.

## Frozen finite estimands

For baseline-optimal declared reference action `a*`, measurement outcome `x`,
and direction-aware posterior objective values, the finite v1 contract uses:

- `delta-EV_x`: the direction-aware shift from the baseline optimal expected
  system objective to the posterior optimal expected objective;
- `VSI_x`: the direction-aware posterior gain from replacing `a*` by a
  posterior-optimal action, which is nonnegative under the classical matched
  action/value assumptions;
- `EVSI = sum_x p(x) VSI_x = sum_x p(x) delta-EV_x`; and
- `rVSI_delta = sum_x p(x) 1[VSI_x <= delta]` for declared `delta >= 0`.

The two tower identities are expectation-linear statements only. They do not
imply equal outcome-wise values, variances, standard deviations, quantiles or
tails. Negative `delta-EV_x` with positive `VSI_x` is therefore required test
evidence rather than treated as a contradiction.

## Equation 10 and source discrepancy

Equation 10 defines the standard deviation of the predictive distribution of
`VSI_x`. The governed implementation therefore uses the probability-weighted
population functional
`sqrt(sum_x p(x) (VSI_x - EVSI)^2)` with `ddof = 0`. The supplementary MATLAB
code and reported Table 3 values use an unweighted sample standard deviation
over enumerated outcomes. That calculation changes when outcome categories are
split and is not Equation 10 for unequal predictive probabilities, so it is
recorded as source-implementation divergence and is not copied.

## Tie and maturity boundaries

`rVSI0` is probability mass at zero decision value for the declared baseline
reference action. With baseline or posterior ties it is not, in general, the
same as reference-action exclusion, mandatory-switch mass or complete-tie-set
change mass; all are reported separately. The exact finite evaluator does not
establish continuous-outcome integration, fitted likelihood calibration,
dynamic/adaptive information value, underlying-system risk, scientific
validity, stable maturity, polyglot parity, release or issue closure.
