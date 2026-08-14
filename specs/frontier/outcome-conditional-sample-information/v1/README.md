# Outcome-conditional sample-information value v1

This experimental Python-only contract evaluates a declared finite prior,
measurement likelihood and action-value table. For every measurement outcome
`x`, it returns the predictive probability, posterior state law, complete
optimal action ties, direction-aware `delta-EV_x`, and nonnegative `VSI_x`.
Utility is maximized and loss is minimized; all reported values share the
declared unit, population, horizon and discount basis.

The aggregate result verifies only the expectation-linear tower identities
`EVSI = E[VSI_x] = E[delta-EV_x]`. Outcome-wise values and distributional
functionals need not agree: the normative fixture deliberately contains a
negative `delta-EV_x` with positive `VSI_x`, and the standard deviations of the
two outcome metrics differ.

Equation 10 is implemented as
`sqrt(sum_x p(x) (VSI_x - EVSI)^2)`, the predictive-probability-weighted
population standard deviation with `ddof = 0`. The source supplement's
unweighted MATLAB/Table 3 sample standard deviation is not used. Splitting one
outcome label into probability-equivalent sublabels therefore leaves EVSI and
`sigma-VSI` unchanged.

`rVSI_delta` is the predictive probability that `VSI_x <= delta` for declared
nonnegative thresholds. Under ties, `rVSI0` is kept separate from the
probability that the reference action is excluded, that every baseline-optimal
action must switch, or that the complete optimal tie set changes. The result
also includes weighted quantiles, lower-tail means, cost placement, probability
and Bayes residuals, and a SHA-256-bound input copy used for independent result
reconstruction.

`tie_tolerance` is an absolute tolerance in the declared `value_unit`; it is
not a dimensionless probability tolerance and has no artificial unit-dependent
upper bound. It controls complete tie-set and presentation diagnostics only:
the declared reference action must attain the exact baseline extremum. Value
metrics and Equation 10 dispersion are never zeroed with the probability
tolerance. At lower-tail mass zero, v1 reports the limiting finite essential
minimum, matching the level-zero weighted quantile convention.

Prior and state-conditional likelihood vectors whose sums are within the
declared `probability_tolerance` are normalized before any value calculation.
The result retains the original input contract and reports the pre-normalization
prior and maximum likelihood-row residuals together with an explicit
normalization-applied flag. Vectors outside the tolerance fail closed.

The finite contract is based on Akinlotan et al. (2024), DOI
`10.1016/j.ecolind.2024.111828`, arXiv `2309.09452v2`. It does not claim
continuous-outcome integration, fitted-estimator calibration, dynamic value,
underlying-system risk, scientific approval, stable maturity or cross-language
parity. Rust, R and Julia are unsupported; Mojo is external.
