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

The finite contract is based on Akinlotan et al. (2024), DOI
`10.1016/j.ecolind.2024.111828`, arXiv `2309.09452v2`. It does not claim
continuous-outcome integration, fitted-estimator calibration, dynamic value,
underlying-system risk, scientific approval, stable maturity or cross-language
parity. Rust, R and Julia are unsupported; Mojo is external.
