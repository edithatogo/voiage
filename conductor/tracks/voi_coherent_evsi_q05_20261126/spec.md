# Specification: Q05 Coherent EVSI Qualification

Covers roadmap packages **V03** (regression & moment-matching EVSI), **V04** (importance-sampling EVSI), **V05** (non-normal & survival/multi-arm study models), **V07** (multilevel, QMC & unbiased estimators), and **F02** (modern inference & diagnostic boundary).

## Acceptance Criteria
1. Reproduce `Q03` conjugate Normal-Normal, Beta-Binomial, and tail-certified Gamma-Poisson EVSI cases across nested, regression, moment-matching, and importance-sampling estimators with explicit per-arm vs. total sample-size distinction.
2. Fail closed on mismatched priors, invalid joint updates, malformed callbacks, and importance-weight degeneracy (retaining raw estimates and ESS/PSIS diagnostics without silent clamping).
3. Report bias, RMSE, coverage (when intervals are supplied), ESS, decision error, runtime, and memory across prespecified seeds and budgets.
4. Admit multilevel/QMC/unbiased extensions (`V07`) only with reference correctness and an accuracy-vs-cost comparison against plain Monte Carlo.
