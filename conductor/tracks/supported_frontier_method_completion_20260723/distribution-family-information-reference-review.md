# Distribution-family information value reference and boundary review

Issue: [#557](https://github.com/edithatogo/voiage/issues/557).

## Frozen estimand and conditioning order

Let `M` be a finite, declared set of mutually exclusive candidate distribution
families or distributional assumptions with pre-information probabilities
`pi[m]`. Let `theta_m` denote the family-specific uncertain parameters and let
`U(a, m, theta_m)` be the value or negative loss for a common feasible action
`a`. Define the conditional action means

\[
\mu_{m,a}=E_{\theta_m\mid M=m}[U(a,m,\theta_m)].
\]

The v1 perfect distribution-family information estimand is

\[
V_0=\max_a\sum_m\pi_m\mu_{m,a},\qquad
V_M=\sum_m\pi_m\max_a\mu_{m,a},\qquad
VDI_M=V_M-V_0.
\]

The conditioning order is normative: first integrate parameter uncertainty
within each family, then optimize separately after learning `M`, and only then
average over the pre-information family probabilities. Moving `max` inside the
within-family expectation would additionally resolve `theta_m` and therefore
estimate more than distribution-family information. Averaging before the
resolved optimization recovers the current-information decision and produces
no information contrast.

This coherent finite-mixture construction establishes `VDI_M >= 0` without
clipping. With the same objective, feasible actions and joint mixture law it is
bounded above by perfect information about both `M` and `theta_M`. Net VDI is
`VDI_M - information_cost`, so it may be negative. Numerical ties must return
the complete tied action set under declared absolute and relative tolerances;
canonical name order is presentation only and cannot remove a tie.

## Relationship to EVPPI and existing structural helpers

When the discrete family indicator `M` is represented as one component of a
coherent joint uncertain state, `VDI_M` is mathematically EVPPI for that
discrete component. The dedicated contract is still necessary: candidate
families may have different parameter spaces, priors and evaluators, and their
probabilities and within-family conditioning must remain explicit. It is not
parameter EVPPI for a parameter inside one selected family.

The current `voiage.methods.structural.structural_evpi` implementation averages
samplewise action maxima within each structure. That resolves within-structure
parameter draws as well as the structure and therefore is not the v1 `VDI_M`
estimand above. Its `structural_evppi` subset/renormalization interface likewise
does not encode observation of a named family indicator across the complete
model partition. Neither helper may be relabelled or wrapped as #557 without
changing and independently assuring its conditioning contract.

Imperfect model-discrimination data would require a declared likelihood
`p(y | M, theta_M)`, posterior family probabilities and pre-posterior averaging;
that is a later EVSI design, not part of this perfect-information v1 slice.

## Terminology and scientific boundary

“Distributional information” can also mean information about the distribution
of health, costs or equity across population groups. Distributional
cost-effectiveness analysis evaluates those equity distributions and social
value judgements; it is not information about which probability family is
correct. Public surfaces should therefore pair the issue's VDI label with the
unambiguous canonical identifier `distribution-family-information-value` and
must not describe the result as distributional-equity VOI, subgroup
heterogeneity, robustness, goodness of fit, model-selection accuracy or a
Bayes factor.

The family set and probabilities are decision-model inputs, not truths inferred
by this estimator. Zero-probability families do not contribute, and omitted
plausible families are outside the estimand. Learning that one declared family
is correct is a strong perfect-information assumption; scientific promotion
requires review that the partition is mutually exclusive, sufficiently
exhaustive for the stated use and meaningful for the decision.

## Units, probabilities and provenance

All `U` values, conditional values, baseline value, resolved value, VDI and
information cost must share one declared value or loss unit, objective
direction, population scale, horizon, discount convention and cost location.
Positive scaling by `c` scales VDI by `c`; adding one action-independent
constant to every state cancels. Family or action permutation cannot change
the value or complete tie sets.

Family probabilities must be finite, non-negative, normalized under a declared
tolerance and bound to named family IDs. The runtime must not silently infer
equal weights, renormalize a partial family subset or treat information
criteria as probabilities. Provenance must record the probability source and
date, candidate-family definitions and parameterizations, priors/posteriors,
data and reference revision, evaluator and software versions, random-number
method and seeds, draw counts, and every transformation into the common value
unit. Alternative outputs must be row-aligned within each family. Cross-family
draws need not be paired, but any common-random-number mapping must be declared
rather than inferred.

## Estimator assurance

The first executable assurance set should include an enumerable analytical
mixture whose `mu[m,a]`, baseline action, family-conditional actions and VDI are
known exactly. A simulation estimator must report every conditional action
mean, Monte Carlo standard error or repeated-run interval, draw count by
family, convergence/stopping rule, tie/decision stability and arithmetic
residuals; a scalar VDI without these diagnostics is inadequate evidence.

Tests must cover family/action permutation, positive scaling and translation,
one-family and identical-optimum zero value, decision-switch positive value,
the perfect-information upper bound, signed net VDI, zero-probability families,
near ties, highly imbalanced probabilities, unequal family draw counts, and
NaN/Inf, duplicate IDs, invalid probabilities, missing alternatives,
within-family misalignment and unstable estimator evidence. Optimization bias
and Monte Carlo error must be assessed around both maxima; non-negativity must
follow from the same estimated conditional means rather than post-hoc clipping.

## Independent references

- Strong, Oakley and Brennan,
  [Estimating multiparameter partial expected value of perfect information from a probabilistic sensitivity analysis sample](https://pmc.ncbi.nlm.nih.gov/articles/PMC4819801/),
  gives the conditional-expectation/conditional-optimization definition of
  EVPPI and simulation-assurance context.
- Jackson et al.,
  [Accounting for uncertainty in health economic decision models by using model averaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC2667305/),
  treats structural alternatives through an explicit probability distribution
  over models and documents the assumptions behind model averaging.
- Price et al.,
  [Model averaging in the presence of structural uncertainty about treatment effects](https://pubmed.ncbi.nlm.nih.gov/21402291/),
  connects model averaging, treatment decisions and expected value of
  information under structural uncertainty.
- Heath et al.,
  [Value of Information Analytical Methods: ISPOR Task Force Report 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7373630/),
  distinguishes structural-uncertainty representations and requires the
  information calculation to match the decision model and uncertainty source.
- Cookson et al.,
  [Distributional Cost-Effectiveness Analysis: A Tutorial](https://pmc.ncbi.nlm.nih.gov/articles/PMC4853814/),
  establishes the separate equity-distribution meaning that creates the
  terminology collision addressed above.

Search limit: these sources establish the initial estimand, terminology,
assurance and boundary tests; this record is not a systematic review. Named
scientific review, model-partition approval, stable promotion, model-
discrimination EVSI and cross-language parity remain separate gates.
