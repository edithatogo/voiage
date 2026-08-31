# Synthetic health example: methods and equations

This note records the decision model, analytical study model, and Monte Carlo
sampling procedure behind the worked example in `paper.md`. All data are
synthetic. The example demonstrates a calculation; it does not represent a
clinical outcome model or recommend a study design.

## Decision model

The uncertain incremental health effect and programme cost are generated
independently as

\[
\delta \sim N(0.06, 0.03^2), \qquad
C \sim N(3000, 650^2).
\]

At a willingness-to-pay threshold \(\lambda=50{,}000\) value units per
quality-adjusted life year, incremental net benefit is

\[
B = \lambda\delta-C.
\]

The generating mean of \(B\) is zero and its standard deviation is
1,634.778 value units. The theoretical probability that the programme is
preferred is therefore 0.5. The fixed 10,000-draw sample has mean incremental
net benefit of -15.860 value units because of Monte Carlo sampling error. EVPI
and EVPPI are estimated from that sample, whereas EVSI uses the declared
generating prior. The generating-distribution benchmarks are 652.182 value
units for each eligible future person for expected value of perfect
information, 598.413 for expected value of partial perfect information about
health gain, and 259.312 for expected value of partial perfect information
about programme cost.

## Analytical study model

The uncertain incremental effect has the prior

\[
\delta \sim N(\mu_0,\tau^2).
\]

This is an algebraic study-model illustration rather than a proposed trial. It
does not specify follow-up, QALY measurement, missing data, treatment switching,
or mapping from observed outcomes into a clinical decision model. For a
two-group normal study with equal allocation, total sample size \(n\), and known
common individual outcome variance \(\sigma^2\), the difference in sample means
follows

\[
\bar Y_1-\bar Y_0\mid\delta \sim N(\delta,4\sigma^2/n).
\]

The variance of the posterior mean before observing the study result is

\[
v_{\mathrm{pre}} =
\frac{\tau^4}{\tau^2+4\sigma^2/n}.
\]

When incremental net benefit is \(a\delta+b\), the possible posterior means
before any study result is observed have a preposterior distribution. This
distribution is normal with mean \(m=a\mu_0+b\) and standard deviation
\(s=|a|\sqrt{v_{\mathrm{pre}}}\). In this example,
\(a=\lambda=50{,}000\) and \(b=-E[C]=-3{,}000\). Cost uncertainty is
independent of the study and is integrated over its mean when the post-study
decision is made; its variance therefore does not enter this study-specific
EVSI calculation. The two-option expected value of sample information is

\[
\operatorname{EVSI}
=s\phi(m/s)+m\Phi(m/s)-\max(0,m),
\]

where \(\phi\) and \(\Phi\) are the standard normal density and distribution
functions. When \(s=0\), the expression is evaluated by continuity and EVSI is
zero. For the worked example, \(\sigma=1.0\) QALY and \(n=200\) give EVSI of
124.179 value units for each eligible future person whose decision could use
the evidence.

## Population value and study cost

Benefits accrue at the end of each year, whereas study costs occur at time
zero. For annual population \(N\), time horizon \(T\), discount rate \(i\),
delay \(d\), and realised proportion \(r\), discounted opportunities are

\[
O(d,r)=rN\sum_{t=d+1}^{T}(1+i)^{-t}.
\]

All monetary quantities are undated synthetic units and imply no jurisdictional
or payer perspective. The 3% rate discounts future decision opportunities, not
the generated health and cost outcomes. Immediate use sets \(d=0\) and \(r=1\),
discounting 1,300 eligible decisions per year for ten years and giving
11,089.264 discounted opportunities. The delayed scenario sets \(d=2\) and
\(r=0.60\), omitting the first two years and giving 5,161.052 discounted
opportunities. Delay is fixed independently of sample size; recruitment,
follow-up, and reporting time are not modelled. For fixed study cost \(F\),
per-participant cost \(c\), and total sample size \(n\),

\[
C_{\mathrm{study}}(n)=F+cn,
\qquad
\operatorname{ENBS}(n;d,r)=\operatorname{EVSI}(n)O(d,r)
-C_{\mathrm{study}}(n).
\]

Here, \(F=1.2\) million value units and \(c=100\) value units.

Expected net benefit of sampling is population EVSI minus study cost. The
evaluated sample sizes are 50, 100, 200, 400, 800, and 1,200. These discrete
evaluations bracket sign changes but do not estimate a continuous optimum.

## Monte Carlo sampling intervals

The EVPPI point estimates use ordinary least squares. Net benefit is linear in
each independently generated input in this example, so the regression model is
correctly specified here. This does not validate the estimator for nonlinear or
correlated models. Its target is the expected gain from learning each selected
input under the finite probabilistic-sensitivity-analysis sample; it is not an
exact finite-sample conditional decomposition. Each bootstrap replicate refits
that same estimator.

The EVPI, EVPPI, and preference-probability intervals describe Monte Carlo
sampling error in the 10,000 paired health-gain and cost draws. The procedure
resamples those pairs 1,000 times with replacement, recalculates each statistic,
and reports the 2.5th and 97.5th percentiles. The bootstrap seed is 20260724.

## Reproduction

Run:

```console
python scripts/verify_paper_reproduction.py --manifest paper/reproduction-manifest.json
```

The script, inputs, fixed random seeds, environment lock digest, output hashes,
and verification command are recorded in
`paper/reproduction-manifest.json`. Machine-readable results are in
`paper/data/`. The verifier uses an isolated checkout of the exact source commit
and the archived project and lock declared by the manifest. This is a new replay
selection; the original receipt is preserved separately and its `v2.0.0` source
label remains unverified.
