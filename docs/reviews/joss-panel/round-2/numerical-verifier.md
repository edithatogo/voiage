# Independent numerical-verifier report: round 2

Reviewed repository state: `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`
plus the uncommitted shared-worktree changes present on 24 July 2026.

Recommendation: **major revision before submission**

Score: **890/1000**

Fail-closed status: **scientific estimator overclaim and immutable-version
blockers present; score capped below 950**

This is an internal AI-assisted numerical review, not a JOSS editorial
decision.

## Executive finding

The fixed-seed health example is internally coherent. I independently
reproduced its probability of choosing the programme, EVPI, regression-based
EVPPI estimates, analytical normal--normal EVSI values, population scaling,
research costs, ENBS values, and the two reported crossover intervals. The
current CSV files and manuscript values agree with those calculations.

The new Rust analytical normal--normal EVSI kernel implements the correct
closed-form calculation for the model it declares. Its normal-CDF
approximation introduces negligible error for the examined range. The Python
health script now calls that Rust-owned kernel rather than maintaining a
separate paper-only formula.

The generic `evsi(..., method="two_loop")` path is not yet a defensible
general or “exact” two-loop estimator. It estimates separate marginal normal
priors for arm means from arbitrary PSA samples, discards prior dependence
between arms during posterior simulation, and does not reject non-normal or
correlated priors. In a deliberately simple correlated bivariate-normal
reference problem, it returned approximately 224--245 per person while the
correct conjugate multivariate-normal EVSI was 1.36. This is a scientific
contract defect, not ordinary Monte Carlo variation.

The new two-loop reference test also does not establish recovery of the stated
124.179 benchmark. Its finite generated prior has an incremental-effect mean of
0.058393 rather than 0.060000, so the moment-fitted estimator's own analytical
target is approximately 88.36. Five 400-by-400 runs returned 81.45, 90.56,
99.97, 107.48, and 97.65. The single asserted run passes only because the test
allows an absolute error of 35 around 124.179. The test therefore conflates
prior-sample error, outer-loop error, inner-loop error, and estimator validity.

The JOSS manuscript is more restrained than the API documentation, but it
still presents EVSI as a package calculation and describes only estimators
“without a declared likelihood” as non-stable. That wording does not disclose
that the default two-loop path silently imposes independent marginal-normal
arm priors. The valid analytical function should remain the only stable EVSI
claim until the generic two-loop API either preserves the declared joint prior
or rejects unsupported inputs and documents its complete model.

## Independent analytical reconstruction

### EVPI and EVPPI

For the health example, incremental net benefit is

\[
  B = 50{,}000 Q - C,
\]

where \(Q \sim N(0.06, 0.03^2)\), \(C \sim N(3000, 650^2)\), and the generated
draws are independent. The population distribution therefore has mean zero
and standard deviation

\[
  \sqrt{(50{,}000 \times 0.03)^2 + 650^2}.
\]

The corresponding population analytical quantities are:

| Quantity | Independent population value |
| --- | ---: |
| EVPI | 652.182171953 |
| EVPPI for health effect | 598.413420602 |
| EVPPI for programme cost | 259.312482261 |

Using NumPy's generator directly with seed `20260723` and 10,000 draws, then
performing the reductions and ordinary least-squares EVPPI calculation without
calling `voiage`, produced:

| Quantity | Independent fixed-seed value | Tracked CSV | Manuscript |
| --- | ---: | ---: | ---: |
| Probability programme preferred | 0.492400000 | 0.492400000 | 49.24% |
| EVPI | 644.153547435 | 644.15354743 | 644 |
| Effect EVPPI | 589.666167416 | 589.66616742 | 590 |
| Cost EVPPI | 249.594993528 | 249.59499353 | 250 |

The fixed-seed numbers are correct estimates for the generated PSA sample.
They are not the closed-form population values. Lines 120--123 of `paper.md`
should call all four values fixed-seed estimates, rather than attaching
“fixed-seed” grammatically only to the preference probability.

The Rust EVPI reduction is the standard
\(E[\max_d NB_d]-\max_d E[NB_d]\). The Rust EVPPI path uses separate
ordinary least-squares regressions by strategy. For this linear synthetic
model, those fitted conditional means reproduce the intended regression
estimator. The reported EVPPI values should not be described as exact EVPPI
for arbitrary nonlinear models; the current manuscript does not make that
stronger claim.

### Analytical normal--normal EVSI

Let the incremental effect have prior variance \(\tau^2\). For an
equal-allocation two-arm study of total size \(n\), with known individual
outcome variance \(\sigma^2\), the sampling variance of the difference in arm
means is \(4\sigma^2/n\). The preposterior variance of the posterior mean is

\[
  v_{\mathrm{pre}} =
  \frac{\tau^4}{\tau^2 + 4\sigma^2/n}.
\]

If incremental net benefit is \(a\theta+b\), its preposterior mean is normal
with mean \(m=a\mu+b\) and standard deviation
\(s=|a|\sqrt{v_{\mathrm{pre}}}\). Thus

\[
 \mathrm{EVSI}
 = s\phi(m/s)+m\Phi(m/s)-\max(0,m).
\]

The Rust implementation at
`rust/crates/voiage-numerics/src/evsi_normal_normal.rs` matches this derivation.
For the health example \(m=0\), and I independently obtained:

| Total sample size | EVSI per person |
| ---: | ---: |
| 50 | 63.117286371 |
| 100 | 88.768917852 |
| 200 | 124.179365521 |
| 400 | 171.952831105 |
| 800 | 233.720374634 |
| 1,200 | 275.918834249 |

These values match the tracked result CSV. The 200-participant statement at
lines 122--123 is correct for the declared analytical model.

### Rust approximation accuracy

The Rust kernel uses the Abramowitz--Stegun error-function approximation. On
an independently evaluated grid of standard-normal \(z\) values from -20 to
20 in increments of 0.001:

- maximum absolute CDF error was \(6.97\times10^{-8}\);
- maximum absolute error in \(E[\max(0,X)]\) for unit standard deviation was
  \(2.11\times10^{-7}\); and
- two off-centre health-scale cases differed from the `erf` reference by
  approximately \(6.35\times10^{-5}\) value units.

This is negligible for the reported example. However, the sole exact-value
Rust test uses \(m=0\), where the CDF term is multiplied by zero. It therefore
does not test the CDF approximation that dominates off-centre cases. Add an
independent grid or several prespecified positive- and negative-\(z\)
references, including a negative net-benefit slope.

## ENBS and crossover verification

The immediate population opportunity count is

\[
1300\sum_{y=1}^{10}1.03^{-y}=11089.263687809.
\]

For the delayed scenario, 60% uptake and no benefit in years 1--2 give

\[
0.6\times1300\sum_{y=3}^{10}1.03^{-y}=5161.051850163.
\]

Research cost is \(1{,}200{,}000+100n\). Independent subtraction of research
cost from population EVSI produced:

| \(n\) | Immediate/full uptake | Two-year delay/60% uptake |
| ---: | ---: | ---: |
| 50 | -505,075.77 | -879,248.41 |
| 100 | -225,618.06 | -751,859.01 |
| 200 | 157,057.73 | -579,103.86 |
| 400 | 666,830.29 | -352,542.52 |
| 800 | 1,311,786.86 | -73,757.03 |
| 1,200 | 1,739,736.71 | 104,031.41 |

The manuscript's statements that immediate/full-uptake ENBS becomes positive
between 100 and 200 participants and delayed/60%-uptake ENBS becomes positive
between 800 and 1,200 participants are correct for the evaluated sample-size
grid. “Between” should continue to be understood as a grid crossover, not as a
continuous sample-size optimisation result.

## Generic two-loop posterior simulation

### What is correct

For separate independent normal arm-mean priors with a common known outcome
standard deviation, the update in `_bayesian_update()` uses the correct
normal--normal posterior mean and variance. `_evsi_two_loop()` now uses
`n_inner_loops` as the number of posterior draws for each simulated data set,
and seeded execution is repeatable without consuming NumPy's global random
stream. The focused callback-count and reproducibility tests exercise those
software properties.

### Scientific mismatch

The public API accepts a general `ParameterSet`, but the implementation:

1. replaces every `mean_<arm>` prior by its empirical marginal mean and
   standard deviation;
2. assumes a normal marginal prior regardless of the supplied distribution;
3. simulates each posterior arm mean independently;
4. does not carry the covariance among arm means into the posterior; and
5. does not validate or declare those independence and normality assumptions.

To isolate the consequence, I generated a bivariate-normal prior with marginal
arm standard deviations near 0.03, incremental-effect mean 0.06, and
incremental-effect standard deviation 0.00304 because the arm means were
strongly positively correlated. With 100 observations per arm and the same
linear net-benefit model:

- the correct multivariate-normal conjugate calculation gave EVSI
  **1.3631**;
- three package runs with 1,000 outer and 1,000 inner draws gave
  **238.88**, **244.60**, and **224.00**.

The discrepancy is caused by discarding prior covariance, not by simulation
noise. A generic study-specific two-loop estimator needs either a model-owned
posterior callback/joint prior update or a narrow explicit contract that
rejects unsupported dependence and distributional forms.

### Defect in the new recovery test

`tests/test_evsi_scientific_contract.py` draws only 2,000 prior samples. Their
realised incremental-effect mean is 0.058393 and the corresponding realised
incremental-net-benefit mean is -80.35, not zero. Under the estimator's fitted
marginal-normal assumptions, the analytical target is approximately 88.36,
not 124.179. Results across five seeds were 81.45--107.48.

The assertion `124.1793655206238 ± 35` is therefore too permissive and targets
the wrong finite-prior problem. It can pass an estimator that does not recover
the stated reference. A valid test should use a deterministic prior design
whose relevant moments and dependence are exact, calculate a reference for
that finite design, report a Monte Carlo standard error or repeated-run
coverage criterion, and include a test that unsupported prior dependence
fails closed.

## Test evidence

The following checks completed successfully:

- the complete `voiage-numerics` Rust test suite: 64 tests;
- the three dedicated Rust normal--normal EVSI tests;
- the Python basic-method, sample-information, scientific-contract, and
  health-example focused suites;
- the native runtime-adapter focused suite: 18 tests; and
- independent formula and simulation scripts described above.

Green tests do not resolve the two-loop defect because no current test supplies
a correlated joint prior or a non-normal prior. The analytical Rust test also
does not exercise nonzero \(z\).

## Exact manuscript findings

| Lines in `paper.md` | Assessment | Required action |
| ---: | --- | --- |
| 30--37 | Correct high-level definitions | Retain. |
| 39--45 | Partly overbroad | The analytical normal--normal EVSI function is valid, but “calculates these measures” can be read as a general stable EVSI claim. Name the analytical model as the stable EVSI scope. |
| 97--104 | Material omission | State that the generic two-loop path currently assumes separate independent normal arm-mean priors estimated from PSA samples, or classify it as non-stable until that contract is enforced. “Estimators without a declared likelihood” is insufficient because a likelihood alone does not define the prior or posterior. |
| 106--110 | Supported with coverage gap | The listed assurance categories exist, but they do not currently test joint-prior preservation in two-loop EVSI or off-centre analytical-kernel accuracy. |
| 120--123 | Numerically correct but imprecise about estimation | Replace with wording such as: “In the same 10,000-draw fixed-seed sample, the estimated EVPI was 644 value units per person, while regression-based EVPPI estimates were 590 for health effect and 250 for programme cost.” |
| 122--126 | Correct for the declared analytical model and evaluated grid | Retain after adding a short clause giving the normal prior SD, known outcome SD, equal allocation, and fixed expected cost, or point to a compact assumptions table. |
| 126--128 | Restrained interpretation | Retain. |
| 157--162 | Reproducibility blocker | The described kernel and tests exist only in the dirty worktree and not in cited release 1.0.0. Cite the immutable release and commit that actually contain the verified implementation and regenerated evidence. |

Suggested scientific-boundary wording:

> The stable EVSI calculation currently covers an equal-allocation two-arm
> study with a normal prior for the incremental effect, known common outcome
> variance, and incremental net benefit linear in that effect. Other EVSI
> interfaces remain non-stable unless their complete sampling model, joint
> prior, and posterior update are declared and validated.

## Rubric score

| Dimension | Score | Deduction |
| --- | ---: | --- |
| Scope, significance, and research use | 168/180 | The software scope and worked decision are concrete, but both demonstrations are author-created and no completed external research use is evidenced. |
| Statement of need and audience | 116/120 | The decision problem is clear; the stable EVSI scope is not explicit enough for an applied reader selecting an estimator. |
| State of the field and build-versus-contribute case | 124/130 | Relevant alternatives and the shared-kernel rationale are treated fairly; numerical-method depth is not compared in enough detail to show where `voiage` remains narrower. |
| Scientific and numerical accuracy | 118/150 | Health arithmetic and the analytical Rust kernel are correct. Major deductions apply because the generic two-loop estimator silently discards prior dependence, is called exact in its docstring, and has a misleading recovery test. |
| Software design and research relevance | 93/100 | The Rust boundary is meaningful and the analytical EVSI contract is well located; the generic posterior-update interface cannot faithfully represent the joint prior accepted by the public API. |
| Reproducibility, packaging, documentation, and tests | 78/100 | Strong focused and Rust coverage, but missing correlation/non-normality and off-centre-kernel tests; the verified implementation is uncommitted and absent from release 1.0.0. |
| Research-impact statement | 61/80 | The synthetic result is informative and honestly labelled, but it is a demonstration rather than realised independent impact. |
| Structure, metadata, and JOSS format | 59/60 | The required numerical narrative is concise and appropriately placed; the assumptions would benefit from one compact table or clause. |
| Clarity, accessibility, and sentence quality | 52/55 | Results are readable, but “normal--normal,” the regression status of EVPPI, and the distinction between analytical and generic two-loop EVSI need plainer explanation. |
| Citations, provenance, declarations, and AI disclosure | 21/25 | Numerical sources and declarations are present; the exact reviewed software revision and release remain future-tense. |
| **Total** | **890/1000** | **Major revision; fail-closed cap applies.** |

## Blocking actions

1. Remove the generic “exact two-loop” claim, or implement a posterior
   interface that preserves the declared joint prior and likelihood.
2. Make unsupported normality or prior dependence fail closed; at minimum,
   document and enforce the complete independent-normal arm-prior contract.
3. Replace the permissive 124.179 two-loop recovery test with a correctly
   targeted, error-budgeted reference test.
4. Add a correlated-prior rejection or correct multivariate-posterior test.
5. Add off-centre and negative-slope analytical Rust reference cases.
6. Identify the fixed-seed health values explicitly as estimates and state the
   analytical study assumptions beside the EVSI result.
7. Commit, run hosted checks, release, and cite the exact immutable revision
   containing the verified kernel, manuscript, tests, and generated evidence.

The health-example numbers themselves do not need to change. The required
revision is to the generic estimator contract, its evidence, and the precision
of the manuscript's scope claims.
