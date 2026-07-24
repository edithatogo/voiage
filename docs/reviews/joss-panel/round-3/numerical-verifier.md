# Round 3 independent numerical-methods review

## Disposition

**Major revision; fail closed. Score: 934/1000.**

The numerical substance has improved materially since round 2. The analytical
normal--normal kernel implements the declared model correctly. The built-in
two-loop route now uses one coherent fitted joint Gaussian prior for the current
decision, prior-predictive observations, and posterior calculations. It
preserves correlation, uses genuine Gaussian draws, checks finite reductions,
and returns a negative Monte Carlo point estimate without truncating it.

I nevertheless do not consider the current JOSS candidate numerically
submission-ready. The manuscript describes an implementation and v2 contract
that are absent from the cited immutable release. In addition, the stable
analytical Python entry point leaks raw `TypeError` and `OverflowError`
exceptions for validly expressible but unsupported sample-size inputs, contrary
to the v2 error contract. These are evidence and public-contract defects rather
than errors in the central equations, but each is material for a reviewer
trying to install and verify the software named by the paper.

This is an independent technical review, not an editorial decision or an
acceptance prediction.

## Exact revision and worktree scope

The review was performed on 24 July 2026 against:

- repository: `/Volumes/PortableSSD/GitHub/voiage`;
- branch: `codex/joss-panel-review`;
- `HEAD`: `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`;
- worktree state: dirty;
- SHA-256 of the combined tracked diff restricted to the reviewed
  implementation, tests, v2 contract, and `paper.md`:
  `0c667f3a417d6472cd1811219f21be41d7a99769fbdedf6f6e689df336c73796`.

At the start of this review, the relevant tracked modifications were:

- `paper.md`;
- `voiage/methods/sample_information.py`;
- `voiage/_runtime.py`;
- `rust/crates/voiage-python/src/lib.rs`; and
- `tests/test_sample_information.py`.

The following reviewed files were untracked and therefore are not represented
by `HEAD`:

- `rust/crates/voiage-numerics/src/evsi_normal_normal.rs`;
- `rust/crates/voiage-numerics/tests/evsi_normal_normal.rs`;
- `tests/test_evsi_scientific_contract.py`;
- `tests/test_v2_stable_api_contract.py`; and
- `specs/v2/`.

File digests used to identify that uncommitted review state were:

| File | SHA-256 |
| --- | --- |
| `rust/crates/voiage-numerics/src/evsi_normal_normal.rs` | `ed0403ffda2de6a900059d1a41825bcc95e02f89346a843114669101f0c49002` |
| `rust/crates/voiage-numerics/tests/evsi_normal_normal.rs` | `0985dc696f5246ef3dec70ad9a2e46cff229fe74f5d51c039a35873630ef2d7f` |
| `tests/test_evsi_scientific_contract.py` | `d95f77f4fd8bc7b28981c80521d51c9e9735f8cb1c200b6b1fe49d0ce7d103ee` |
| `specs/v2/README.md` | `5f6d0cb6c35a67bf62c89dba713127c84c598035aeb99fd4c2feb03150bb9955e19c99` |
| `specs/v2/stable-api.json` | `8e8a4d2a07c1c7c9ba713127c84c598035aeb99fd4c2feb03150bb9955e19c99` |
| `specs/v2/compatibility-policy.json` | `d7a66a41b652f5681ed5f4a5f43193e069a45d1c872189588b6e4ac656596985` |
| `tests/test_v2_stable_api_contract.py` | `b86562135e996c46c3a7c05d45c176a73024b7026d08d80e69c70bd94c58ca12` |

I inspected the generic built-in and callback two-loop paths, the Rust
analytical implementation and PyO3 boundary, the scientific and API-contract
tests, the normative v2 files, and the corresponding claims in `paper.md`. I
did not modify runtime code.

## Independent equations and calculations

### Analytical normal--normal EVSI

Let the uncertain incremental effect be

\[
\theta \sim N(\mu,\tau^2).
\]

For a two-arm normal study with equal allocation, total sample size \(n\), and
known common individual outcome variance \(\sigma^2\), the observed difference
in sample means has variance

\[
v_y = \frac{\sigma^2}{n/2}+\frac{\sigma^2}{n/2}
    = \frac{4\sigma^2}{n}.
\]

The preposterior variance of the posterior mean is

\[
v_{\mathrm{pre}}
 = \operatorname{Var}_y\{E(\theta\mid y)\}
 = \frac{\tau^4}{\tau^2+v_y}.
\]

If incremental net benefit is \(a\theta+b\), its posterior mean before seeing
the data has marginal distribution \(N(m,s^2)\), where

\[
m=a\mu+b,\qquad
s=|a|\sqrt{v_{\mathrm{pre}}}.
\]

The two-strategy EVSI is therefore

\[
\operatorname{EVSI}
=s\phi(m/s)+m\Phi(m/s)-\max(0,m).
\]

The implementation at
`rust/crates/voiage-numerics/src/evsi_normal_normal.rs:88-101` matches these
equations. Independent SciPy calculations produced:

| Case | Independent EVSI | Rust/Python reference |
| --- | ---: | ---: |
| Health example, \(n=200\) | 124.1793655206238 | 124.1793655206238 |
| Positive slope, off-centre \(m=70\) | 27.70137907021558 | 27.70137907021558 |
| Negative slope, off-centre \(m=-70\) | 27.70137907021558 | 27.70137907021558 |

On a grid of 2,401 standardised means from \(-12\) to \(12\), the greatest
absolute difference between the kernel's normal-CDF approximation and SciPy's
reference EVSI was \(2.09\times10^{-7}\), at approximately \(z=3.08\).
That error is immaterial for the reported health result.

### Joint Gaussian observation and posterior model

For fitted prior \(N(\mu,\Sigma)\), observation matrix \(H\), and known
observation covariance \(R\), the implementation uses

\[
K=\Sigma H^\mathsf{T}(H\Sigma H^\mathsf{T}+R)^{-1},
\]

\[
\mu_{\mathrm{post}}=\mu+K(y-H\mu),
\]

and the Joseph covariance update

\[
\Sigma_{\mathrm{post}}
=(I-KH)\Sigma(I-KH)^\mathsf{T}+KRK^\mathsf{T}.
\]

The Cholesky solves at
`voiage/methods/sample_information.py:581-603` implement this gain and update
without explicitly inverting the predictive covariance.

For the three-parameter adversarial case used in the scientific tests, my
independent calculation gave posterior mean

\[
(0.1590940113,\ 0.4851475857,\ -0.2020337826).
\]

The Joseph covariance and the independent precision-form covariance,

\[
(\Sigma^{-1}+H^\mathsf{T}R^{-1}H)^{-1},
\]

agreed to a maximum absolute difference of
\(5.55\times10^{-17}\). The smallest posterior eigenvalue was
\(0.0109198\), so the reference case was positive definite rather than being
rescued by eigenvalue clipping.

### Prior coherence and nested integration

The repaired built-in path fits `prior_mean` and `prior_covariance` once at
`voiage/methods/sample_information.py:425-488`. It then:

1. draws the current-information integration sample from that fitted Gaussian
   at lines 772--788;
2. draws each study result directly from the matching prior-predictive
   distribution at lines 790--808; and
3. conditions the same fitted Gaussian and draws the posterior at lines
   809--824.

The earlier empirical-truth/fitted-Gaussian hybrid is therefore removed.

As an independent correlated-prior check, I fitted a two-arm Gaussian prior
with correlation 0.9 and compared the package with a conjugate calculation.
For the realised fitted prior, the analytical reference was
23.66494909. Four runs with 2,500 outer loops and 1,000 posterior draws gave
22.7655, 26.0747, 25.7171, and 25.5541. Their mean was 25.0278 and their
between-seed standard error was 0.7619; the difference from the reference was
1.79 such standard errors. This is consistent with the expected finite nested
Monte Carlo error and inner-loop decision-maximisation bias. It is not evidence
that arbitrary loop counts are sufficient for a new applied model.

### Genuine Gaussian sampling

`_positive_semidefinite_draws()` now generates ordinary independent standard
normal variates at
`voiage/methods/sample_information.py:491-520`. It no longer replaces them
with moment-matched or antithetic deterministic cohorts.

For one million independently generated unit-normal draws, I obtained:

- mean: -0.0003077;
- variance: 0.998984;
- fourth raw moment: 2.98527;
- proportion above 1.96: 0.024868; and
- maximum draw: 5.56389.

These results support the manuscript's “genuine Gaussian Monte Carlo draws”
wording at `paper.md:128`.

## Adversarial cases

| Case | Observed result | Assessment |
| --- | --- | --- |
| Strongly correlated prior | Joint posterior and two-loop estimate retained the fitted covariance | Pass |
| Unobserved but correlated third parameter | Its posterior mean and covariance changed consistently with conditioning | Pass |
| Near-unit correlations, \(\rho=\pm0.999\) | Finite posterior draws | Pass |
| Representable scales \(10^{-100}\) and \(10^{100}\) | Finite posterior draws | Pass |
| Fewer/equal PSA rows than uncertain dimensions | `InputError`, diagnostic `rank_deficient_prior` | Pass |
| Distinct arm labels normalising to the same parameter | `InputError`, diagnostic `arm_parameter_name_collision` | Pass |
| Boolean, fractional, zero, and negative loop counts | `InputError`, diagnostic `invalid_loop_count` | Pass |
| Non-constant, non-finite, non-positive, underflowing, or overflowing `sd_outcome` | Rejected | Pass |
| Finite \(10^{308}\) model entries whose mean overflows | `InputError`, diagnostic `non_finite_reduction` | Pass |
| One-dimensional Gaussian tail and fourth moment | Consistent with \(N(0,1)\) | Pass |
| Deliberately incoherent custom posterior yielding EVSI \(-0.5\) | Returned \(-0.5\) with `evsi_negative_monte_carlo_estimate` warning | Raw negative semantics pass; warning is incomplete |
| Analytical `total_sample_size=True` | Typed `InputError` | Pass |
| Analytical `total_sample_size=200.0` | Raw `TypeError` | Fail |
| Analytical `total_sample_size=-2` | Raw `OverflowError` | Fail |
| Built-in arm sample size \(10^{400}\) | Raw `OverflowError` during conversion to float | Fail |

## Earlier-defect reconciliation

| Earlier concern | Current evidence | Verdict |
| --- | --- | --- |
| Current value, outer truths, and posterior targeted different priors | All built-in integration stages now use the same fitted Gaussian at `sample_information.py:749-832` | **Fixed** |
| Correlated parameters were replaced by independent marginal updates | Matrix conditioning updates the full parameter vector at lines 562--619; correlated and three-parameter references pass | **Fixed** |
| “Gaussian” cohorts were moment matched rather than genuinely random | `rng.standard_normal` is used directly at lines 518--520; tail/fourth-moment checks pass | **Fixed** |
| Finite entries could overflow during strategy or final aggregation | `_finite_strategy_means` and final reductions reject non-finite results at lines 105--116, 825--831, 880--886, and 1245--1250 | **Fixed for the reviewed reduction paths** |
| Fitted covariance rank was not guarded | \(n_{\mathrm{PSA}}>p\) is enforced at lines 440--445; singular positive-semidefinite priors remain deliberately supported | **Fixed as declared** |
| Arm-name normalisation could alias two observations | Collision check at lines 411--416 | **Fixed** |
| Loop counts accepted booleans/fractions or leaked range errors | Public validation at lines 1113--1124 | **Fixed** |
| Negative nested estimates were silently changed to zero | Lines 1251--1263 warn and return the raw value; direct \(-0.5\) adversarial result confirmed | **Fixed**, with warning caveat below |
| Analytical off-centre/negative-slope accuracy was untested | Rust tests at `evsi_normal_normal.rs:25-34` cover both signs and match an independent reference | **Fixed** |
| Paper claimed the repaired estimator was in the cited release | `paper.md:138-141` and `paper.md:211-218` explicitly admit that v1.0.0 does not contain it and promise a future release | **Not fixed; now candid but still a submission blocker** |

## Defects and required actions

### N1 — Blocker: the reviewed software is not the software cited by the paper

**Evidence**

- The analytical implementation, its Rust tests, the v2 contract, and the main
  scientific-contract tests are untracked at the reviewed `HEAD`.
- `paper.md:138-141` says release 1.0.0 does not contain the analytical EVSI
  implementation or revised generic contract.
- `paper.md:211-218` cites release 1.0.0 and a Software Heritage snapshot, then
  says the submitted paper will cite a future release from the reviewed
  revision.
- `specs/v2/README.md:32-35` correctly states that the contract alone does not
  establish a published v2 release.

**Why it matters**

A JOSS reviewer installing the cited immutable release cannot reproduce the
software design and EVSI claims being reviewed. Prospective release wording is
not release evidence.

**Required action**

Commit the reviewed implementation and evidence, pass hosted and clean-install
checks, publish the exact reviewed release, archive that release, and replace
the prospective wording with the exact version, commit, release URL,
release-evidence manifest, and snapshot SWHID. Regenerate and rerun this review
against that immutable state.

### N2 — Major: analytical sample-size inputs violate the declared error contract

**Evidence**

- The public wrapper forwards `total_sample_size` without validating its Python
  type or range at `voiage/methods/sample_information.py:36-58`.
- PyO3 converts it to `usize` before entering the Rust kernel at
  `rust/crates/voiage-python/src/lib.rs:716-742`.
- `normal_normal_two_arm_evsi(..., total_sample_size=200.0)` raised raw
  `TypeError: 'float' object cannot be interpreted as an integer`.
- `normal_normal_two_arm_evsi(..., total_sample_size=-2)` raised raw
  `OverflowError: can't convert negative int to unsigned`.
- The v2 method contract at `specs/v2/stable-api.json:188-206` lists
  `InputError`, `BackendNotAvailableError`, and `NumericalError`, not raw
  conversion exceptions.
- The same class of leak occurs in the built-in generic route when a
  `DecisionOption` contains an integer too large to convert to float:
  `sample_information.py:543-545` raised raw `OverflowError` for \(10^{400}\).

**Why it matters**

This is a stable public-input boundary. Unsupported values should fail through
the documented typed diagnostics before foreign-function or floating-point
conversion.

**Required action**

Validate integer kind, boolean exclusion, positivity, parity, and supported
range in the public analytical wrapper. Bound or safely convert generic arm
sample sizes before constructing observation variances. Add adversarial tests
that assert package exception type and diagnostic identity.

### N3 — Moderate: the negative-estimate warning diagnoses only Monte Carlo size

**Evidence**

At `sample_information.py:1251-1257`, every negative two-loop result receives
the same instruction to increase loops and assess convergence. A deliberately
incoherent custom posterior returned exactly \(-0.5\) with that warning;
increasing loop counts cannot correct the callback model.

For the coherent built-in model, a negative point estimate is indeed evidence
of numerical error because population EVSI is non-negative. For the custom
route it can also indicate an incoherent simulator/posterior pair or a mismatch
between the empirical prior and callbacks.

**Required action**

Keep the raw-return behaviour. Amend the generic warning and documentation to
recommend checking prior/likelihood/posterior coherence as well as repeated
seeds, loop counts, and convergence. If desired, use a more specific warning
for the built-in coherent route.

### N4 — Minor evidence gap: negative semantics lack a direct assertion

The focused suite happens to emit warnings for negative estimates, and
`tests/test_v2_stable_api_contract.py:57-67` checks the metadata string, but no
test directly asserts that a known negative estimate is both returned
unchanged and accompanied by the expected diagnostic. The direct adversarial
calculation in this review supplied that evidence only outside the test suite.

Add a deterministic test for both the returned negative value and warning
identity. This protects the scientific choice against later reintroduction of
silent clipping.

## Test and tool evidence

The following commands passed on the scoped dirty worktree:

1. `uv run --extra ci --extra dev pytest
   tests/test_evsi_scientific_contract.py
   tests/test_sample_information.py
   tests/test_v2_stable_api_contract.py --no-cov -q`
   — **78 passed**.
2. `cargo test --manifest-path rust/Cargo.toml -p voiage-numerics
   --test evsi_normal_normal`
   — **4 passed**.
3. `cargo test --manifest-path rust/Cargo.toml -p voiage-numerics`
   — all unit, integration, property, differential, metamorphic, fuzz-style,
   thread-safety, and documentation tests passed.
4. `uv run python scripts/validate_joss.py`
   — repository-owned JOSS checks passed.

Passing repository checks do not resolve N1 because they ran against an
uncommitted dirty tree, nor N2 because the focused tests do not cover the raw
conversion exceptions.

## Manuscript claim audit

| `paper.md` lines | Claim | Assessment |
| ---: | --- | --- |
| 119--123 | Analytical study assumptions | Correct and consistent with the Rust formula |
| 123--128 | One fitted multivariate-normal prior used throughout | Correct for the repaired built-in path |
| 128--130 | Genuine Gaussian draws and untruncated negatives | Correct, but the diagnostic should also mention model coherence for custom callbacks |
| 131--134 | Custom joint update; analytical route stable; generic route developing | Substantively restrained, although “stable call signature” and “developing estimator maturity” should remain explicitly distinguished from the v2 API status |
| 136--141 | Current tests versus v1.0.0 contents | Candid and correct, but confirms the release blocker |
| 157--168 | Health-example numerical results and interpretation | Consistent with the declared analytical model and appropriately labelled synthetic |
| 211--218 | Release and Software Heritage evidence | Not submission-ready; the future-tense release is not the reviewed immutable artefact |

The paper does not overclaim the generic estimator as a universally validated
method. Its main numerical wording is now proportionate. The remaining
release-language defect is not a prose polish issue: it identifies missing
reproducibility evidence.

## Score

| Dimension | Score | Deduction |
| --- | ---: | --- |
| Scope, significance, and research use | 175/180 | Concrete developer research use and a useful worked decision are shown; independent use remains undocumented but is not disguised. |
| Statement of need and audience | 118/120 | Clear applied need; a little numerical-contract detail remains concentrated in the design section. |
| State of the field and build-versus-contribute case | 126/130 | Comparisons are fair; numerical depth relative to mature EVSI packages remains concise. |
| Scientific and numerical accuracy | 143/150 | Central equations, posterior conditioning, prior coherence, and reported example are correct. Deductions reflect nested-Monte-Carlo bias/uncertainty not being quantified and the over-specific custom-route warning. |
| Software design and research relevance | 97/100 | The shared analytical kernel and explicit callback model are defensible; stable API status and developing estimator maturity need consistently explicit separation. |
| Reproducibility, packaging, documentation, and tests | 84/100 | Strong focused evidence, but the reviewed implementation is untracked/unreleased and invalid sample-size conversions are untested. |
| Research-impact statement | 65/80 | The synthetic example and same-author integration are candid and useful, but realised independent impact is not yet evidenced. |
| Structure, metadata, and JOSS format | 60/60 | Numerical material is proportionate and the repository validator passes. |
| Clarity, accessibility, and sentence quality | 54/55 | The method boundaries are unusually clear; the negative-diagnostic wording needs one qualification. |
| Citations, provenance, declarations, and AI disclosure | 12/25 | The paper cites v1.0.0 and its snapshot while describing a future v2 implementation; exact reviewed release provenance is absent. |
| **Total** | **934/1000** | **Major revision; fail-closed cap applies.** |

## Fail-closed conclusion

The earlier scientific defects in prior coherence, Gaussian sampling,
correlation propagation, finite reduction, rank/collision/loop validation, and
negative-result truncation are fixed in the scoped working tree. The
normal--normal analytical equation and the paper's health-example value are
correct.

The review remains **fail closed** because the paper's cited release does not
contain the reviewed implementation and because the declared stable analytical
error boundary is not yet enforced. Resolve N1 and N2 before submission. Resolve
N3 and N4 before claiming complete numerical-review closure. A subsequent
review should use the immutable release and archive identifiers, rerun the
independent calculations, and treat the current 934/1000 as superseded rather
than carrying it forward.
