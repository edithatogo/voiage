# Domain reviewer report: round 2

Role: independent value-of-information, health-economics, trial-design, and
decision-science reviewer

Reviewed state: working tree based on
`1dfcd3d42af591ca15fcc03ad958123cc153dbbf`, with uncommitted Round 2 changes

Recommendation: major revisions

Score: **886/1000**

Fail-closed status: **blocker present; score capped below 950**

## Overall finding

The principal Round 1 scientific defect has been addressed. The worked example
now calls a public analytical normal--normal EVSI implementation owned by the
Rust numerical crate and exposed through Python. The default two-loop path now
simulates a study, performs a conjugate known-variance update, consumes the
requested number of posterior draws, and can recover the analytical reference.
The regression, efficient, and moment-based compatibility paths emit
`FutureWarning` messages and are not presented as stable scientific contracts.

The reported health-example numbers are correct. Independent calculations
reproduced the EVPI, both EVPPI values, all six analytical EVSI values, the
discounted opportunity populations, and both ENBS sign changes.

The paper is not yet ready for submission. It describes the ENBS results as
being conditional on “stated population and cost assumptions”, but those
assumptions are absent from the manuscript. It also omits important restrictions
of the generic two-loop implementation, contains two incorrect author given
names in the foundational ISPOR citation, and does not yet identify an immutable
reviewed software revision. These are material scientific-reporting and
citation defects under the fail-closed rubric.

## Material findings

### 1. The worked-example assumptions are not stated

`paper.md` lines 124--126 report two ENBS thresholds under “the stated
population and cost assumptions”. The manuscript does not state the annual
population, time horizon, discount rate, study cost function, normal prior,
known outcome standard deviation, independence assumption, or which uncertainty
the study informs. Those values exist only in
`scripts/generate_paper_health_example.py`.

The omitted assumptions are:

- incremental health effect:
  \(\theta \sim N(0.060, 0.030^2)\);
- incremental programme cost:
  \(C \sim N(3000, 650^2)\), independent of \(\theta\);
- incremental net benefit:
  \(50{,}000\theta-C\);
- equal allocation between two arms;
- known individual outcome standard deviation of 1.0;
- the study informs the health effect but not programme cost;
- 1,300 affected people per year for ten years;
- annual discount rate of 3%;
- study cost of 1,200,000 plus 100 per participant;
- delayed scenario: evidence starts in year 3 and reaches 60% of the eligible
  population.

Required replacement after the reported EVPPI values:

> For the study analysis, the incremental health effect had a normal prior with
> mean 0.060 and standard deviation 0.030. The programme cost was independent of
> the health effect and remained uninformed by the study. The equal-allocation
> trial assumed a known individual outcome standard deviation of 1.0. Population
> ENBS used 1,300 eligible people per year for ten years, 3% annual discounting,
> and a study cost of 1,200,000 plus 100 per participant.

The delayed scenario should then state explicitly that usable evidence begins
after two years, corresponding to opportunities in years 3--10.

This defect is a manuscript blocker because a reader cannot reconstruct or
interpret the reported research decision from the paper.

### 2. The analytical normal--normal path is correct

For a two-arm equal-allocation study with total sample size \(n\), common known
individual outcome variance \(\sigma^2\), and uncertain incremental effect
\(\theta\sim N(\mu,\tau^2)\), the sampling variance of the difference in sample
means is

\[
s^2 = \frac{4\sigma^2}{n}.
\]

The preposterior variance of the posterior mean is

\[
\operatorname{Var}\{E(\theta\mid Y)\}
= \frac{\tau^4}{\tau^2+s^2}.
\]

For linear incremental net benefit \(a\theta+b\), the implementation correctly
evaluates the positive part of the resulting normal preposterior distribution
and subtracts the current-information decision value. The Rust code also
rejects non-finite inputs, non-positive standard deviations, zero sample sizes,
and odd total sample sizes.

The independent analytical result for the manuscript's \(n=200\) case is
`124.1793655206238`, exactly matching the Rust and Python public-path result to
the reported tolerance.

### 3. The corrected two-loop path recovers the reference, but its scope is
under-declared

The implementation now:

1. samples one true parameter row from the probabilistic sensitivity analysis;
2. simulates arm-level normal outcomes with a common known standard deviation;
3. fits a normal prior to each `mean_<arm>` marginal using the supplied prior
   sample;
4. applies the conjugate normal update separately to each informed arm mean;
5. draws `n_inner_loops` posterior samples; and
6. averages the posterior expected maximum net benefit over outer-loop studies.

Using 20,000 prior draws, 3,000 outer loops, 1,000 inner draws, and seed
`314159`, the reviewer obtained `123.33835758286374`, compared with the
analytical value `124.1793655206238`.

The implementation is coherent for the tested independent normal arm-mean
model. It is not a generic Bayesian update for arbitrary PSA samples. It
silently fits normal marginal priors and replaces informed arm means
independently, thereby discarding correlation among informed means and between
informed means and other parameters. The public documentation and stable API
scope should state:

> The default two-loop path currently supports independent normal priors for
> arm means named `mean_<arm>`, a common finite known `sd_outcome`, and a normal
> sampling model. It is not a general posterior updater for correlated or
> non-normal PSA inputs.

The scientific recovery test is useful but permissive: 400 outer by 400 inner
draws with an absolute tolerance of 35 admits substantial bias. Add either a
Monte Carlo standard-error criterion or several prespecified seeds and a tighter
aggregate criterion.

### 4. Compatibility warnings are present, with one wording defect

The Python dispatcher emits `FutureWarning` for `regression`, `efficient`, and
`moment_based`. This is a material improvement because the efficient and
moment-based implementations retain a heuristic information fraction that
depends on the number of PSA draws.

The warning for `regression` says that the estimator “does not specify a
likelihood or posterior update”, although `_evsi_regression()` calls the normal
trial simulator and conjugate updater. The more accurate warning is:

> The regression EVSI compatibility estimator does not expose a complete,
> validated study-model contract and is not a stable scientific estimator.

The paper's line 103 should likewise refer to compatibility estimators “without
a complete validated study-model contract”, rather than implying that every
warned path contains no likelihood code.

### 5. The health-example calculations are correct

Independent calculations gave:

| Quantity | Independent result | Manuscript or data result | Finding |
| --- | ---: | ---: | --- |
| Analytical probability programme preferred | 0.5000 | 0.4924 from 10,000 draws | Consistent with Monte Carlo error |
| Analytical EVPI | 652.1822 | 644.1535 | Inside reported bootstrap interval |
| Analytical health-effect EVPPI | 598.4134 | 589.6662 | Inside reported bootstrap interval |
| Analytical programme-cost EVPPI | 259.3125 | 249.5950 | Inside reported bootstrap interval |
| EVSI, \(n=50\) | 63.117286 | 63.117286 | Match |
| EVSI, \(n=100\) | 88.768918 | 88.768918 | Match |
| EVSI, \(n=200\) | 124.179366 | 124.179366 | Match |
| EVSI, \(n=400\) | 171.952831 | 171.952831 | Match |
| EVSI, \(n=800\) | 233.720375 | 233.720375 | Match |
| EVSI, \(n=1,200\) | 275.918834 | 275.918834 | Match |
| Immediate opportunity population | 11,089.2637 | Implied by data | Match |
| Delayed 60% opportunity population | 5,161.0519 | Implied by data | Match |

Immediate/full-uptake ENBS is `-225,618.06` at \(n=100\) and `157,057.73`
at \(n=200\). Delayed/60%-uptake ENBS is `-73,757.03` at \(n=800\) and
`104,031.41` at \(n=1,200\). Both manuscript sign-change statements are
therefore correct.

The EVPPI calculation is linear ordinary least squares. It is exact for the
worked example's linear net-benefit model in the population limit; the finite
simulation differences from the closed-form values are consistent with the
reported bootstrap intervals.

### 6. The state-of-the-field review remains incomplete

The addition of `dampack`, correction of the `voi` author roles, and direct
SAVI citation improve the comparison. Two Round 1 requests remain unresolved:

- Tuffaha et al., “A Review of Web-Based Tools for Value-of-Information
  Analysis”, DOI `10.1007/s40258-021-00662-4`, directly supports the fragmented
  web-tool context and should be cited in the statement of need or state of the
  field.
- `tadamcz/value-of-information` is a directly relevant Python implementation.
  It addresses a narrower binary decision with a one-dimensional noisy signal,
  but omitting it makes the build-versus-contribute comparison appear
  R-centred by construction.

The comparison need not imply that the Python project is a mature or equivalent
alternative. One bounded sentence is sufficient:

> A separate Python project evaluates a narrower binary decision with a
> one-dimensional noisy signal; `voiage` instead represents labelled
> multi-strategy probabilistic models and health-economic perfect- and
> sample-information measures.

### 7. One cited record contains incorrect author names

The `rothery2020voi` BibTeX record lists “John F. Murray” and “Gary D. Sanders
Schmidler”. Crossref, PubMed (`PMID 32197720`), and the article identify
**James F. Murray** and **Gillian D. Sanders Schmidler**. Correct both names
before submission.

SourceRight validated a disposable CSL conversion after DOI normalisation and
assigning explicit types to converted `@misc` records. The unnormalised Pandoc
conversion produced expected canonicalisation diagnostics. SourceRight's
structural pass does not supersede the external metadata discrepancy above.

### 8. Several package claims remain broader than the scalar APIs demonstrate

Lines 39--41 say that `voiage` keeps labels, units, draws, assumptions,
warnings, and provenance “with the analysis” while calculating the four named
measures. The main `evpi()`, `evppi()`, `evsi()`, and `enbs()` functional APIs
return scalar floats. Other repository objects and metadata contracts preserve
some of the listed information, but the sentence reads as a guarantee about
every calculation result.

Replace it with:

> `voiage` calculates these measures from probabilistic decision models.
> Labelled decision objects and separate metadata records can retain strategy
> names, units, study and population assumptions, warnings, and provenance for
> review.

Similarly, line 57 should replace “rejects malformed or inconsistent inputs”
with the bounded “validates specified shapes, finite values, labels, and
method-specific assumptions before calculation”.

### 9. The impact statement is honest but should use “worked example”

The paper correctly states that both demonstrations were created by the author
and are not independent-adoption evidence. “Synthetic research demonstration”
still sounds stronger than the evidence. Use “synthetic worked example”.

Line 127 should use “illustrate” rather than “show”:

> These synthetic results illustrate how the source of uncertainty, study size,
> implementation delay, uptake, population, and research cost can alter the
> estimated value of research under the declared model.

Attributable non-author installation or research use remains externally
pending. It is an eligibility/readiness issue rather than a defect that can be
repaired by changing the manuscript.

### 10. The reviewed revision is not immutable

The paper says that the exact reviewed revision and release-evidence manifest
“will be frozen before submission”. The reviewed implementation currently
exists as uncommitted changes on top of
`1dfcd3d42af591ca15fcc03ad958123cc153dbbf`; it is not part of v1.0.0.

Before JOSS submission, replace the future-tense sentence with the exact commit
or release identifier reviewed, and bind the paper, source archive, SBOM, test
evidence, and release-evidence manifest to that identifier. The current
v1.0.0 citation cannot by itself support claims about the newly corrected EVSI
implementation.

## Executed checks

The reviewer executed:

```console
cargo test --manifest-path rust/Cargo.toml \
  -p voiage-numerics --test evsi_normal_normal
```

Result: 3 passed.

```console
uv run --extra ci --extra dev pytest \
  tests/test_evsi_scientific_contract.py --no-cov -q
```

Result: 5 passed.

```console
uv run --extra ci --extra dev pytest \
  tests/test_paper_health_example.py --no-cov -q
```

Result: 5 passed.

```console
uv run --extra ci python scripts/validate_joss.py
uv run --extra ci pytest tests/test_joss_readiness.py --no-cov -q
git diff --check
```

Result: repository-owned JOSS validation passed; 7 readiness tests passed;
`git diff --check` found no whitespace errors. These checks establish internal
contracts, not JOSS acceptance or scientific validity.

The paper renders to 1,238 words including references and remains within the
rubric's stated word range.

## Scientific sentence audit

“Verified” means that the sentence is supported by the inspected implementation,
data, cited source, or independent calculation. “Bounded” means that it is an
explicit purpose, trade-off, or limitation rather than an empirical finding.
“Revise” identifies an unresolved scientific, evidential, or overclaiming issue.

| ID | Lines | Verdict | Audit |
| ---: | ---: | --- | --- |
| 1 | 30--31 | Verified | The VOI definition is accurate and supported by Rothery et al. |
| 2 | 31--34 | Verified | The four practical questions map appropriately to the four measures. |
| 3 | 35--37 | Verified | EVPI, EVPPI, EVSI, and ENBS are expanded and distinguished correctly. |
| 4 | 39--41 | Revise | The scalar functional APIs do not themselves retain every listed item with every result. |
| 5 | 41--42 | Verified | Python is the broadest current interface. |
| 6 | 42--43 | Verified | Rust owns shared rules and selected calculations, including EVPI and analytical normal--normal EVSI. |
| 7 | 43--44 | Verified | The R and Julia packages expose a narrower shared EVPI surface. |
| 8 | 44--45 | Bounded | This is a design intention; the preceding wording should be narrowed first. |
| 9 | 49--51 | Revise | The multi-tool observation is plausible but should cite the Tuffaha web-tool review. |
| 10 | 51--52 | Verified | Moving a labelled decision analysis requires more than copying a numerical array. |
| 11 | 52--54 | Verified | The listed items affect interpretation; population, horizon, delay, uptake, and design can also alter numerical results. |
| 12 | 54--55 | Bounded | This is a reasonable risk statement, not an observed error-rate claim. |
| 13 | 57--58 | Revise | “Rejects malformed or inconsistent inputs” is absolute and broader than the enumerated validators. |
| 14 | 58--60 | Bounded | The audience and intended uses are stated as intentions. |
| 15 | 60--63 | Bounded | The structure is domain-neutral, but only the health application is demonstrated; retain prospective wording for other domains. |
| 16 | 67--68 | Verified | The cited literature establishes VOI in decision analysis and health economics. |
| 17 | 69--70 | Verified | CRAN `voi` 1.0.3 describes EVPI, EVPPI, EVSI, ENBS, and multiple methods. |
| 18 | 70--71 | Verified | The BCEA description and citation are accurate. |
| 19 | 71--72 | Verified | The `dampack` description and author list agree with CRAN metadata. |
| 20 | 72--74 | Verified | SAVI 2.2.1 provides overall EVPI and regression-based EVPPI. |
| 21 | 74--76 | Bounded | The suitability statement is appropriately conditional. |
| 22 | 78--80 | Revise | The preservation claim is broader than the scalar result paths and needs the replacement in finding 8. |
| 23 | 80--82 | Bounded | An R-only extension would not itself create the chosen language-neutral binary boundary. |
| 24 | 81--82 | Bounded | Independent implementations do require separate parity maintenance. |
| 25 | 82--84 | Verified | The method-depth and R/Julia limitations are candid and consistent with the repository. |
| 26 | 88--89 | Verified | The architecture separates selected shared calculations from language-specific workflows. |
| 27 | 89--90 | Verified | Rust implements common types, validators, and selected numerical kernels. |
| 28 | 90--91 | Verified | Python owns labelled data and user-defined model orchestration. |
| 29 | 91--93 | Verified | R and Julia call shared EVPI and do not reproduce the full Python interface. |
| 30 | 93--95 | Bounded | The stated maintenance benefit and packaging cost are credible design trade-offs. |
| 31 | 97--98 | Verified | The four maturity states match the governed metadata taxonomy. |
| 32 | 98--99 | Verified with qualification | Separate metadata contracts disclose approximation and backend status, but scalar results do not automatically carry the envelope. |
| 33 | 99--101 | Verified | The warning against universal suitability is scientifically appropriate. |
| 34 | 101--103 | Verified | The analytical normal--normal path declares and implements the listed assumptions. |
| 35 | 103--104 | Revise | The regression path contains a hard-coded likelihood/update; use “without a complete validated study-model contract”. |
| 36 | 106--108 | Verified | Tests cover the listed software-assurance categories. |
| 37 | 108--110 | Verified | The repository contains the stated test types and hosted workflows. |
| 38 | 110--112 | Verified | The public v1.0.0 release contains the stated distribution files and checksums. |
| 39 | 112--113 | Verified with boundary | Attestations exist and a separate workflow generates an SBOM; the v1.0.0 release itself does not contain an SBOM asset. |
| 40 | 117--118 | Revise | Use “synthetic worked example”, not “synthetic research demonstration”. |
| 41 | 118--120 | Verified | The synthetic comparison is accurately described. |
| 42 | 120--121 | Verified | `0.4924` matches the fixed-seed output and is compatible with the analytical probability 0.5. |
| 43 | 121--122 | Verified | EVPI 644.15, effect EVPPI 589.67, and cost EVPPI 249.59 match the data and independent checks. |
| 44 | 122--123 | Verified | Analytical normal--normal EVSI at \(n=200\) is 124.1793655 per person. |
| 45 | 123--125 | Revise | The sign change is correct, but the population and cost assumptions are not stated in the manuscript. |
| 46 | 125--126 | Revise | The delayed sign change is correct, but the delay timing, eligible population, horizon, and cost assumptions are absent. |
| 47 | 126--128 | Revise | “Illustrate under the declared model” is more proportionate than “show”. |
| 48 | 130--132 | Verified | The versioned integration bundle contains decision records, provenance, schema identifiers, and expected results. |
| 49 | 132--133 | Verified | The author-created status and non-adoption boundary are explicit. |
| 50 | 133--134 | Verified | Git history begins in July 2025. |
| 51 | 134--135 | Verified as a boundary | No attributable non-author research use is documented in the inspected evidence. |
| 52 | 157 | Verified with revision boundary | The Python package and v1.0.0 release are public, but v1.0.0 predates the Round 2 EVSI corrections. |
| 53 | 157--159 | Verified | The script uses fixed seeds and the committed outputs are synthetic. |
| 54 | 159--161 | Verified | The Software Heritage snapshot resolves as `swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`. |
| 55 | 161--162 | Revise | Future freezing is not evidence of the exact reviewed revision; replace with the immutable identifier before submission. |

No unresolved promotional superlative or novelty claim was found. Authentext-style
claim review identified three avoidable strength problems: the absolute
validation claim, “research demonstration”, and “show” for synthetic results.

## Rubric

| Dimension | Score | Deductions |
| --- | ---: | --- |
| Scope, significance, and research use | 160/180 | The software is in scope and the author-created evidence is accurately bounded, but realised non-author research use is absent and non-health use remains prospective. |
| Statement of need and audience | 112/120 | Clear applied questions and audience; deduct for the uncited multi-tool context and overbroad preservation/validation language. |
| State of the field and build-versus-contribute case | 108/130 | Good R and SAVI comparison and candid trade-off; deduct for omitting the directly relevant web-tool review and narrower Python implementation. |
| Scientific and numerical accuracy | 132/150 | Analytical Rust path, two-loop recovery, warnings, and all worked-example numbers verify; deduct for unstated assumptions, under-declared two-loop restrictions, and inaccurate regression-warning wording. |
| Software design and research relevance | 95/100 | The boundary and trade-off are clear; deduct because metadata and provenance wording exceeds what scalar APIs automatically return. |
| Reproducibility, packaging, documentation, and tests | 91/100 | Focused scientific, Rust, example, and JOSS checks pass; deduct because the corrected implementation is uncommitted and not bound to a reviewed release or manifest. |
| Research-impact statement | 61/80 | Concrete reproducible examples and honest non-adoption boundary; deduct for “research demonstration”, prospective cross-domain impact, and no independent research use. |
| Structure, metadata, and JOSS format | 58/60 | Required sections and word range pass; a compact assumptions table or figure reference would materially improve the worked example. |
| Clarity, accessibility, and sentence quality | 52/55 | Generally plain and restrained; deduct for “stated” assumptions that are absent, one absolute validator claim, and “show” for a synthetic illustration. |
| Citations, provenance, declarations, and AI disclosure | 17/25 | Citation keys resolve and declarations are complete; deduct for two incorrect author given names, the missing Tuffaha source, the omitted Python comparator, and the unfrozen reviewed revision. |
| **Total** | **886/1000** | |

The incorrect foundational citation record, missing worked-example assumptions,
and unresolved reviewed revision each satisfy a fail-closed condition. The
Round 2 domain score therefore cannot exceed 950 even though the numerical
implementation now passes the reference problem.

## Required disposition before round 3

1. State all worked-example distributional, study, population, discounting,
   uptake, delay, and cost assumptions in the manuscript.
2. Correct James F. Murray and Gillian D. Sanders Schmidler in
   `rothery2020voi`.
3. Cite the Tuffaha web-tool review and compare the narrower Python
   `value-of-information` project.
4. Narrow the provenance and input-validation claims.
5. Document the independent-normal-arm scope and correlation limitation of the
   generic two-loop path.
6. Correct the regression compatibility-warning wording.
7. Replace “research demonstration” and “show” with the bounded language above.
8. Commit the corrected implementation, bind the manuscript and evidence to an
   immutable revision, and rerun the analytical, simulation, citation, render,
   and clean-install checks at that revision.

## External gates

- Attributable non-author installation or research-use evidence remains pending
  under issue #471.
- JOSS screening, reviewer assignment, review, acceptance, and DOI assignment
  are external editorial outcomes.
- These external gates do not invalidate the corrected calculations, and they
  must not be represented as complete through manuscript wording.
