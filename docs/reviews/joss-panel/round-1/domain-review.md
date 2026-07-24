# Domain reviewer report

Role: Value-of-information, health-economics, trial-design, and
decision-science reviewer

Recommendation: major revisions

Score: **784/1000**

## Overall finding

The synthetic health example is methodologically coherent and its committed
numbers agree with analytical checks. The main blocker is that `paper.md`
presents expected value of sample information (EVSI) as a stable package
capability, while the valid normal-normal calculation used by the example is a
paper-specific function. Several public package EVSI estimators do not encode a
defensible sampling model.

This is a repository-level scientific finding, not only a wording problem.

## Material findings

### 1. Unqualified EVSI claims are unsupported

`paper.md` lines 34–39 state that `voiage` provides EVSI.

Evidence:

- `scripts/generate_paper_health_example.py:76` implements the example's valid
  normal-normal calculation outside the public package API.
- `voiage/methods/sample_information.py:319` treats each probabilistic
  sensitivity-analysis draw as a separate prior mean in the default two-loop
  path rather than conditioning a coherent prior distribution or reweighting
  posterior draws.
- `rust/crates/voiage-numerics/src/evsi_efficient.rs:133` and
  `rust/crates/voiage-numerics/src/evsi_moment.rs:105` derive an information
  fraction from `trial sample size / (trial sample size + number of PSA
  draws)`. The estimated information from a proposed study therefore changes
  when the analyst changes the Monte Carlo draw count.
- `rust/crates/voiage-numerics/src/evsi.rs:88` resamples net-benefit rows
  without a likelihood or posterior update. This measures variation in sample
  means, not EVSI for a specified study.

For a matching normal-normal example with total trial size 200, the analytical
EVSI is 124.18 per person. The reviewer obtained:

- public two-loop path: approximately 3,243–3,380;
- efficient estimator with 1,000 PSA draws: 94.09;
- efficient estimator with 2,000 draws: 45.96;
- efficient estimator with 10,000 draws: 4.44.

Required remediation:

- promote a specified, validated normal-normal model into the public API;
- replace, demote, or accurately label the current heuristic estimators;
- add analytical-reference, simulation-recovery, PSA-size-invariance, and
  likelihood-sensitivity tests;
- avoid a generic stable-EVSI claim until prespecified reference problems are
  recovered.

Interim manuscript wording:

> `voiage` provides EVPI, regression-based EVPPI, ENBS,
> cost-effectiveness acceptability frontiers, dominance analysis, and
> diagnostic reporting. The accompanying health example evaluates a proposed
> study using a separately specified normal-normal EVSI model. Other EVSI
> estimators remain subject to method-specific validation.

### 2. The impact statement overstates package use

Lines 118–122 say the synthetic study uses `voiage` to compare alternative
study sizes. It uses package EVPI, EVPPI, and ENBS, but obtains study-size EVSI
from the paper-specific analytical function.

Accurate wording before package remediation:

> The synthetic health example uses `voiage` for EVPI, EVPPI, and ENBS,
> together with an independently implemented normal-normal EVSI calculation
> for the proposed study.

### 3. The Summary is too technical

Lines 41–48 introduce binding-independent Rust components, domain contracts,
serialization, and stable surfaces. These details displace the scientific
purpose from the required non-specialist opening.

Suggested wording:

> The Python package is the primary user interface. Shared calculations and
> data rules are implemented in Rust so that selected results can also be
> reproduced from R and Julia. Version 1 provides the broadest functionality
> in Python; the current R and Julia packages directly share only the EVPI
> calculation.

### 4. Define perfect and sample information separately

Lines 30–31 define VOI only through collecting additional information. EVPI
and EVPPI concern hypothetical perfect resolution; EVSI concerns a specified
data-generating process.

Suggested wording:

> Value of Information analysis quantifies the expected improvement in a
> decision outcome from resolving uncertainty under a specified information
> scenario. Perfect-information measures describe hypothetical resolution of
> all or selected uncertainty; sample-information measures evaluate a
> particular proposed study.

The reviewer recommends the ISPOR task-force guidance:
Rothery et al. (2020), DOI `10.1016/j.jval.2020.01.004`.

### 5. Reframe the uncited fragmentation claim

Lines 52–58 assert practical consequences of fragmented tools without direct
evidence. A bounded design statement would be:

> Existing VOI workflows span specialist packages, web tools, programming
> languages, and model-output formats. Moving results between them requires
> analysts to preserve strategy labels, units, parameter groupings, population
> assumptions, and the distinction between perfect and sample information.

The reviewer recommends Tuffaha et al. on software for VOI analysis, DOI
`10.1007/s40258-021-00662-4`.

### 6. Restrict the R and Julia parity statement

Lines 65–68 imply a wider shared input and calculation surface than is
currently released.

Suggested wording:

> The current R and Julia packages call the same versioned Rust implementation
> for EVPI. Their wider VOI functionality is narrower than Python and is not
> claimed to have complete cross-language parity.

### 7. Make the build-versus-contribute case neutral

Lines 86–88 speculate that adding the project's features to another package
would change that package's purpose.

Suggested wording:

> `voiage` was developed for a requirement not covered by the compared
> health-economic tools: a versioned decision/result contract shared across
> Python, R, and Julia, while retaining explicit maturity and provenance
> metadata. Implementing this as an extension to an R-centred package would not
> provide the required language-neutral binary boundary. The trade-off is that
> `voiage` currently has less methodological depth than specialist R tools and
> substantially narrower R and Julia interfaces.

The comparison should also acknowledge the Python project
`tadamcz/value-of-information`, while accurately distinguishing its binary
decision and noisy-signal scope.

### 8. Distinguish software assurance from scientific correctness

Lines 110–114 call a broad test catalogue “Correctness evidence”. Software
tests do not alone prove that an estimator targets the intended estimand.

Suggested wording:

> Software-assurance evidence includes …

and:

> Analytical and simulation-reference tests are used where a trustworthy
> comparator exists; software tests do not by themselves establish the
> suitability of an estimator for a substantive decision problem.

### 9. Impact remains an eligibility risk

The paper correctly says its developer-created materials are not independent
adoption evidence. “Two concrete developer-led research uses” should instead
be “reproducible developer-created research materials”. Independent non-author
installation and research-use evidence remains externally pending in issue
#471.

### 10. Keep cross-domain claims bounded

General use of VOI outside health economics is supportable. The reviewer
identified environmental-management sources with DOI
`10.1111/2041-210X.12423` and DOI `10.3389/fenvs.2022.805245`. The software's
own non-health application remains prospective and should not be described as
empirically validated.

## Worked-example verification

The reviewer found:

- the equal-allocation, two-arm likelihood variance \(4\sigma^2/n\) is
  correct;
- the preposterior variance of the posterior mean is correct;
- holding cost at its prior mean is coherent when the study informs only the
  effect and the declared independence assumption holds;
- analytical EVPI is 652.18 and simulated EVPI is 644.15;
- analytical effect EVPPI is 598.41 and simulated effect EVPPI is 589.67;
- analytical cost EVPPI is 259.31 and simulated cost EVPPI is 249.59;
- each analytical value lies inside the reported bootstrap interval;
- EVSI rises from 63.12 at \(n=50\) to 275.92 at \(n=1,200\);
- immediate, full-uptake ENBS crosses zero between 100 and 200 participants;
- delayed, 60-percent-uptake ENBS crosses zero between 800 and 1,200
  participants;
- the sensitivity conclusions match the committed CSV evidence.

Five worked-example tests and seven JOSS-readiness tests passed. The
repository-owned validator checks structure, metadata, and citation keys; it
does not establish estimator validity.

## Rubric

| Dimension | Score |
| --- | ---: |
| Scope, significance, and research use | 142/180 |
| Statement of need and audience | 104/120 |
| State of the field and build-versus-contribute case | 100/130 |
| Scientific and numerical accuracy | 88/150 |
| Software design and research relevance | 86/100 |
| Reproducibility, packaging, documentation, and tests | 88/100 |
| Research-impact statement | 44/80 |
| Structure, metadata, and JOSS format | 58/60 |
| Clarity, accessibility, and sentence quality | 51/55 |
| Citations, provenance, declarations, and AI disclosure | 23/25 |
| **Total** | **784/1000** |

## External gates

- Issue #471 lacks independent non-author installation and research-use
  evidence.
- A permanent arXiv identifier remains externally pending but is not a JOSS
  requirement.
- JOSS screening, review, and acceptance are external editorial decisions.

The EVSI scientific defect is repository-owned and is not an external gate.
